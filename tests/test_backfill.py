"""Partitioned layers, date windows, and what a re-run is allowed to destroy.

The property under test is narrow and easy to lose: a run must replace the days
it produced and nothing else. Every test here is really the same question asked
of a different layer -- what is still on disk afterwards?
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from tripetl.config import PipelineConfig
from tripetl.pipeline import NothingToPublish, run_pipeline
from tripetl.quality.gate import QualityGateFailed
from tripetl.schema import PARTITION_COLUMN
from tripetl.sources import SAMPLE_DAYS, SAMPLE_EPOCH

pytestmark = pytest.mark.slow

ROWS = 300

#: The dates the generator covers, as the partition values they become.
FIRST_DAY = SAMPLE_EPOCH.date()
LAST_DAY = date.fromordinal(FIRST_DAY.toordinal() + SAMPLE_DAYS - 1)
MIDDLE_DAY = date.fromordinal(FIRST_DAY.toordinal() + 3)


def _config(tmp_path: Path, **overrides) -> PipelineConfig:
    """A pristine sample: every gate passes, so the assertions are about layout."""
    defaults = {
        "warehouse": str(tmp_path / "warehouse"),
        "sample_rows": ROWS,
        "sample_seed": 41,
        "sample_dirty_rate": 0.0,
        "shuffle_partitions": 2,
    }
    return PipelineConfig(**{**defaults, **overrides})


def _partitions(path: str) -> set[str]:
    """The partition values a dataset has on disk, read off the directory names."""
    root = Path(path)
    if not root.exists():
        return set()
    return {
        entry.name.split("=", 1)[1]
        for entry in root.iterdir()
        if entry.is_dir() and entry.name.startswith(f"{PARTITION_COLUMN}=")
    }


def _gold_by_day(spark, config: PipelineConfig) -> dict[str, list[tuple]]:
    """The published mart keyed by day, so days can be compared one at a time."""
    rows = spark.read.parquet(config.gold_path).collect()
    by_day: dict[str, list[tuple]] = {}
    for row in rows:
        key = row[PARTITION_COLUMN].isoformat()
        by_day.setdefault(key, []).append((row["pu_location_id"], row["trips"]))
    return {day: sorted(values) for day, values in by_day.items()}


# -- layout -------------------------------------------------------------------


@pytest.fixture(scope="module")
def published(tmp_path_factory, spark):
    """One full run over the generator's whole week."""
    config = _config(tmp_path_factory.mktemp("backfill"))
    return config, run_pipeline(config, spark=spark)


def test_every_layer_is_partitioned_by_pickup_date(published):
    config, _ = published
    expected = {FIRST_DAY.isoformat(), LAST_DAY.isoformat()}

    for path in (config.bronze_path, config.silver_path, config.gold_path):
        partitions = _partitions(path)
        assert expected <= partitions, path
        assert len(partitions) == SAMPLE_DAYS, path


def test_the_partition_column_survives_the_round_trip(published, spark):
    """Partition values live in the path, so they have to read back as dates."""
    config, _ = published
    gold = spark.read.parquet(config.gold_path)

    days = {row[PARTITION_COLUMN] for row in gold.select(PARTITION_COLUMN).distinct().collect()}

    assert dict(gold.dtypes)[PARTITION_COLUMN] == "date"
    assert days == {
        date.fromordinal(FIRST_DAY.toordinal() + offset) for offset in range(SAMPLE_DAYS)
    }


def test_quarantined_rows_are_partitioned_too(tmp_path, spark):
    """A backfill that rebuilds a day should rebuild that day's evidence with it."""
    config = _config(tmp_path, sample_dirty_rate=0.05, sample_rows=400)
    run_pipeline(config, spark=spark)

    assert _partitions(config.quarantine_path("bronze"))


def test_partitioning_can_be_turned_off(tmp_path, spark):
    config = _config(tmp_path, partition_by=())
    run_pipeline(config, spark=spark)

    assert _partitions(config.gold_path) == set()
    assert spark.read.parquet(config.gold_path).count() > 0


# -- windows ------------------------------------------------------------------


def test_a_window_narrows_the_rows_a_run_reads(tmp_path, spark):
    config = _config(tmp_path, since=MIDDLE_DAY, until=MIDDLE_DAY)
    result = run_pipeline(config, spark=spark)

    assert _partitions(config.bronze_path) == {MIDDLE_DAY.isoformat()}
    assert _partitions(config.gold_path) == {MIDDLE_DAY.isoformat()}
    # The report describes the window, not the file the window was cut from.
    assert result.stage("bronze").report.total_rows < ROWS


def test_an_open_ended_window_runs_to_the_end_of_the_data(tmp_path, spark):
    config = _config(tmp_path, since=LAST_DAY)
    run_pipeline(config, spark=spark)

    assert _partitions(config.gold_path) == {LAST_DAY.isoformat()}


def test_a_window_that_ends_before_it_starts_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="since must not be after until"):
        _config(tmp_path, since=LAST_DAY, until=FIRST_DAY)


def test_a_window_naming_days_the_input_does_not_have_blocks(tmp_path, spark):
    """An empty window is a row-count failure, which is what that rule is for."""
    absent = date.fromordinal(LAST_DAY.toordinal() + 30)
    config = _config(tmp_path, since=absent, until=absent)

    with pytest.raises(QualityGateFailed) as excinfo:
        run_pipeline(config, spark=spark)

    assert excinfo.value.report.stage == "bronze"
    assert excinfo.value.report.total_rows == 0


def test_an_empty_window_publishes_nothing_even_with_the_gates_off(tmp_path, spark):
    """--no-gates downgrades rule verdicts; it cannot conjure rows to publish."""
    absent = date.fromordinal(LAST_DAY.toordinal() + 30)
    config = _config(tmp_path, since=absent, until=absent, enforce_gates=False)

    with pytest.raises(NothingToPublish) as excinfo:
        run_pipeline(config, spark=spark)

    assert excinfo.value.stage == "bronze"
    assert not Path(config.gold_path).exists()


# -- what a re-run destroys ---------------------------------------------------


def test_a_backfill_replaces_only_the_day_it_names(tmp_path, spark):
    """The point of the whole exercise: rebuild one day, keep the other six."""
    config = _config(tmp_path)
    run_pipeline(config, spark=spark)
    before = _gold_by_day(spark, config)
    assert len(before) == SAMPLE_DAYS

    # A different seed means genuinely different trips for the day rebuilt --
    # if the backfill leaked outside its window the other days would move too.
    backfill = _config(tmp_path, sample_seed=99, since=MIDDLE_DAY, until=MIDDLE_DAY)
    run_pipeline(backfill, spark=spark)
    after = _gold_by_day(spark, config)

    assert set(after) == set(before), "a backfill must not remove days it did not name"
    for day, rows in before.items():
        if day != MIDDLE_DAY.isoformat():
            assert after[day] == rows, f"{day} was rewritten by a backfill that did not name it"
    assert after[MIDDLE_DAY.isoformat()] != before[MIDDLE_DAY.isoformat()]


def test_rerunning_the_same_window_changes_nothing(tmp_path, spark):
    """A retried task must replace its day, not append a second copy of it."""
    config = _config(tmp_path)
    run_pipeline(config, spark=spark)
    before = _gold_by_day(spark, config)

    rerun = _config(tmp_path, since=MIDDLE_DAY, until=MIDDLE_DAY)
    run_pipeline(rerun, spark=spark)

    assert _gold_by_day(spark, config) == before


def test_an_unpartitioned_backfill_takes_the_other_days_with_it(tmp_path, spark):
    """The failure mode partitioning exists to prevent, pinned so it stays fixed."""
    config = _config(tmp_path, partition_by=())
    run_pipeline(config, spark=spark)
    assert len(_gold_by_day(spark, config)) == SAMPLE_DAYS

    backfill = _config(tmp_path, partition_by=(), since=MIDDLE_DAY, until=MIDDLE_DAY)
    run_pipeline(backfill, spark=spark)

    assert list(_gold_by_day(spark, config)) == [MIDDLE_DAY.isoformat()]
