"""End-to-end stage behaviour against a temporary warehouse."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tripetl.config import PipelineConfig
from tripetl.pipeline import NothingToPublish, run_bronze, run_gold, run_pipeline, run_silver
from tripetl.quality.drift import SchemaDriftError
from tripetl.quality.engine import FAILURE_COLUMN
from tripetl.quality.gate import QualityGateFailed
from tripetl.quality.history import ReportHistory, compare_runs
from tripetl.sources import generate_sample

pytestmark = pytest.mark.slow

ROWS = 400


def _config(tmp_path: Path, **overrides) -> PipelineConfig:
    defaults = {
        "warehouse": str(tmp_path / "warehouse"),
        "sample_rows": ROWS,
        "sample_seed": 11,
        "sample_dirty_rate": 0.04,
        "shuffle_partitions": 2,
    }
    return PipelineConfig(**{**defaults, **overrides})


@pytest.fixture(scope="module")
def completed(tmp_path_factory, spark):
    """One full run, shared by the assertions that only inspect its output."""
    tmp_path = tmp_path_factory.mktemp("pipeline")
    config = _config(tmp_path)
    return config, run_pipeline(config, spark=spark)


def test_all_three_stages_run_and_pass(completed):
    _, result = completed
    assert [stage.stage for stage in result.stages] == ["bronze", "silver", "gold"]
    assert result.passed
    assert result.gold_rows > 0


def test_row_counts_shrink_through_the_layers(completed):
    _, result = completed
    bronze = result.stage("bronze")
    silver = result.stage("silver")

    assert bronze.rows_in == ROWS + int(ROWS * 0.04 / 2)
    assert bronze.rows_out == bronze.rows_in - bronze.rows_quarantined
    # Silver's input is bronze's output; dedupe removes the duplicated trips.
    assert silver.rows_in == bronze.rows_out
    assert silver.rows_out <= silver.rows_in


def test_bad_rows_are_quarantined_with_their_reasons(completed, spark):
    config, result = completed
    bronze = result.stage("bronze")
    assert bronze.rows_quarantined > 0

    quarantined = spark.read.parquet(config.quarantine_path("bronze"))
    assert quarantined.count() == bronze.rows_quarantined
    assert FAILURE_COLUMN in quarantined.columns
    # Every quarantined row names at least one rule.
    assert all(len(row[FAILURE_COLUMN]) >= 1 for row in quarantined.collect())


def test_each_stage_writes_its_report(completed):
    config, _ = completed
    for stage in ("bronze", "silver", "gold"):
        payload = json.loads(Path(config.report_path(stage)).read_text())
        assert payload["stage"] == stage
        assert payload["passed"] is True
        assert payload["results"]


def test_the_published_mart_is_readable(completed, spark):
    config, result = completed
    gold = spark.read.parquet(config.gold_path)
    assert gold.count() == result.gold_rows
    assert "median_duration_min" in gold.columns


def test_a_failing_gate_stops_the_run(tmp_path, spark):
    config = _config(tmp_path, sample_dirty_rate=0.6)

    with pytest.raises(QualityGateFailed) as excinfo:
        run_pipeline(config, spark=spark)

    assert excinfo.value.report.stage == "bronze"
    assert excinfo.value.report.errors


def test_a_failing_gate_still_leaves_its_report_behind(tmp_path, spark):
    """The evidence must survive the failure, or the only way to find out what
    happened is to re-run the job against data that may have moved on."""
    config = _config(tmp_path, sample_dirty_rate=0.6)

    with pytest.raises(QualityGateFailed):
        run_pipeline(config, spark=spark)

    payload = json.loads(Path(config.report_path("bronze")).read_text())
    assert payload["passed"] is False
    assert payload["error_count"] > 0


def test_a_failing_gate_publishes_nothing(tmp_path, spark):
    config = _config(tmp_path, sample_dirty_rate=0.6)

    with pytest.raises(QualityGateFailed):
        run_pipeline(config, spark=spark)

    assert not Path(config.gold_path).exists()


def test_gates_can_be_downgraded_to_reporting(tmp_path, spark):
    """`enforce_gates=False` surveys a bad extract end to end."""
    config = _config(tmp_path, sample_dirty_rate=0.6, enforce_gates=False)
    result = run_pipeline(config, spark=spark)

    assert not result.passed
    assert result.stage("bronze").report.errors
    # It still published, which is exactly why this is not the default.
    assert result.gold_rows > 0


def test_quarantine_can_be_disabled(tmp_path, spark):
    config = _config(tmp_path, quarantine_enabled=False)
    bronze = run_bronze(config, spark=spark)

    assert bronze.rows_quarantined == 0
    assert bronze.rows_out == bronze.rows_in


def test_stages_are_runnable_independently(tmp_path, spark):
    """The property Airflow depends on: each task is its own entry point."""
    config = _config(tmp_path)

    bronze = run_bronze(config, spark=spark)
    silver = run_silver(config, spark=spark)
    gold = run_gold(config, spark=spark)

    assert bronze.rows_out > 0
    assert silver.rows_in == bronze.rows_out
    assert gold.rows_in == silver.rows_out
    assert gold.rows_out > 0


def test_a_rerun_is_idempotent(tmp_path, spark):
    """Reprocessing the same input must not duplicate published rows."""
    config = _config(tmp_path)
    first = run_pipeline(config, spark=spark)
    second = run_pipeline(config, spark=spark)

    assert first.gold_rows == second.gold_rows
    assert first.stage("silver").rows_out == second.stage("silver").rows_out


def _write_input(spark, tmp_path: Path, *, drop: str | None = None) -> str:
    """A raw extract on disk, optionally missing a column, for the input path."""
    path = str(tmp_path / "input")
    raw = generate_sample(spark, rows=ROWS, seed=11, dirty_rate=0.0)
    if drop is not None:
        raw = raw.drop(drop)
    raw.write.parquet(path)
    return path


def test_a_clean_input_passes_the_schema_check_and_runs(tmp_path, spark):
    input_path = _write_input(spark, tmp_path)
    config = _config(tmp_path, input_path=input_path)

    result = run_pipeline(config, spark=spark)

    assert result.passed
    diff = json.loads(Path(config.schema_diff_path).read_text())
    assert diff["conforms"] is True


def test_a_dropped_input_column_stops_the_run_before_bronze(tmp_path, spark):
    """The point of the check: fail naming the missing column, not a stage of
    rules failing at 0% on a column that reads as nulls."""
    input_path = _write_input(spark, tmp_path, drop="fare_amount")
    config = _config(tmp_path, input_path=input_path)

    with pytest.raises(SchemaDriftError) as excinfo:
        run_pipeline(config, spark=spark)

    assert any(drift.column == "fare_amount" for drift in excinfo.value.diff.blocking)
    # Stopped before bronze: no stage report, no published mart.
    assert not Path(config.report_path("bronze")).exists()
    assert not Path(config.gold_path).exists()


def test_the_schema_diff_is_recorded_even_when_it_blocks(tmp_path, spark):
    input_path = _write_input(spark, tmp_path, drop="fare_amount")
    config = _config(tmp_path, input_path=input_path)

    with pytest.raises(SchemaDriftError):
        run_pipeline(config, spark=spark)

    diff = json.loads(Path(config.schema_diff_path).read_text())
    assert diff["conforms"] is False
    assert diff["blocking_count"] >= 1


def test_the_schema_check_can_be_downgraded_with_the_gates(tmp_path, spark):
    """enforce_gates=False surveys a drifted extract instead of stopping at it."""
    input_path = _write_input(spark, tmp_path, drop="fare_amount")
    config = _config(tmp_path, input_path=input_path, enforce_gates=False)

    # The dropped column reads as nulls, so every row breaks the bronze
    # non_negative/in_range rules on fare_amount and every row is quarantined.
    # The run therefore stops at bronze having nothing to publish -- which is
    # already past the schema check, so the check did not raise.
    with pytest.raises(NothingToPublish) as excinfo:
        run_pipeline(config, spark=spark)

    assert excinfo.value.stage == "bronze"
    assert Path(config.schema_diff_path).exists()
    # The report still describes what was read, gates or no gates.
    payload = json.loads(Path(config.report_path("bronze")).read_text())
    assert payload["passed"] is False


def test_the_schema_check_can_be_turned_off_independently(tmp_path, spark):
    input_path = _write_input(spark, tmp_path, drop="fare_amount")
    config = _config(tmp_path, input_path=input_path, check_input_schema=False)

    # No SchemaDriftError -- the check is skipped -- so the run proceeds to the
    # bronze gate, which then blocks on the all-null fare column.
    with pytest.raises(QualityGateFailed) as excinfo:
        run_pipeline(config, spark=spark)
    assert excinfo.value.report.stage == "bronze"
    assert not Path(config.schema_diff_path).exists()


def test_a_generated_run_writes_no_schema_diff(completed):
    """No input_path means nothing to diff; the sample is on-schema by construction."""
    config, _ = completed
    assert not Path(config.schema_diff_path).exists()


def test_unknown_stage_lookup_raises(completed):
    _, result = completed
    with pytest.raises(KeyError, match="platinum"):
        result.stage("platinum")


def test_pipeline_summary_mentions_the_verdict(completed):
    _, result = completed
    text = result.to_text()
    assert "PASSED" in text
    assert "bronze" in text and "gold" in text


# -- report history ---------------------------------------------------------


def test_a_run_records_every_stage_under_one_run_id(completed):
    """Three stages, three files, one name -- otherwise a run is unreconstructable."""
    config, _ = completed
    history = ReportHistory(config.history_dir)

    assert history.stages() == ("bronze", "gold", "silver")
    run_ids = {history.entries(stage)[-1].run_id for stage in history.stages()}
    assert len(run_ids) == 1


def test_a_recorded_report_matches_the_one_written_at_the_stage_boundary(completed):
    config, result = completed
    (entry,) = ReportHistory(config.history_dir).entries("bronze")
    assert entry.report == result.stage("bronze").report


def test_history_can_be_turned_off(tmp_path, spark):
    config = _config(tmp_path, keep_history=False)
    run_pipeline(config, spark=spark)
    assert not Path(config.history_dir).exists()


def test_a_blocked_run_still_records_its_history(tmp_path, spark):
    """The failed run is exactly the one whose predecessors you go looking for."""
    config = _config(tmp_path, sample_dirty_rate=0.6)
    with pytest.raises(QualityGateFailed):
        run_pipeline(config, spark=spark)

    (entry,) = ReportHistory(config.history_dir).entries("bronze")
    assert not entry.report.passed


def test_a_second_run_becomes_a_comparable_trend(tmp_path, spark):
    clean_config = _config(tmp_path, sample_seed=21)
    run_pipeline(clean_config, spark=spark)
    run_pipeline(
        _config(tmp_path, sample_seed=22, sample_dirty_rate=0.2, enforce_gates=False), spark=spark
    )

    trend = compare_runs(ReportHistory(clean_config.history_dir), "bronze")

    assert trend is not None
    assert trend.previous_run_id != trend.current_run_id
    # Six times the defect rate moves several rules well past the tolerance.
    assert not trend.stable


def test_history_is_pruned_to_the_configured_limit(tmp_path, spark):
    # Gates off: this is about retention, and whether a given seed happens to
    # trip a threshold on 400 rows is beside the point.
    for seed in (31, 32, 33):
        run_pipeline(
            _config(tmp_path, sample_seed=seed, history_limit=2, enforce_gates=False),
            spark=spark,
        )

    entries = ReportHistory(_config(tmp_path).history_dir).entries("bronze")
    assert len(entries) == 2


def test_an_explicit_run_id_is_used_verbatim(tmp_path, spark):
    """What Airflow passes, so a cleared task overwrites rather than duplicates."""
    config = _config(tmp_path, run_id="manual__2026-08-05T06:00:00+00:00")
    run_pipeline(config, spark=spark)
    run_pipeline(config, spark=spark)

    entries = ReportHistory(config.history_dir).entries("bronze")
    assert [entry.run_id for entry in entries] == ["manual__2026-08-05T06:00:00+00:00"]


def test_a_history_limit_below_one_is_refused(tmp_path):
    with pytest.raises(ValueError, match="history_limit"):
        _config(tmp_path, history_limit=0)
