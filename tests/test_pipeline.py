"""End-to-end stage behaviour against a temporary warehouse."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tripetl.config import PipelineConfig
from tripetl.pipeline import run_bronze, run_gold, run_pipeline, run_silver
from tripetl.quality.engine import FAILURE_COLUMN
from tripetl.quality.gate import QualityGateFailed

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


def test_unknown_stage_lookup_raises(completed):
    _, result = completed
    with pytest.raises(KeyError, match="platinum"):
        result.stage("platinum")


def test_pipeline_summary_mentions_the_verdict(completed):
    _, result = completed
    text = result.to_text()
    assert "PASSED" in text
    assert "bronze" in text and "gold" in text
