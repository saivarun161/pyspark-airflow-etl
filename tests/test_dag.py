"""DAG integrity.

A broken DAG file does not fail loudly -- Airflow keeps serving the last
version it parsed successfully and reports an import error in a corner of the
UI. These tests are the thing that turns that into a red build. They are
skipped when Airflow is not installed, since the core package does not need it.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Must precede the Airflow import: on first import Airflow materialises its home
# directory, and without this it would litter the developer's home directory.
os.environ.setdefault("AIRFLOW_HOME", tempfile.mkdtemp(prefix="tripetl-airflow-"))

pytest.importorskip("airflow", reason="install the 'airflow' extra to run DAG tests")

DAG_PATH = Path(__file__).resolve().parents[1] / "dags" / "trip_etl_dag.py"


@pytest.fixture(scope="module")
def dag_module():
    spec = importlib.util.spec_from_file_location("trip_etl_dag", DAG_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["trip_etl_dag"] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop("trip_etl_dag", None)


@pytest.fixture
def dag(dag_module):
    return dag_module.dag_object


def test_the_dag_file_imports_cleanly(dag):
    """The check that catches a typo before the scheduler silently serves stale."""
    assert dag is not None
    assert dag.dag_id == "trip_etl"


def test_it_has_exactly_the_three_medallion_tasks(dag):
    assert set(dag.task_dict) == {"bronze", "silver", "gold"}


def test_the_stages_run_in_order(dag):
    assert dag.task_dict["bronze"].downstream_task_ids == {"silver"}
    assert dag.task_dict["silver"].downstream_task_ids == {"gold"}
    assert dag.task_dict["gold"].downstream_task_ids == set()


def test_it_does_not_backfill_or_overlap_itself(dag):
    """Two concurrent runs would write the same warehouse paths."""
    assert dag.catchup is False
    assert dag.max_active_runs == 1


def test_every_runtime_knob_is_exposed_as_a_param(dag):
    assert set(dag.params) == {
        "warehouse",
        "input_path",
        "rows",
        "seed",
        "dirty_rate",
        "enforce_gates",
    }


def test_defaults_run_offline(dag):
    """A fresh Airflow install can run this DAG with nothing else configured."""
    assert dag.params["input_path"] is None
    assert dag.params["rows"] > 0


def test_tasks_retry_transient_failures(dag):
    for task in dag.tasks:
        assert task.retries == 1


def test_the_summary_payload_stays_small(dag_module):
    """XCom carries metadata about the run, never the data itself."""
    from tripetl.pipeline import StageResult
    from tripetl.quality.report import QualityReport

    result = StageResult(
        stage="bronze",
        rows_in=100,
        rows_out=95,
        rows_quarantined=5,
        output_path="warehouse/bronze/trips",
        report=QualityReport(stage="bronze", ruleset="bronze", total_rows=100, results=()),
    )
    payload = dag_module._summary(result)

    assert payload["rows_out"] == 95
    assert payload["gate_passed"] is True
    assert payload["warnings"] == []
    assert all(isinstance(value, (str, int, bool, list)) for value in payload.values())
