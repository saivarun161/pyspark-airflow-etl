"""Daily trip ETL: bronze, silver, gold, with a quality gate between each.

Each task is its own Spark application. That mirrors how this runs for real --
three ``spark-submit`` calls against a cluster, not one long-lived driver
babysat by the scheduler -- and it means a failed stage can be cleared and
retried on its own without recomputing the stages before it.

Clearing a task is safe to do twice. Each layer is written partitioned by
pickup date with a dynamic overwrite, so a retry replaces the partitions it
produced rather than appending a second copy of them -- and a run given a
``since``/``until`` window rebuilds those days without disturbing any other.

The gates need no special Airflow machinery. A blocking rule raises
``QualityGateFailed``, the task fails, and downstream tasks do not run. The
mart is simply not republished, which is the correct outcome: yesterday's
numbers staying up beats today's bad numbers going out.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from airflow.sdk import Param, dag, get_current_context, task

from tripetl.config import DEFAULT_PARTITION_BY, PipelineConfig
from tripetl.pipeline import StageResult, run_bronze, run_gold, run_silver

DAG_ID = "trip_etl"

#: Runtime knobs, editable from the UI on a manual trigger. Defaults run the
#: pipeline against generated sample data, so the DAG is demonstrable on a
#: fresh Airflow instance with nothing else configured.
PARAMS = {
    "warehouse": Param("warehouse", type="string", title="Warehouse root"),
    "input_path": Param(
        None,
        type=["null", "string"],
        title="Raw input path (blank generates a sample)",
    ),
    "rows": Param(20_000, type="integer", minimum=0, title="Sample rows"),
    "seed": Param(20260803, type="integer", title="Sample seed"),
    "dirty_rate": Param(0.04, type="number", minimum=0, maximum=1, title="Injected defect rate"),
    "enforce_gates": Param(True, type="boolean", title="Fail the run on a bad gate"),
    "check_input_schema": Param(
        True,
        type="boolean",
        title="Check the input's layout against the declared schema",
    ),
    "keep_history": Param(True, type="boolean", title="Keep every run's quality report"),
    "history_limit": Param(30, type="integer", minimum=1, title="Runs of history to retain"),
    # A backfill is a normal run with a window. Left blank the run processes
    # everything the input holds, which is what the generated-sample demo
    # wants; a deployment reading monthly extracts sets these to the day it is
    # rebuilding and the partitioned writes confine the run to it.
    "since": Param(
        None,
        type=["null", "string"],
        title="Process trips picked up on or after (YYYY-MM-DD)",
    ),
    "until": Param(
        None,
        type=["null", "string"],
        title="Process trips picked up on or before (YYYY-MM-DD)",
    ),
}

DEFAULT_ARGS = {
    "owner": "data-platform",
    # A retry is here for transient cluster problems -- a lost executor, a
    # flaky object-store read. It will not rescue a failed gate: bad data is
    # deterministic and the second attempt fails identically, which is what
    # should happen.
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _config() -> PipelineConfig:
    """Build a pipeline config from the current run's params."""
    context = get_current_context()
    params = context["params"]
    return PipelineConfig(
        warehouse=params["warehouse"],
        input_path=params["input_path"] or None,
        enforce_gates=params["enforce_gates"],
        check_input_schema=params["check_input_schema"],
        sample_rows=params["rows"],
        sample_seed=params["seed"],
        sample_dirty_rate=params["dirty_rate"],
        keep_history=params["keep_history"],
        history_limit=params["history_limit"],
        since=_as_date(params["since"]),
        until=_as_date(params["until"]),
        partition_by=DEFAULT_PARTITION_BY,
        # Airflow's own run id, rather than one generated per task from the
        # clock. Three tasks in three processes would otherwise file their
        # reports under three different names seconds apart, and "what did
        # bronze say on the run where gold went wrong" would have no answer.
        # It also makes a cleared task idempotent: the retry overwrites its
        # entry instead of adding a second one for the same run.
        run_id=_run_id(context),
    )


def _as_date(value: object) -> date | None:
    """Read a window bound out of the params.

    Params carry JSON, so a date arrives as a string -- and as an empty string
    when someone clears the field in the UI rather than nulling it, which has
    to mean the same thing as leaving it blank.
    """
    if not value:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _run_id(context: dict[str, object]) -> str | None:
    """Name this run after the DAG run that triggered it.

    ``dag_run`` is absent when a task function is called outside a run -- in a
    unit test, or from the shell -- so this degrades to ``None`` and lets the
    pipeline generate one rather than failing on a missing key.
    """
    dag_run = context.get("dag_run")
    return getattr(dag_run, "run_id", None)


def _summary(result: StageResult) -> dict[str, object]:
    """Reduce a stage result to something XCom can carry.

    XCom is for small metadata, never for data. What lands here is the shape of
    the run -- counts and a verdict -- so the next task and anyone reading the
    UI can see what happened without opening the warehouse.
    """
    return {
        "stage": result.stage,
        "rows_in": result.rows_in,
        "rows_out": result.rows_out,
        "rows_quarantined": result.rows_quarantined,
        "output_path": result.output_path,
        "gate_passed": result.report.passed,
        "warnings": [warning.name for warning in result.report.warnings],
    }


@dag(
    dag_id=DAG_ID,
    description="NYC taxi trip ETL with data-quality gates between layers",
    schedule="0 6 * * *",
    start_date=datetime(2026, 6, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    params=PARAMS,
    tags=["etl", "pyspark", "data-quality"],
    doc_md=__doc__,
)
def trip_etl():
    @task(task_display_name="bronze: ingest and validate")
    def bronze() -> dict[str, object]:
        return _summary(run_bronze(_config()))

    @task(task_display_name="silver: dedupe, enrich and validate")
    def silver(upstream: dict[str, object]) -> dict[str, object]:
        return _summary(run_silver(_config()))

    @task(task_display_name="gold: publish daily zone metrics")
    def gold(upstream: dict[str, object]) -> dict[str, object]:
        return _summary(run_gold(_config()))

    # Passing each result into the next task is what declares the dependency;
    # the payload itself is only used for reporting.
    gold(silver(bronze()))


dag_object = trip_etl()
