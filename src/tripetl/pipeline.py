"""Stage functions and the end-to-end run.

Each stage reads from the warehouse and writes back to it, so a stage is
runnable on its own: the CLI chains all three inside one SparkSession, while
Airflow calls the same functions from three separate task processes, each with
its own session. Neither caller is privileged, and there is exactly one
implementation of the logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from tripetl.config import PipelineConfig
from tripetl.quality import engine, rulesets
from tripetl.quality.drift import enforce_schema
from tripetl.quality.gate import enforce
from tripetl.quality.history import ReportHistory, new_run_id
from tripetl.quality.report import QualityReport
from tripetl.quality.rules import RuleSet
from tripetl.schema import PARTITION_COLUMN
from tripetl.session import session_scope
from tripetl.sources import generate_sample, raw_schema_diff, read_raw
from tripetl.transforms.aggregate import daily_zone_metrics
from tripetl.transforms.clean import clean, deduplicate
from tripetl.transforms.enrich import enrich

logger = logging.getLogger(__name__)


class NothingToPublish(RuntimeError):
    """A stage finished with no rows to write.

    Not a gate failure -- the rules may all have passed -- but not something to
    carry on from either. A partitioned write of an empty frame produces no
    partitions at all, so the next stage reads a path that was never created
    and fails on schema inference, three stages of log output away from the run
    that emptied it.

    Raised whatever ``enforce_gates`` says, because that knob governs how rule
    verdicts are treated, and this is not a verdict: there is no dataset to
    hand downstream regardless of anyone's opinion about the rules.
    """

    def __init__(self, stage: str, rows_in: int) -> None:
        self.stage = stage
        self.rows_in = rows_in
        super().__init__(
            f"stage {stage!r} has nothing to publish: 0 rows survived out of {rows_in:,} read"
        )


@dataclass(frozen=True)
class StageResult:
    """What one stage did, and what the gate made of it."""

    stage: str
    rows_in: int
    rows_out: int
    rows_quarantined: int
    output_path: str
    report: QualityReport

    def summary(self) -> str:
        verdict = "ok" if self.report.passed else "GATE FAILED"
        return (
            f"{self.stage:<7} {self.rows_in:>9,} in -> {self.rows_out:>9,} out"
            f"  quarantined={self.rows_quarantined:<7,} {verdict}"
        )


@dataclass(frozen=True)
class PipelineResult:
    """The whole run."""

    stages: tuple[StageResult, ...]

    @property
    def passed(self) -> bool:
        return all(stage.report.passed for stage in self.stages)

    @property
    def gold_rows(self) -> int:
        return self.stages[-1].rows_out if self.stages else 0

    def stage(self, name: str) -> StageResult:
        for result in self.stages:
            if result.stage == name:
                return result
        raise KeyError(f"no stage named {name!r} in this run")

    def to_text(self) -> str:
        lines = [stage.summary() for stage in self.stages]
        lines.append("-" * 68)
        lines.append(
            f"pipeline {'PASSED' if self.passed else 'FAILED'}; {self.gold_rows:,} rows published"
        )
        return "\n".join(lines)


def _write(config: PipelineConfig, df: DataFrame, path: str) -> None:
    """Write a layer, replacing only the partitions this run produced.

    Spark's default is a *static* overwrite, which deletes everything under the
    output path before writing -- the whole dataset when it is unpartitioned,
    every partition when it is not. Either way, reprocessing one day of January
    takes February with it, which is why so many pipelines reach for ``append``
    and get duplicated days the first time a task is retried instead.

    ``partitionOverwriteMode=dynamic`` narrows the delete to the partitions
    actually present in this DataFrame. A run scoped to one day replaces that
    day and touches nothing else, so a backfill is a backfill and a retry is a
    no-op rather than a second copy.

    Set as a writer option rather than on the session, because the session is
    frequently not this code's to configure: the CLI builds one, Airflow builds
    one per task, and a test hands one in. An option travels with the write.
    """
    writer = df.write.mode(config.write_mode)
    if config.partition_by:
        writer = writer.partitionBy(*config.partition_by).option(
            "partitionOverwriteMode", "dynamic"
        )
    writer.parquet(path)


def _publish(config: PipelineConfig, df: DataFrame, path: str, *, stage: str, rows_in: int) -> int:
    """Count a stage's output, insist there is some, and write it."""
    rows_out = df.count()
    if rows_out == 0:
        raise NothingToPublish(stage, rows_in)
    _write(config, df, path)
    return rows_out


def _restrict(config: PipelineConfig, df: DataFrame, what: str) -> DataFrame:
    """Narrow a frame to the run's date window.

    Applied at bronze *before* the gate, so the report describes the rows the
    run actually processed rather than a file it only partly read -- a report
    covering January while the run published one day of it would misattribute
    every rate in it.

    Applied again to the silver and gold reads, where it is not a filter so
    much as a directory listing: the window names partitions, so Spark prunes
    the rest instead of scanning them. A one-day backfill should cost one day
    of I/O, not a full pass over the warehouse to find the day.
    """
    if not config.has_window:
        return df

    logger.info("%s restricted to pickup dates %s", what, config.window_label)
    partition = F.col(PARTITION_COLUMN)
    if config.since is not None:
        df = df.filter(partition >= F.lit(config.since))
    if config.until is not None:
        df = df.filter(partition <= F.lit(config.until))
    return df


def _write_report(config: PipelineConfig, report: QualityReport) -> str:
    """Persist a report next to the data it describes.

    Written *before* the gate is enforced. A run that fails must still leave
    behind the evidence of why, otherwise the first thing anyone does after an
    incident is re-run the job to find out what happened -- against data that
    may have changed in the meantime.
    """
    path = Path(config.report_path(report.stage))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(), encoding="utf-8")
    _record_history(config, report)
    return str(path)


def _record_history(config: PipelineConfig, report: QualityReport) -> None:
    """Copy the report into the run history, then trim the oldest entries.

    Also before the gate, and for a stronger reason than the report itself:
    the run that blocks is the one whose predecessors you want to look at, and
    a history that skips failed runs is missing exactly the entries that
    explain how the failure arrived.

    Never allowed to break a run. A warehouse whose history directory is
    read-only is a warehouse that should still publish its mart; losing a
    trend is a smaller failure than losing the pipeline.
    """
    if not config.keep_history:
        return
    try:
        history = ReportHistory(config.history_dir)
        history.record(report, run_id=config.run_id)
        history.prune(report.stage, keep=config.history_limit)
    except OSError:
        logger.warning("could not record %s history under %s", report.stage, config.history_dir)


def _check_input_schema(config: PipelineConfig, session: SparkSession) -> None:
    """Compare the input's layout to the declared one before reading a row.

    Recorded and enforced ahead of the read for the same reason the quality
    reports are written before their gate: the run that stops is the one whose
    evidence you need. A column that disappeared upstream is one line here,
    against a stage report full of rules failing at 0% on a column that reads
    as nulls for reasons the report cannot see.
    """
    if not config.input_path or not config.check_input_schema:
        return

    diff = raw_schema_diff(session, config.input_path, fmt=config.input_format)
    path = Path(config.schema_diff_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(diff.to_json(), encoding="utf-8")
    enforce_schema(diff, strict=config.enforce_gates)


def _guard(
    config: PipelineConfig,
    df: DataFrame,
    ruleset: RuleSet,
    stage: str,
    *,
    quarantine: bool,
) -> tuple[DataFrame, QualityReport, int]:
    """Evaluate, persist the report, enforce the gate, then split off bad rows.

    Returns the rows that should flow onward, the report, and how many rows
    were quarantined.
    """
    report = engine.evaluate(df, ruleset, stage=stage)
    _write_report(config, report)
    enforce(report, strict=config.enforce_gates)

    if not (quarantine and config.quarantine_enabled):
        return df, report, 0

    passing, failing = engine.split(df, ruleset.quarantine_rules())
    quarantined_rows = failing.count()
    if quarantined_rows:
        logger.warning(
            "quarantining %d rows at stage %s -> %s",
            quarantined_rows,
            stage,
            config.quarantine_path(stage),
        )
    _write(config, failing, config.quarantine_path(stage))
    return passing, report, quarantined_rows


def run_bronze(config: PipelineConfig, *, spark: SparkSession | None = None) -> StageResult:
    """Ingest raw trips, key them, and gate on structural validity."""
    with session_scope(spark, **config.session_kwargs()) as session:
        if config.input_path:
            _check_input_schema(config, session)
            raw = read_raw(session, config.input_path, fmt=config.input_format)
        else:
            logger.info(
                "no input_path configured; generating %d sample rows (seed=%d)",
                config.sample_rows,
                config.sample_seed,
            )
            raw = generate_sample(
                session,
                rows=config.sample_rows,
                seed=config.sample_seed,
                dirty_rate=config.sample_dirty_rate,
            )

        # Cached because the gate reads it twice: once to measure, once to
        # split. Without this the sample generator runs again between them and
        # the numbers in the report describe different rows than the ones
        # written out.
        shaped = _restrict(config, clean(raw), "bronze input").cache()
        try:
            passing, report, quarantined = _guard(
                config, shaped, rulesets.bronze_ruleset(), "bronze", quarantine=True
            )
            rows_out = _publish(
                config,
                passing,
                config.bronze_path,
                stage="bronze",
                rows_in=report.total_rows,
            )
        finally:
            shaped.unpersist()

        return StageResult(
            stage="bronze",
            rows_in=report.total_rows,
            rows_out=rows_out,
            rows_quarantined=quarantined,
            output_path=config.bronze_path,
            report=report,
        )


def run_silver(config: PipelineConfig, *, spark: SparkSession | None = None) -> StageResult:
    """Deduplicate and enrich bronze, then gate on derived-value plausibility."""
    with session_scope(spark, **config.session_kwargs()) as session:
        bronze = _restrict(config, session.read.parquet(config.bronze_path), "bronze")
        rows_in = bronze.count()

        enriched = enrich(deduplicate(bronze)).cache()
        try:
            passing, report, quarantined = _guard(
                config, enriched, rulesets.silver_ruleset(), "silver", quarantine=True
            )
            rows_out = _publish(
                config, passing, config.silver_path, stage="silver", rows_in=rows_in
            )
        finally:
            enriched.unpersist()

        return StageResult(
            stage="silver",
            rows_in=rows_in,
            rows_out=rows_out,
            rows_quarantined=quarantined,
            output_path=config.silver_path,
            report=report,
        )


def run_gold(config: PipelineConfig, *, spark: SparkSession | None = None) -> StageResult:
    """Aggregate silver into the published mart and gate on the result.

    Nothing is quarantined here. A gold row that fails a check means the
    aggregation is wrong, not that one input was bad -- dropping it would
    publish a mart that is quietly missing a zone rather than one that visibly
    failed to build.
    """
    with session_scope(spark, **config.session_kwargs()) as session:
        silver = _restrict(config, session.read.parquet(config.silver_path), "silver")
        rows_in = silver.count()

        gold = daily_zone_metrics(silver).cache()
        try:
            published, report, _ = _guard(
                config, gold, rulesets.gold_ruleset(), "gold", quarantine=False
            )
            rows_out = _publish(config, published, config.gold_path, stage="gold", rows_in=rows_in)
        finally:
            gold.unpersist()

        return StageResult(
            stage="gold",
            rows_in=rows_in,
            rows_out=rows_out,
            rows_quarantined=0,
            output_path=config.gold_path,
            report=report,
        )


def run_pipeline(config: PipelineConfig, *, spark: SparkSession | None = None) -> PipelineResult:
    """Run bronze, silver and gold in one session.

    Raises :class:`~tripetl.quality.gate.QualityGateFailed` at the first gate
    that fails, or :class:`~tripetl.quality.drift.SchemaDriftError` before
    bronze reads an input whose layout moved. Neither is raised when
    ``config.enforce_gates`` is false.

    A run id is settled here rather than per stage, so all three reports land
    in the history under one name. Leaving each stage to generate its own from
    the clock would key them seconds apart, and "what did bronze say on the run
    where gold went wrong" would stop being answerable.
    """
    if config.keep_history and config.run_id is None:
        config = replace(config, run_id=new_run_id())

    with session_scope(spark, **config.session_kwargs()) as session:
        stages = (
            run_bronze(config, spark=session),
            run_silver(config, spark=session),
            run_gold(config, spark=session),
        )
    return PipelineResult(stages=stages)
