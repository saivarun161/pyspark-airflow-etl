"""Stage functions and the end-to-end run.

Each stage reads from the warehouse and writes back to it, so a stage is
runnable on its own: the CLI chains all three inside one SparkSession, while
Airflow calls the same functions from three separate task processes, each with
its own session. Neither caller is privileged, and there is exactly one
implementation of the logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from tripetl.config import PipelineConfig
from tripetl.quality import engine, rulesets
from tripetl.quality.drift import enforce_schema
from tripetl.quality.gate import enforce
from tripetl.quality.report import QualityReport
from tripetl.quality.rules import RuleSet
from tripetl.session import session_scope
from tripetl.sources import generate_sample, raw_schema_diff, read_raw
from tripetl.transforms.aggregate import daily_zone_metrics
from tripetl.transforms.clean import clean, deduplicate
from tripetl.transforms.enrich import enrich

logger = logging.getLogger(__name__)


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
    return str(path)


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
    failing.write.mode(config.write_mode).parquet(config.quarantine_path(stage))
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
        shaped = clean(raw).cache()
        try:
            passing, report, quarantined = _guard(
                config, shaped, rulesets.bronze_ruleset(), "bronze", quarantine=True
            )
            rows_out = passing.count()
            passing.write.mode(config.write_mode).parquet(config.bronze_path)
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
        bronze = session.read.parquet(config.bronze_path)
        rows_in = bronze.count()

        enriched = enrich(deduplicate(bronze)).cache()
        try:
            passing, report, quarantined = _guard(
                config, enriched, rulesets.silver_ruleset(), "silver", quarantine=True
            )
            rows_out = passing.count()
            passing.write.mode(config.write_mode).parquet(config.silver_path)
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
        silver = session.read.parquet(config.silver_path)
        rows_in = silver.count()

        gold = daily_zone_metrics(silver).cache()
        try:
            published, report, _ = _guard(
                config, gold, rulesets.gold_ruleset(), "gold", quarantine=False
            )
            rows_out = published.count()
            published.write.mode(config.write_mode).parquet(config.gold_path)
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
    """
    with session_scope(spark, **config.session_kwargs()) as session:
        stages = (
            run_bronze(config, spark=session),
            run_silver(config, spark=session),
            run_gold(config, spark=session),
        )
    return PipelineResult(stages=stages)
