"""The gate: the point where a bad report stops the pipeline."""

from __future__ import annotations

import logging

from tripetl.quality.report import QualityReport

logger = logging.getLogger(__name__)


class QualityGateFailed(RuntimeError):
    """Raised when a blocking rule falls below its threshold.

    Carries the whole report, not just a message, so a caller -- an Airflow
    task, a test, the CLI -- can render the detail rather than re-running the
    checks to find out what went wrong.
    """

    def __init__(self, report: QualityReport) -> None:
        self.report = report
        failed = ", ".join(r.name for r in report.errors)
        super().__init__(
            f"quality gate failed at stage {report.stage!r}: {failed} "
            f"({len(report.errors)} blocking, {len(report.warnings)} warning)"
        )


def enforce(report: QualityReport, *, strict: bool = True) -> QualityReport:
    """Log the report and, when ``strict``, raise on a blocking failure.

    Warnings are never fatal. The distinction matters in practice: a rule you
    are still calibrating should not be able to page someone at 3am, but you
    also want to see it in the report every run until you have decided whether
    it earns ERROR severity.

    Passing ``strict=False`` surveys an unfamiliar extract end to end and
    collects every problem, instead of stopping at the first one.
    """
    for warning in report.warnings:
        logger.warning("quality warning at %s -- %s", report.stage, warning.summary())

    if report.passed:
        logger.info(
            "quality gate passed at %s (%d rules, %d rows)",
            report.stage,
            len(report.results),
            report.total_rows,
        )
        return report

    for error in report.errors:
        logger.error("quality failure at %s -- %s", report.stage, error.summary())

    if strict:
        raise QualityGateFailed(report)

    logger.warning(
        "gate at %s not enforced (strict=False); continuing with %d blocking failures",
        report.stage,
        len(report.errors),
    )
    return report
