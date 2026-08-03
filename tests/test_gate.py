"""Gate behaviour: what stops a pipeline and what merely gets logged."""

from __future__ import annotations

import logging

import pytest

from tripetl.quality.gate import QualityGateFailed, enforce
from tripetl.quality.report import QualityReport, RuleResult
from tripetl.quality.rules import Severity


def _result(name: str, *, passed: bool, severity: Severity) -> RuleResult:
    return RuleResult(
        name=name,
        description=name,
        severity=severity.value,
        kind="row",
        passed=passed,
        threshold=0.99,
        pass_rate=1.0 if passed else 0.5,
        failing_rows=0 if passed else 5,
        total_rows=10,
    )


def _report(*results: RuleResult) -> QualityReport:
    return QualityReport(stage="bronze", ruleset="bronze", total_rows=10, results=results)


def test_a_clean_report_passes_through_unchanged():
    report = _report(_result("ok", passed=True, severity=Severity.ERROR))
    assert enforce(report) is report


def test_a_blocking_failure_raises():
    report = _report(_result("broken", passed=False, severity=Severity.ERROR))

    with pytest.raises(QualityGateFailed) as excinfo:
        enforce(report)

    assert "broken" in str(excinfo.value)
    assert "bronze" in str(excinfo.value)


def test_the_exception_carries_the_whole_report():
    """So a caller can render the detail instead of re-running the checks."""
    report = _report(_result("broken", passed=False, severity=Severity.ERROR))

    with pytest.raises(QualityGateFailed) as excinfo:
        enforce(report)

    assert excinfo.value.report is report
    assert excinfo.value.report.errors[0].name == "broken"


def test_warnings_never_stop_a_run():
    report = _report(_result("warned", passed=False, severity=Severity.WARN))
    assert enforce(report, strict=True) is report


def test_non_strict_mode_reports_without_raising():
    report = _report(_result("broken", passed=False, severity=Severity.ERROR))
    assert enforce(report, strict=False) is report


def test_failures_are_logged_before_they_are_raised(caplog):
    report = _report(
        _result("warned", passed=False, severity=Severity.WARN),
        _result("broken", passed=False, severity=Severity.ERROR),
    )

    with (
        caplog.at_level(logging.WARNING, logger="tripetl.quality.gate"),
        pytest.raises(QualityGateFailed),
    ):
        enforce(report)

    messages = [record.getMessage() for record in caplog.records]
    assert any("warned" in message for message in messages)
    assert any("broken" in message for message in messages)
