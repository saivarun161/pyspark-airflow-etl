"""Report classification and rendering."""

from __future__ import annotations

import json

from tripetl.quality.report import QualityReport, RuleResult
from tripetl.quality.rules import Severity


def _result(name: str, *, passed: bool, severity: Severity = Severity.ERROR) -> RuleResult:
    return RuleResult(
        name=name,
        description=f"{name} description",
        severity=severity.value,
        kind="row",
        passed=passed,
        columns=("value",),
        threshold=0.99,
        pass_rate=1.0 if passed else 0.5,
        failing_rows=0 if passed else 50,
        total_rows=100,
    )


def _report(*results: RuleResult) -> QualityReport:
    return QualityReport(stage="bronze", ruleset="bronze", total_rows=100, results=results)


def test_only_failing_error_rules_block():
    report = _report(
        _result("ok", passed=True),
        _result("warned", passed=False, severity=Severity.WARN),
        _result("broken", passed=False),
    )

    assert [r.name for r in report.errors] == ["broken"]
    assert [r.name for r in report.warnings] == ["warned"]
    assert [r.name for r in report.failures] == ["warned", "broken"]
    assert not report.passed


def test_a_report_with_only_warnings_passes():
    report = _report(_result("warned", passed=False, severity=Severity.WARN))
    assert report.passed
    assert report.warnings


def test_a_passing_warn_rule_is_neither_error_nor_warning():
    report = _report(_result("fine", passed=True, severity=Severity.WARN))
    assert report.passed
    assert report.warnings == ()


def test_json_round_trips_to_primitives():
    """Reports are written to disk and read back by other tools."""
    report = _report(_result("ok", passed=True), _result("broken", passed=False))
    payload = json.loads(report.to_json())

    assert payload["stage"] == "bronze"
    assert payload["passed"] is False
    assert payload["error_count"] == 1
    assert payload["warning_count"] == 0
    assert len(payload["results"]) == 2
    assert payload["results"][0]["name"] == "ok"
    assert payload["results"][0]["columns"] == ["value"]


def test_text_rendering_names_the_verdict_and_the_failures():
    report = _report(_result("broken", passed=False))
    text = report.to_text()

    assert "FAILED" in text
    assert "[FAIL] broken" in text
    assert "50 bad rows" in text


def test_markdown_renders_one_row_per_rule():
    report = _report(_result("ok", passed=True), _result("broken", passed=False))
    lines = report.to_markdown().splitlines()

    assert lines[0].startswith("| rule |")
    assert len(lines) == 4  # header, separator, two rules
    assert "**fail**" in lines[3]


def test_dataset_results_render_their_metric():
    result = RuleResult(
        name="unique[id]",
        description="id must be unique",
        severity=Severity.ERROR.value,
        kind="dataset",
        passed=False,
        metric=3.0,
    )
    assert "metric=3" in result.summary()
    assert result.is_blocking
