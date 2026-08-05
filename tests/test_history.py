"""Report history and run-over-run trends.

Pure Python over JSON files and dataclasses -- no SparkSession, no warehouse.
The history is a directory of reports and the trend is a comparison of two of
them, and keeping the tests at that level is what makes it cheap to cover every
grade and every edge of the ordering.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone

import pytest

from tripetl.quality.history import (
    DEFAULT_TOLERANCE,
    DROPPED,
    IMPROVED,
    NEW,
    REGRESSED,
    STEADY,
    HistoryEntry,
    ReportHistory,
    RuleTrend,
    TrendReport,
    compare_reports,
    compare_runs,
    new_run_id,
)
from tripetl.quality.report import QualityReport, RuleResult

EPOCH = datetime(2026, 8, 5, 6, 0, 0, tzinfo=UTC)


def _result(name: str, pass_rate: float, *, threshold: float = 0.98, severity="error"):
    return RuleResult(
        name=name,
        description=name,
        severity=severity,
        kind="row",
        passed=pass_rate >= threshold,
        columns=("a_column",),
        threshold=threshold,
        pass_rate=pass_rate,
        failing_rows=round((1 - pass_rate) * 1000),
        total_rows=1000,
    )


def _dataset_result(name: str, *, passed: bool):
    return RuleResult(
        name=name,
        description=name,
        severity="error",
        kind="dataset",
        passed=passed,
        metric=1.0 if passed else 0.0,
    )


def _report(*results: RuleResult, stage: str = "bronze", rows: int = 1000) -> QualityReport:
    return QualityReport(stage=stage, ruleset=stage, total_rows=rows, results=results)


# -- round-tripping a report ------------------------------------------------


def test_a_report_survives_a_round_trip_through_json():
    original = _report(_result("not_null[a]", 0.999), _dataset_result("unique[id]", passed=True))
    restored = QualityReport.from_dict(json.loads(original.to_json()))
    assert restored == original


def test_a_restored_result_keeps_its_columns_as_a_tuple():
    """JSON has no tuples; a list would compare unequal to the original."""
    original = _result("r", 0.99)
    payload = json.loads(_report(original).to_json())["results"][0]

    assert payload["columns"] == ["a_column"]  # what JSON gave back
    assert RuleResult.from_dict(payload) == original


def test_a_result_missing_an_optional_field_still_restores():
    """An artifact written by an older version has fewer keys, not invalid ones."""
    payload = json.loads(_report(_result("r", 0.99)).to_json())["results"][0]
    del payload["metric"]

    assert RuleResult.from_dict(payload).metric is None


def test_a_report_verdict_is_recomputed_not_trusted():
    """A payload claiming it passed, whose results say otherwise, is not believed."""
    payload = json.loads(_report(_result("r", 0.5)).to_json())
    payload["passed"] = True
    payload["error_count"] = 0
    assert QualityReport.from_dict(payload).passed is False


# -- the store --------------------------------------------------------------


def test_recording_writes_one_file_per_run(tmp_path):
    history = ReportHistory(tmp_path)
    entry = history.record(_report(_result("r", 0.99)), run_id="run-1", now=EPOCH)

    assert history.path_for("bronze", "run-1").is_file()
    assert entry.run_id == "run-1"
    assert entry.stage == "bronze"
    assert entry.recorded_at == "2026-08-05T06:00:00Z"


def test_a_recorded_report_reads_back_identically(tmp_path):
    history = ReportHistory(tmp_path)
    report = _report(_result("r", 0.99), _dataset_result("unique[id]", passed=True))
    history.record(report, run_id="run-1", now=EPOCH)

    (entry,) = history.entries("bronze")
    assert entry.report == report


def test_entries_come_back_oldest_first(tmp_path):
    history = ReportHistory(tmp_path)
    for index in (2, 0, 1):
        history.record(
            _report(_result("r", 0.99)),
            run_id=f"run-{index}",
            now=EPOCH + timedelta(days=index),
        )

    assert [entry.run_id for entry in history.entries("bronze")] == ["run-0", "run-1", "run-2"]


def test_ordering_follows_the_recording_time_not_the_run_id(tmp_path):
    """A backfill runs old logical dates today; they are still the newest runs."""
    history = ReportHistory(tmp_path)
    history.record(_report(_result("r", 0.99)), run_id="2026-08-01", now=EPOCH)
    history.record(_report(_result("r", 0.99)), run_id="2026-07-01", now=EPOCH + timedelta(days=1))

    assert [entry.run_id for entry in history.entries("bronze")] == ["2026-08-01", "2026-07-01"]


def test_re_recording_a_run_id_replaces_it(tmp_path):
    """A cleared Airflow task reruns under the same id; the retry is the truth."""
    history = ReportHistory(tmp_path)
    history.record(_report(_result("r", 0.50)), run_id="run-1", now=EPOCH)
    history.record(_report(_result("r", 0.99)), run_id="run-1", now=EPOCH)

    (entry,) = history.entries("bronze")
    assert entry.report.results[0].pass_rate == 0.99


def test_stages_are_kept_apart(tmp_path):
    history = ReportHistory(tmp_path)
    history.record(_report(_result("r", 0.99), stage="bronze"), run_id="run-1", now=EPOCH)
    history.record(_report(_result("r", 0.99), stage="silver"), run_id="run-1", now=EPOCH)

    assert history.stages() == ("bronze", "silver")
    assert len(history.entries("bronze")) == 1
    assert len(history.entries("silver")) == 1


def test_an_unknown_stage_has_no_entries(tmp_path):
    assert ReportHistory(tmp_path).entries("gold") == ()
    assert ReportHistory(tmp_path).latest("gold") is None
    assert ReportHistory(tmp_path).stages() == ()


def test_a_corrupt_entry_is_skipped_not_fatal(tmp_path):
    history = ReportHistory(tmp_path)
    history.record(_report(_result("r", 0.99)), run_id="run-1", now=EPOCH)
    history.path_for("bronze", "run-2").write_text("{ truncated", encoding="utf-8")

    assert [entry.run_id for entry in history.entries("bronze")] == ["run-1"]


def test_latest_returns_the_newest_run(tmp_path):
    history = ReportHistory(tmp_path)
    history.record(_report(_result("r", 0.99)), run_id="run-1", now=EPOCH)
    history.record(_report(_result("r", 0.98)), run_id="run-2", now=EPOCH + timedelta(days=1))

    assert history.latest("bronze").run_id == "run-2"


def test_latest_before_a_known_run_skips_it(tmp_path):
    history = ReportHistory(tmp_path)
    history.record(_report(_result("r", 0.99)), run_id="run-1", now=EPOCH)
    history.record(_report(_result("r", 0.98)), run_id="run-2", now=EPOCH + timedelta(days=1))

    assert history.latest("bronze", before="run-2").run_id == "run-1"
    assert history.latest("bronze", before="run-1") is None


def test_latest_before_an_unrecorded_run_is_simply_the_newest(tmp_path):
    """Asking what preceded the run you are in the middle of."""
    history = ReportHistory(tmp_path)
    history.record(_report(_result("r", 0.99)), run_id="run-1", now=EPOCH)

    assert history.latest("bronze", before="run-in-flight").run_id == "run-1"


# -- pruning ----------------------------------------------------------------


def test_pruning_keeps_the_newest_runs(tmp_path):
    history = ReportHistory(tmp_path)
    for index in range(5):
        history.record(
            _report(_result("r", 0.99)), run_id=f"run-{index}", now=EPOCH + timedelta(days=index)
        )

    removed = history.prune("bronze", keep=2)

    assert removed == ("run-0", "run-1", "run-2")
    assert [entry.run_id for entry in history.entries("bronze")] == ["run-3", "run-4"]


def test_pruning_under_the_limit_removes_nothing(tmp_path):
    history = ReportHistory(tmp_path)
    history.record(_report(_result("r", 0.99)), run_id="run-1", now=EPOCH)

    assert history.prune("bronze", keep=30) == ()
    assert len(history.entries("bronze")) == 1


def test_pruning_rejects_a_negative_limit(tmp_path):
    with pytest.raises(ValueError, match="non-negative"):
        ReportHistory(tmp_path).prune("bronze", keep=-1)


# -- run ids ----------------------------------------------------------------


def test_a_run_id_sorts_chronologically_and_is_filename_safe():
    early = new_run_id(now=EPOCH)
    later = new_run_id(now=EPOCH + timedelta(seconds=1))

    assert early == "20260805T060000Z"
    assert early < later
    assert ":" not in early and "/" not in early


def test_a_run_id_is_utc_whatever_zone_the_clock_is_in():
    """Otherwise a laptop in Tokyo files its runs a day ahead of the cluster."""
    tokyo = datetime(2026, 8, 5, 15, 0, 0, tzinfo=timezone(timedelta(hours=9)))
    assert new_run_id(now=tokyo) == "20260805T060000Z"


# -- grading a trend --------------------------------------------------------


def test_a_falling_pass_rate_beyond_the_tolerance_regresses():
    trend = compare_reports(_report(_result("r", 0.999)), _report(_result("r", 0.985)))
    (rule,) = trend.rules

    assert rule.kind == REGRESSED
    assert rule.delta == pytest.approx(-0.014)
    assert not trend.stable


def test_a_rising_pass_rate_beyond_the_tolerance_improves():
    trend = compare_reports(_report(_result("r", 0.981)), _report(_result("r", 0.999)))
    assert trend.rules[0].kind == IMPROVED
    assert trend.stable  # an improvement is not instability


def test_movement_inside_the_tolerance_is_steady():
    trend = compare_reports(_report(_result("r", 0.9990)), _report(_result("r", 0.9960)))
    assert trend.rules[0].kind == STEADY
    assert trend.stable


@pytest.mark.parametrize(
    ("move", "expected"),
    [
        (-DEFAULT_TOLERANCE * 2, REGRESSED),
        (-DEFAULT_TOLERANCE / 2, STEADY),
        (0.0, STEADY),
        (DEFAULT_TOLERANCE / 2, STEADY),
        (DEFAULT_TOLERANCE * 2, IMPROVED),
    ],
)
def test_the_tolerance_brackets_what_counts_as_movement(move, expected):
    """Graded on the size of the move, with a threshold too low to be crossed.

    The boundary itself is deliberately not asserted: a pass rate is a ratio of
    two counts, so landing exactly on the tolerance is a float coincidence
    rather than a case anyone hits, and pinning it would be a test of IEEE 754.
    """
    before, after = 0.99, 0.99 + move
    trend = compare_reports(
        _report(_result("r", before, threshold=0.5)),
        _report(_result("r", after, threshold=0.5)),
    )
    assert trend.rules[0].kind == expected


def test_a_tighter_tolerance_catches_a_smaller_slip():
    reports = (_report(_result("r", 0.9999)), _report(_result("r", 0.9980)))
    assert compare_reports(*reports).rules[0].kind == STEADY
    assert compare_reports(*reports, tolerance=0.001).rules[0].kind == REGRESSED


def test_crossing_the_threshold_regresses_however_small_the_move():
    """The gate now blocks; a trend calling that 'steady' would contradict it."""
    trend = compare_reports(_report(_result("r", 0.9801)), _report(_result("r", 0.9799)))
    (rule,) = trend.rules

    assert rule.kind == REGRESSED
    assert rule.broke
    assert trend.broken == (rule,)


def test_a_rule_that_started_passing_improves_however_small_the_move():
    trend = compare_reports(_report(_result("r", 0.9799)), _report(_result("r", 0.9801)))
    assert trend.rules[0].kind == IMPROVED
    assert trend.rules[0].broke is False


def test_a_dataset_rule_trends_on_its_verdict_alone():
    trend = compare_reports(
        _report(_dataset_result("unique[id]", passed=True)),
        _report(_dataset_result("unique[id]", passed=False)),
    )
    (rule,) = trend.rules

    assert rule.delta is None
    assert rule.kind == REGRESSED
    assert rule.broke


def test_an_unchanged_dataset_rule_is_steady():
    trend = compare_reports(
        _report(_dataset_result("unique[id]", passed=True)),
        _report(_dataset_result("unique[id]", passed=True)),
    )
    assert trend.rules[0].kind == STEADY


# -- rule sets that changed shape -------------------------------------------


def test_a_rule_added_since_last_run_is_new():
    trend = compare_reports(
        _report(_result("a", 0.99)),
        _report(_result("a", 0.99), _result("b", 0.90)),
    )
    kinds = {rule.name: rule.kind for rule in trend.rules}

    assert kinds == {"a": STEADY, "b": NEW}
    assert trend.stable  # a new rule failing is not a regression, it is news


def test_a_rule_removed_since_last_run_is_dropped():
    trend = compare_reports(
        _report(_result("a", 0.99), _result("b", 0.99)),
        _report(_result("a", 0.99)),
    )
    kinds = {rule.name: rule.kind for rule in trend.rules}

    assert kinds == {"a": STEADY, "b": DROPPED}
    assert next(r for r in trend.rules if r.name == "b").current_pass_rate is None


# -- volume -----------------------------------------------------------------


def test_row_movement_is_reported_alongside_the_rules():
    """Every rule can pass at 100% on a file that arrived a third short."""
    trend = compare_reports(
        _report(_result("r", 1.0), rows=1000),
        _report(_result("r", 1.0), rows=600),
    )

    assert trend.stable
    assert trend.row_delta == -400
    assert trend.row_change == pytest.approx(-0.4)


def test_row_change_is_undefined_against_an_empty_previous_run():
    trend = compare_reports(_report(_result("r", 1.0), rows=0), _report(_result("r", 1.0), rows=5))
    assert trend.row_change is None
    assert trend.row_delta == 5


# -- guards -----------------------------------------------------------------


def test_comparing_two_different_stages_is_refused():
    with pytest.raises(ValueError, match="one stage over time"):
        compare_reports(_report(stage="bronze"), _report(stage="silver"))


def test_a_negative_tolerance_is_refused():
    with pytest.raises(ValueError, match="non-negative"):
        compare_reports(_report(), _report(), tolerance=-0.1)


# -- rendering --------------------------------------------------------------


def test_text_names_both_runs_and_the_verdict():
    trend = compare_reports(
        _report(_result("r", 0.999)),
        _report(_result("r", 0.985)),
        previous_run_id="run-1",
        current_run_id="run-2",
    )
    text = trend.to_text()

    assert "REGRESSED" in text
    assert "run-1 -> run-2" in text
    assert "DOWN" in text


def test_text_reports_a_stable_run_plainly():
    text = compare_reports(_report(_result("r", 0.99)), _report(_result("r", 0.99))).to_text()
    assert "STABLE" in text


def test_text_handles_two_runs_with_no_rules():
    assert "no rules in common" in compare_reports(_report(), _report()).to_text()


def test_markdown_starts_with_a_header_row():
    trend = compare_reports(_report(_result("r", 0.99)), _report(_result("r", 0.99)))
    assert trend.to_markdown().startswith("| rule |")


def test_json_carries_the_counts_and_the_movement():
    trend = compare_reports(
        _report(_result("r", 0.999), rows=1000),
        _report(_result("r", 0.985), rows=900),
    )
    payload = json.loads(trend.to_json())

    assert payload["stable"] is False
    assert payload["regression_count"] == 1
    assert payload["row_delta"] == -100
    assert payload["rules"][0]["kind"] == REGRESSED


def test_each_grade_summarises_readably():
    assert "not checked last run" in RuleTrend("r", NEW, current_pass_rate=0.9).summary()
    assert "no longer checked" in RuleTrend("r", DROPPED, previous_pass_rate=0.9).summary()
    assert (
        "pass -> fail"
        in RuleTrend("r", REGRESSED, previous_passed=True, current_passed=False).summary()
    )


def test_an_empty_trend_report_is_stable():
    trend = TrendReport(
        stage="bronze",
        previous_run_id="a",
        current_run_id="b",
        previous_rows=0,
        current_rows=0,
    )
    assert trend.stable
    assert trend.row_change is None


# -- comparing straight from the store --------------------------------------


def test_compare_runs_diffs_the_two_most_recent_entries(tmp_path):
    history = ReportHistory(tmp_path)
    history.record(_report(_result("r", 0.30)), run_id="run-0", now=EPOCH)
    history.record(_report(_result("r", 0.999)), run_id="run-1", now=EPOCH + timedelta(days=1))
    history.record(_report(_result("r", 0.985)), run_id="run-2", now=EPOCH + timedelta(days=2))

    trend = compare_runs(history, "bronze")

    assert (trend.previous_run_id, trend.current_run_id) == ("run-1", "run-2")
    assert trend.rules[0].kind == REGRESSED


def test_compare_runs_needs_two_runs(tmp_path):
    history = ReportHistory(tmp_path)
    assert compare_runs(history, "bronze") is None

    history.record(_report(_result("r", 0.99)), run_id="run-1", now=EPOCH)
    assert compare_runs(history, "bronze") is None


def test_a_history_entry_round_trips_through_json(tmp_path):
    entry = HistoryEntry(
        run_id="run-1",
        stage="bronze",
        recorded_at="2026-08-05T06:00:00Z",
        report=_report(_result("r", 0.99)),
    )
    assert HistoryEntry.from_dict(json.loads(entry.to_json())) == entry
