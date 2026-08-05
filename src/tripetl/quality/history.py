"""Report history: what the last run said, so this run can be compared to it.

A gate answers one question about one batch -- is this good enough to publish?
-- and answers it without reference to anything that came before. That is the
right shape for a gate. It is also why a gate cannot see the failure mode that
matters most in practice: a rule that degrades slowly.

``not_null[passenger_count]`` at 99.98%, then 99.4%, then 98.6%, against a 98%
threshold, passes three times. The gate is not wrong on any of those days --
the batch really was publishable. But something upstream started breaking a
fortnight ago, and the run that finally blocks is the one with no history to
explain it. Each run overwriting ``_quality/bronze.json`` is what throws that
away: the evidence exists, once, and then the next run deletes it.

So every report is kept under ``_quality/history/<stage>/<run_id>.json``, and
a run can be compared against the one before it. Movement is graded the way
:mod:`tripetl.quality.drift` grades a column: on whether the change is one
someone should act on, not merely on whether a number differs.

Trends do not block by default. A gate encodes a policy someone committed to;
a trend is a signal still being calibrated, and a check that pages someone
because a rate wobbled 0.2% overnight is a check that gets muted in a week.
:func:`compare_runs` reports, and the caller decides.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from tripetl.quality.report import QualityReport, RuleResult

logger = logging.getLogger(__name__)

#: The rule passes at a materially lower rate than it did last run.
REGRESSED = "regressed"

#: Materially higher. Worth surfacing rather than only reporting the bad news:
#: a jump the day after someone changed an upstream job is how you confirm the
#: fix landed, and a jump nobody can explain is its own kind of suspicious.
IMPROVED = "improved"

#: Moved by less than the tolerance, or not at all.
STEADY = "steady"

#: Present this run, absent last run -- a rule set gained a rule.
NEW = "new"

#: Present last run, absent this run. Worth naming, because a rule quietly
#: leaving a rule set looks identical in the report to a rule that never
#: existed, and "we stopped checking" is not the same as "it passes".
DROPPED = "dropped"

#: How far a pass rate may move before it is called a change. Daily feeds
#: wobble; grading every fraction of a percent as a regression produces a
#: trend report that is noise on a good day and ignored on a bad one. Half a
#: percentage point is deliberately coarse -- it is the floor at which a
#: single day's movement is worth a human look, not the sensitivity you would
#: want for a rule sitting at 99.99%. Pass a tighter ``tolerance`` for those.
DEFAULT_TOLERANCE = 0.005

#: How many runs to keep per stage before the oldest are dropped. A daily
#: pipeline fills a month at this size, which is long enough to see a slow
#: drift start and short enough that the directory stays readable.
DEFAULT_KEEP = 30

_SUFFIX = ".json"


def new_run_id(*, now: datetime | None = None) -> str:
    """A run identifier that sorts chronologically as a string.

    Compact UTC rather than full ISO-8601, because this becomes a filename and
    the colons in ``2026-08-05T17:04:11+00:00`` are legal on POSIX, awkward on
    Windows, and unpleasant in a shell everywhere.
    """
    moment = now or datetime.now(UTC)
    return moment.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _timestamp(*, now: datetime | None = None) -> str:
    moment = now or datetime.now(UTC)
    return moment.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class HistoryEntry:
    """One recorded run of one stage."""

    run_id: str
    stage: str
    recorded_at: str
    report: QualityReport

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "recorded_at": self.recorded_at,
            "report": self.report.to_dict(),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=list)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> HistoryEntry:
        report = QualityReport.from_dict(payload["report"])  # type: ignore[arg-type]
        return cls(
            run_id=str(payload["run_id"]),
            stage=str(payload["stage"]),
            recorded_at=str(payload["recorded_at"]),
            report=report,
        )

    @property
    def sort_key(self) -> tuple[str, str]:
        """Order by when it was recorded, then by id.

        Recording time rather than the id alone, because a run id is not always
        ours to choose: Airflow passes its logical date, which for a backfill
        runs backwards through dates that were recorded in order. Sorting on
        the id would then interleave a backfill with the live runs and compare
        each against whichever date happened to sort next to it.
        """
        return (self.recorded_at, self.run_id)


class ReportHistory:
    """A directory of past quality reports, one subdirectory per stage.

    Deliberately files-and-JSON rather than a database. The reports already
    land on disk beside the data they describe; a history that needs a service
    running to be read is a history nobody reads during an incident, which is
    the only time it matters.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def stage_dir(self, stage: str) -> Path:
        return self.root / stage

    def path_for(self, stage: str, run_id: str) -> Path:
        return self.stage_dir(stage) / f"{run_id}{_SUFFIX}"

    def record(
        self,
        report: QualityReport,
        *,
        run_id: str | None = None,
        now: datetime | None = None,
    ) -> HistoryEntry:
        """Write a report into the history under ``run_id``.

        Recording the same run id twice overwrites, which is the behaviour a
        retry wants: Airflow clears a failed task and runs it again under the
        same logical date, and the history should then hold what the retry
        found rather than two entries disagreeing about one run.
        """
        entry = HistoryEntry(
            run_id=run_id or new_run_id(now=now),
            stage=report.stage,
            recorded_at=_timestamp(now=now),
            report=report,
        )
        path = self.path_for(entry.stage, entry.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(entry.to_json(), encoding="utf-8")
        logger.debug("recorded %s report for run %s at %s", entry.stage, entry.run_id, path)
        return entry

    def entries(self, stage: str) -> tuple[HistoryEntry, ...]:
        """Every recorded run for a stage, oldest first.

        Unreadable files are skipped with a warning rather than raising. A
        half-written artifact from a run that was killed mid-write should cost
        you one line of history, not the ability to read the rest of it.
        """
        return tuple(sorted(self._load_all(stage), key=lambda entry: entry.sort_key))

    def _load_all(self, stage: str) -> Iterator[HistoryEntry]:
        directory = self.stage_dir(stage)
        if not directory.is_dir():
            return
        for path in sorted(directory.glob(f"*{_SUFFIX}")):
            try:
                yield HistoryEntry.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, KeyError, TypeError):
                logger.warning("skipping unreadable history entry %s", path)

    def latest(self, stage: str, *, before: str | None = None) -> HistoryEntry | None:
        """The most recent recorded run, or the one preceding ``before``.

        A ``before`` that names no recorded run is not an error: it is the
        ordinary case of asking "what came before the run I am in the middle
        of", which has not been written yet. The latest entry is then the
        answer, since everything on disk precedes it.
        """
        entries = self.entries(stage)
        cutoff = next((entry for entry in entries if entry.run_id == before), None)
        if cutoff is not None:
            entries = tuple(entry for entry in entries if entry.sort_key < cutoff.sort_key)
        return entries[-1] if entries else None

    def prune(self, stage: str, *, keep: int = DEFAULT_KEEP) -> tuple[str, ...]:
        """Drop all but the newest ``keep`` runs. Returns the ids removed."""
        if keep < 0:
            raise ValueError(f"keep must be non-negative, got {keep}")

        entries = self.entries(stage)
        doomed = entries[: max(len(entries) - keep, 0)]
        for entry in doomed:
            self.path_for(stage, entry.run_id).unlink(missing_ok=True)
        if doomed:
            logger.debug("pruned %d %s history entries", len(doomed), stage)
        return tuple(entry.run_id for entry in doomed)

    def stages(self) -> tuple[str, ...]:
        if not self.root.is_dir():
            return ()
        return tuple(sorted(child.name for child in self.root.iterdir() if child.is_dir()))


@dataclass(frozen=True)
class RuleTrend:
    """How one rule moved between two runs."""

    name: str
    kind: str
    previous_pass_rate: float | None = None
    current_pass_rate: float | None = None
    previous_passed: bool | None = None
    current_passed: bool | None = None
    severity: str | None = None

    @property
    def delta(self) -> float | None:
        """Change in pass rate, or ``None`` when either side has no rate.

        Dataset rules -- uniqueness, row counts -- are pass/fail with no rate
        to subtract, so they trend on their verdict alone.
        """
        if self.previous_pass_rate is None or self.current_pass_rate is None:
            return None
        return self.current_pass_rate - self.previous_pass_rate

    @property
    def broke(self) -> bool:
        """Passed last run, failing now. The regression that has already bitten."""
        return self.previous_passed is True and self.current_passed is False

    @property
    def is_regression(self) -> bool:
        return self.kind == REGRESSED

    def summary(self) -> str:
        marker = {
            REGRESSED: "DOWN",
            IMPROVED: " UP ",
            STEADY: "  = ",
            NEW: "NEW ",
            DROPPED: "GONE",
        }.get(self.kind, "  ? ")

        if self.kind == NEW:
            rate = f"{self.current_pass_rate:.4%}" if self.current_pass_rate is not None else "-"
            detail = f"not checked last run, now {rate}"
        elif self.kind == DROPPED:
            rate = f"{self.previous_pass_rate:.4%}" if self.previous_pass_rate is not None else "-"
            detail = f"was {rate}, no longer checked"
        elif self.delta is None:
            was = "pass" if self.previous_passed else "fail"
            now = "pass" if self.current_passed else "fail"
            detail = f"{was} -> {now}"
        else:
            detail = (
                f"{self.previous_pass_rate:.4%} -> {self.current_pass_rate:.4%} ({self.delta:+.4%})"
            )
        if self.broke:
            detail += ", now below its threshold"
        return f"[{marker}] {self.name}: {detail}"


@dataclass(frozen=True)
class TrendReport:
    """Every rule's movement between two runs of one stage."""

    stage: str
    previous_run_id: str
    current_run_id: str
    previous_rows: int
    current_rows: int
    rules: tuple[RuleTrend, ...] = ()
    tolerance: float = DEFAULT_TOLERANCE

    @property
    def regressions(self) -> tuple[RuleTrend, ...]:
        return tuple(trend for trend in self.rules if trend.kind == REGRESSED)

    @property
    def improvements(self) -> tuple[RuleTrend, ...]:
        return tuple(trend for trend in self.rules if trend.kind == IMPROVED)

    @property
    def broken(self) -> tuple[RuleTrend, ...]:
        """Rules that crossed their threshold since last run."""
        return tuple(trend for trend in self.rules if trend.broke)

    @property
    def stable(self) -> bool:
        """True when nothing regressed. Improvements and new rules are not instability."""
        return not self.regressions

    @property
    def row_delta(self) -> int:
        return self.current_rows - self.previous_rows

    @property
    def row_change(self) -> float | None:
        """Row count movement as a fraction of the previous run.

        Volume is the check no rule set covers: every rule can pass at 100% on
        a file that arrived with a third of its rows. ``None`` when the
        previous run had no rows, since the ratio is undefined rather than
        infinite.
        """
        if self.previous_rows == 0:
            return None
        return self.row_delta / self.previous_rows

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "previous_run_id": self.previous_run_id,
            "current_run_id": self.current_run_id,
            "previous_rows": self.previous_rows,
            "current_rows": self.current_rows,
            "row_delta": self.row_delta,
            "row_change": self.row_change,
            "tolerance": self.tolerance,
            "stable": self.stable,
            "regression_count": len(self.regressions),
            "rules": [asdict(trend) for trend in self.rules],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_text(self) -> str:
        verdict = "STABLE" if self.stable else "REGRESSED"
        change = f" ({self.row_change:+.2%})" if self.row_change is not None else ""
        lines = [
            f"quality trend :: {self.stage} :: {verdict}",
            f"  {self.previous_run_id} -> {self.current_run_id}"
            f"   rows: {self.previous_rows:,} -> {self.current_rows:,}{change}"
            f"   regressions: {len(self.regressions)}   tolerance: {self.tolerance:.2%}",
        ]
        if not self.rules:
            lines.append("  no rules in common")
        lines.extend(f"  {trend.summary()}" for trend in self.rules)
        return "\n".join(lines)

    def to_markdown(self) -> str:
        header = "| rule | movement | previous | current | delta |\n| --- | --- | --- | --- | --- |"
        rows = []
        for trend in self.rules:
            previous = (
                f"{trend.previous_pass_rate:.4%}" if trend.previous_pass_rate is not None else "-"
            )
            current = (
                f"{trend.current_pass_rate:.4%}" if trend.current_pass_rate is not None else "-"
            )
            delta = f"{trend.delta:+.4%}" if trend.delta is not None else "-"
            kind = f"**{trend.kind}**" if trend.kind == REGRESSED else trend.kind
            rows.append(f"| `{trend.name}` | {kind} | {previous} | {current} | {delta} |")
        return "\n".join([header, *rows])


def _grade(
    previous: RuleResult,
    current: RuleResult,
    *,
    tolerance: float,
) -> str:
    """Classify one rule's movement between two runs.

    Crossing the threshold is a regression whatever the size of the move: a
    rule that was passing and is now failing has changed what the pipeline
    does, and calling a 0.1% slip "steady" on the day it started blocking runs
    would be a report that contradicts the gate standing next to it.
    """
    if previous.passed and not current.passed:
        return REGRESSED
    if not previous.passed and current.passed:
        return IMPROVED

    if previous.pass_rate is None or current.pass_rate is None:
        return STEADY

    delta = current.pass_rate - previous.pass_rate
    if delta < -tolerance:
        return REGRESSED
    if delta > tolerance:
        return IMPROVED
    return STEADY


def compare_reports(
    previous: QualityReport,
    current: QualityReport,
    *,
    previous_run_id: str = "previous",
    current_run_id: str = "current",
    tolerance: float = DEFAULT_TOLERANCE,
) -> TrendReport:
    """Diff two reports for the same stage, rule by rule.

    Args:
        previous: The earlier run's report.
        current: The report being judged.
        previous_run_id: Named in the output so a trend says which two runs it
            compared, rather than leaving that to whoever kept the shell
            history.
        current_run_id: As above, for the later run.
        tolerance: How far a pass rate may move before it counts. See
            :data:`DEFAULT_TOLERANCE`.

    Raises:
        ValueError: The two reports describe different stages. Comparing
            bronze against silver would produce a page of ``new`` and
            ``dropped`` rules that says nothing about either.
    """
    if previous.stage != current.stage:
        raise ValueError(
            f"cannot compare stage {previous.stage!r} against {current.stage!r}; "
            "a trend is one stage over time"
        )
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")

    before = {result.name: result for result in previous.results}
    after = {result.name: result for result in current.results}

    trends = [
        RuleTrend(
            name=name,
            kind=_grade(before[name], after[name], tolerance=tolerance) if name in before else NEW,
            previous_pass_rate=before[name].pass_rate if name in before else None,
            current_pass_rate=after[name].pass_rate,
            previous_passed=before[name].passed if name in before else None,
            current_passed=after[name].passed,
            severity=after[name].severity,
        )
        for name in after
    ]
    trends.extend(
        RuleTrend(
            name=name,
            kind=DROPPED,
            previous_pass_rate=result.pass_rate,
            previous_passed=result.passed,
            severity=result.severity,
        )
        for name, result in before.items()
        if name not in after
    )

    return TrendReport(
        stage=current.stage,
        previous_run_id=previous_run_id,
        current_run_id=current_run_id,
        previous_rows=previous.total_rows,
        current_rows=current.total_rows,
        rules=tuple(trends),
        tolerance=tolerance,
    )


def compare_runs(
    history: ReportHistory,
    stage: str,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> TrendReport | None:
    """Compare a stage's two most recent runs.

    Returns ``None`` when fewer than two runs have been recorded -- the first
    run of a new pipeline has nothing to be compared against, and that is not
    an error condition, it is Tuesday.
    """
    entries = history.entries(stage)
    if len(entries) < 2:
        return None

    previous, current = entries[-2], entries[-1]
    return compare_reports(
        previous.report,
        current.report,
        previous_run_id=previous.run_id,
        current_run_id=current.run_id,
        tolerance=tolerance,
    )
