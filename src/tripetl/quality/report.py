"""Results of a quality evaluation, and how to render them."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields

from tripetl.quality.rules import Severity


@dataclass(frozen=True)
class RuleResult:
    """The outcome of one rule against one dataset.

    Deliberately made of primitives only: a report is written to disk as JSON
    at every stage boundary, and that artifact is often the only evidence left
    when someone asks in a fortnight why Tuesday's run was blocked.
    """

    name: str
    description: str
    severity: str
    kind: str
    passed: bool
    columns: tuple[str, ...] = ()
    threshold: float | None = None
    pass_rate: float | None = None
    failing_rows: int | None = None
    total_rows: int | None = None
    metric: float | None = None

    @property
    def is_blocking(self) -> bool:
        """True when this result should stop the pipeline."""
        return not self.passed and self.severity == Severity.ERROR.value

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> RuleResult:
        """Rebuild a result from its JSON form.

        The inverse of :func:`dataclasses.asdict`, with one fixup: JSON has no
        tuples, so ``columns`` comes back as a list and has to be restored.
        Without that a round-tripped result compares unequal to the one that
        was written, which is exactly the comparison history is for.
        """
        fields = {key: payload[key] for key in _RESULT_FIELDS if key in payload}
        columns = fields.get("columns") or ()
        return cls(**{**fields, "columns": tuple(columns)})  # type: ignore[arg-type]

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        if self.kind == "row" and self.pass_rate is not None:
            detail = f"{self.pass_rate:.4%} pass"
            if self.threshold is not None:
                detail += f" (need {self.threshold:.2%})"
            if self.failing_rows:
                detail += f", {self.failing_rows:,} bad rows"
        else:
            detail = f"metric={self.metric:g}" if self.metric is not None else "no metric"
        return f"[{status}] {self.name}: {detail}"


#: Read off the dataclass rather than restated, so a new field cannot be added
#: to a result and silently dropped on the way back in from JSON.
_RESULT_FIELDS = tuple(field.name for field in fields(RuleResult))


@dataclass(frozen=True)
class QualityReport:
    """Every rule result for one stage, plus the verdict."""

    stage: str
    ruleset: str
    total_rows: int
    results: tuple[RuleResult, ...]

    @property
    def passed(self) -> bool:
        """True when nothing blocking failed. Warnings do not affect this."""
        return not self.errors

    @property
    def errors(self) -> tuple[RuleResult, ...]:
        return tuple(r for r in self.results if r.is_blocking)

    @property
    def warnings(self) -> tuple[RuleResult, ...]:
        return tuple(r for r in self.results if not r.passed and r.severity == Severity.WARN.value)

    @property
    def failures(self) -> tuple[RuleResult, ...]:
        return tuple(r for r in self.results if not r.passed)

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "ruleset": self.ruleset,
            "total_rows": self.total_rows,
            "passed": self.passed,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "results": [asdict(r) for r in self.results],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=list)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> QualityReport:
        """Rebuild a report from the JSON written at a stage boundary.

        ``passed``, ``error_count`` and ``warning_count`` are in the payload
        for whoever reads the file by eye, but they are derived here rather
        than restored: recomputing them from the results is what catches a
        hand-edited or half-written artifact instead of trusting its verdict.
        """
        results = payload.get("results") or ()
        return cls(
            stage=str(payload["stage"]),
            ruleset=str(payload["ruleset"]),
            total_rows=int(payload["total_rows"]),  # type: ignore[arg-type]
            results=tuple(RuleResult.from_dict(r) for r in results),  # type: ignore[union-attr]
        )

    def to_text(self) -> str:
        verdict = "PASSED" if self.passed else "FAILED"
        lines = [
            f"quality report :: {self.stage} :: {verdict}",
            f"  rows checked: {self.total_rows:,}   rules: {len(self.results)}"
            f"   errors: {len(self.errors)}   warnings: {len(self.warnings)}",
        ]
        lines.extend(f"  {r.summary()}" for r in self.results)
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Render as a table, for pasting into a CI summary or a ticket."""
        header = (
            "| rule | severity | status | pass rate | threshold | bad rows |\n"
            "| --- | --- | --- | --- | --- | --- |"
        )
        rows = []
        for r in self.results:
            rate = f"{r.pass_rate:.4%}" if r.pass_rate is not None else "-"
            threshold = f"{r.threshold:.2%}" if r.threshold is not None else "-"
            bad = f"{r.failing_rows:,}" if r.failing_rows is not None else "-"
            status = "pass" if r.passed else "**fail**"
            rows.append(f"| `{r.name}` | {r.severity} | {status} | {rate} | {threshold} | {bad} |")
        return "\n".join([header, *rows])
