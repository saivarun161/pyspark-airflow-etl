"""Declarative data-quality rules.

A rule is a *predicate over a row* plus the policy attached to it: how severe a
violation is, and what fraction of rows must satisfy it. Expressing rules as
Spark ``Column`` objects rather than Python callables is what lets the engine
check thirty rules in a single pass over the data -- see
:mod:`tripetl.quality.engine`.

Every builder here returns a predicate that is *never null*. A three-valued
predicate is a trap: ``col < 100`` is null when ``col`` is null, and whether
that counts as a pass or a failure then depends on how the engine happens to
aggregate it. Guarding each predicate with an explicit ``isNotNull`` makes
null-handling a property of the rule you can read, not of the engine.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from pyspark.sql import Column
from pyspark.sql import functions as F


class Severity(StrEnum):
    """How a gate should react when a rule falls below its threshold.

    A ``StrEnum`` so a severity survives the round trip through a JSON report
    without a custom encoder, and still compares equal to the plain string a
    reader of that report would write.
    """

    ERROR = "error"
    WARN = "warn"


# ``eq=False`` is load-bearing. ``Column.__eq__`` builds a new Column rather
# than returning a bool, so a generated ``__eq__`` would produce a Column where
# Python expects a truth value and break any comparison, ``in`` test, or
# deduplication involving a Rule.
@dataclass(frozen=True, eq=False)
class Rule:
    """A row-level expectation.

    Attributes:
        predicate: True for a row that satisfies the rule. Must not be null.
        threshold: The minimum fraction of rows that must pass, in ``[0, 1]``.
            ``1.0`` means "no exceptions".
        quarantine: Whether failing rows should be diverted to the quarantine
            dataset rather than flowing on to the next layer.
    """

    name: str
    predicate: Column
    description: str
    severity: Severity = Severity.ERROR
    threshold: float = 1.0
    quarantine: bool = True
    columns: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"{self.name}: threshold must be in [0, 1], got {self.threshold}")


@dataclass(frozen=True, eq=False)
class DatasetRule:
    """An expectation about the dataset as a whole rather than a single row.

    Uniqueness and row counts cannot be phrased as a row predicate -- no row
    can tell you on its own whether its key is duplicated. These are evaluated
    in a second aggregation, and they never quarantine anything: there is no
    single offending row to divert.
    """

    name: str
    metric: Column
    check: str
    description: str
    severity: Severity = Severity.ERROR
    columns: tuple[str, ...] = field(default=())

    def passes(self, value: float | None) -> bool:
        """Apply :attr:`check` to a measured metric value."""
        if value is None:
            return False
        target = float(self.check.split()[-1])
        op = self.check.split()[0]
        if op == "==":
            return value == target
        if op == "<=":
            return value <= target
        if op == ">=":
            return value >= target
        raise ValueError(f"{self.name}: unsupported check {self.check!r}")


@dataclass(frozen=True, eq=False)
class RuleSet:
    """The rules that guard one stage boundary."""

    name: str
    row_rules: tuple[Rule, ...] = ()
    dataset_rules: tuple[DatasetRule, ...] = ()

    def __len__(self) -> int:
        return len(self.row_rules) + len(self.dataset_rules)

    def quarantine_rules(self) -> tuple[Rule, ...]:
        return tuple(rule for rule in self.row_rules if rule.quarantine)


# ---------------------------------------------------------------------------
# Row-rule builders
# ---------------------------------------------------------------------------


def not_null(
    column: str,
    *,
    severity: Severity = Severity.ERROR,
    threshold: float = 1.0,
    quarantine: bool = True,
) -> Rule:
    """Require ``column`` to be present."""
    return Rule(
        name=f"not_null[{column}]",
        predicate=F.col(column).isNotNull(),
        description=f"{column} must not be null",
        severity=severity,
        threshold=threshold,
        quarantine=quarantine,
        columns=(column,),
    )


def in_range(
    column: str,
    minimum: float,
    maximum: float,
    *,
    allow_null: bool = False,
    severity: Severity = Severity.ERROR,
    threshold: float = 1.0,
    quarantine: bool = True,
) -> Rule:
    """Require ``column`` to fall within ``[minimum, maximum]`` inclusive."""
    col = F.col(column)
    predicate = col.isNotNull() & (col >= F.lit(minimum)) & (col <= F.lit(maximum))
    if allow_null:
        predicate = col.isNull() | predicate
    nullness = "null or " if allow_null else ""
    return Rule(
        name=f"in_range[{column}]",
        predicate=predicate,
        description=f"{column} must be {nullness}between {minimum} and {maximum}",
        severity=severity,
        threshold=threshold,
        quarantine=quarantine,
        columns=(column,),
    )


def non_negative(
    column: str,
    *,
    allow_null: bool = False,
    severity: Severity = Severity.ERROR,
    threshold: float = 1.0,
    quarantine: bool = True,
) -> Rule:
    """Require ``column`` to be zero or greater."""
    col = F.col(column)
    predicate = col.isNotNull() & (col >= F.lit(0))
    if allow_null:
        predicate = col.isNull() | predicate
    nullness = "null or " if allow_null else ""
    return Rule(
        name=f"non_negative[{column}]",
        predicate=predicate,
        description=f"{column} must be {nullness}>= 0",
        severity=severity,
        threshold=threshold,
        quarantine=quarantine,
        columns=(column,),
    )


def accepted_values(
    column: str,
    values: Sequence[object],
    *,
    allow_null: bool = False,
    severity: Severity = Severity.ERROR,
    threshold: float = 1.0,
    quarantine: bool = True,
) -> Rule:
    """Require ``column`` to be drawn from a known set of codes."""
    col = F.col(column)
    predicate = col.isNotNull() & col.isin(list(values))
    if allow_null:
        predicate = col.isNull() | predicate
    shown = ", ".join(str(value) for value in values)
    return Rule(
        name=f"accepted_values[{column}]",
        predicate=predicate,
        description=f"{column} must be one of ({shown})",
        severity=severity,
        threshold=threshold,
        quarantine=quarantine,
        columns=(column,),
    )


def ordered(
    earlier: str,
    later: str,
    *,
    strict: bool = False,
    severity: Severity = Severity.ERROR,
    threshold: float = 1.0,
    quarantine: bool = True,
) -> Rule:
    """Require ``earlier`` to precede ``later``, both being present."""
    lhs, rhs = F.col(earlier), F.col(later)
    comparison = (lhs < rhs) if strict else (lhs <= rhs)
    relation = "strictly before" if strict else "at or before"
    return Rule(
        name=f"ordered[{earlier}<{later}]",
        predicate=lhs.isNotNull() & rhs.isNotNull() & comparison,
        description=f"{earlier} must be {relation} {later}",
        severity=severity,
        threshold=threshold,
        quarantine=quarantine,
        columns=(earlier, later),
    )


def satisfies(
    name: str,
    predicate: Column,
    description: str,
    *,
    severity: Severity = Severity.ERROR,
    threshold: float = 1.0,
    quarantine: bool = True,
    columns: Iterable[str] = (),
) -> Rule:
    """Escape hatch for an expectation the builders above do not cover.

    The caller owns null-safety here; wrap the expression in ``coalesce`` if it
    can evaluate to null.
    """
    return Rule(
        name=name,
        predicate=predicate,
        description=description,
        severity=severity,
        threshold=threshold,
        quarantine=quarantine,
        columns=tuple(columns),
    )


# ---------------------------------------------------------------------------
# Dataset-rule builders
# ---------------------------------------------------------------------------


def unique(column: str, *, severity: Severity = Severity.ERROR) -> DatasetRule:
    """Require ``column`` to hold no duplicate non-null values."""
    return DatasetRule(
        name=f"unique[{column}]",
        metric=F.count(F.col(column)) - F.count_distinct(F.col(column)),
        check="== 0",
        description=f"{column} must be unique",
        severity=severity,
        columns=(column,),
    )


def row_count_at_least(minimum: int, *, severity: Severity = Severity.ERROR) -> DatasetRule:
    """Require the dataset to be at least ``minimum`` rows.

    Guards the failure mode where an upstream extract silently produces an
    empty file: every row-level rule passes vacuously, and a pipeline that only
    checks rates publishes an empty mart with a clean bill of health.
    """
    return DatasetRule(
        name="row_count_at_least",
        metric=F.count(F.lit(1)).cast("double"),
        check=f">= {minimum}",
        description=f"dataset must contain at least {minimum} rows",
        severity=severity,
    )
