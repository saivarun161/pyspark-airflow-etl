"""Rule builders: predicate semantics, null handling, and policy validation."""

from __future__ import annotations

import pytest
from pyspark.sql import functions as F

from tripetl.quality import rules as r
from tripetl.quality.rules import RuleSet, Severity


def _flags(df, rule):
    """Evaluate a rule's predicate row by row, preserving nulls."""
    return [row["p"] for row in df.select(rule.predicate.alias("p")).collect()]


def test_not_null_rejects_missing_values(make_df):
    df = make_df([(1,), (None,)], "value int")
    assert _flags(df, r.not_null("value")) == [True, False]


def test_in_range_is_inclusive_at_both_ends(make_df):
    df = make_df([(0,), (5,), (10,), (11,), (-1,)], "value int")
    assert _flags(df, r.in_range("value", 0, 10)) == [True, True, True, False, False]


def test_in_range_treats_null_as_failure_by_default(make_df):
    df = make_df([(None,)], "value int")
    assert _flags(df, r.in_range("value", 0, 10)) == [False]


def test_in_range_can_admit_nulls_explicitly(make_df):
    df = make_df([(None,), (99,)], "value int")
    assert _flags(df, r.in_range("value", 0, 10, allow_null=True)) == [True, False]


def test_non_negative_rejects_below_zero(make_df):
    df = make_df([(0.0,), (0.01,), (-0.01,), (None,)], "value double")
    assert _flags(df, r.non_negative("value")) == [True, True, False, False]
    assert _flags(df, r.non_negative("value", allow_null=True)) == [
        True,
        True,
        False,
        True,
    ]


def test_accepted_values_limits_to_the_known_set(make_df):
    df = make_df([(1,), (2,), (99,), (None,)], "code int")
    assert _flags(df, r.accepted_values("code", (1, 2))) == [True, True, False, False]


def test_ordered_compares_two_columns(make_df):
    df = make_df(
        [(1, 2), (2, 2), (3, 2), (None, 2)],
        "a int, b int",
    )
    assert _flags(df, r.ordered("a", "b")) == [True, True, False, False]
    assert _flags(df, r.ordered("a", "b", strict=True)) == [True, False, False, False]


# Built lazily: a Column cannot be constructed before a SparkContext exists,
# and parametrize arguments are evaluated at collection time.
@pytest.mark.parametrize(
    "build",
    [
        lambda: r.not_null("value"),
        lambda: r.in_range("value", 0, 10),
        lambda: r.in_range("value", 0, 10, allow_null=True),
        lambda: r.non_negative("value"),
        lambda: r.non_negative("value", allow_null=True),
        lambda: r.accepted_values("value", (1, 2)),
    ],
    ids=[
        "not_null",
        "in_range",
        "in_range_nullable",
        "non_negative",
        "non_negative_nullable",
        "accepted_values",
    ],
)
def test_builders_never_produce_a_null_predicate(make_df, build):
    """The engine's arithmetic depends on this; see rules module docstring."""
    df = make_df([(None,), (1,), (99,)], "value int")
    assert None not in _flags(df, build())


def test_threshold_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError, match="threshold"):
        r.not_null("value", threshold=1.5)
    with pytest.raises(ValueError, match="threshold"):
        r.not_null("value", threshold=-0.1)


def test_rules_stay_usable_as_ordinary_python_objects():
    """Guards the ``eq=False`` on Rule -- see the comment on the dataclass.

    A generated ``__eq__`` would delegate to ``Column.__eq__``, which returns a
    Column rather than a bool, and every one of these would raise or misbehave.
    """
    rule = r.not_null("value")
    other = r.not_null("other")

    assert rule == rule
    assert rule != other
    assert len({rule, other, rule}) == 2
    assert rule in [other, rule]


def test_dataset_rule_check_operators():
    unique = r.unique("id")
    assert unique.passes(0.0) is True
    assert unique.passes(3.0) is False
    # A missing metric cannot be evidence of correctness.
    assert unique.passes(None) is False

    at_least = r.row_count_at_least(100)
    assert at_least.passes(100.0) is True
    assert at_least.passes(99.0) is False


def test_dataset_rule_rejects_an_unknown_operator():
    rule = r.unique("id")
    broken = type(rule)(
        name=rule.name,
        metric=rule.metric,
        check="!= 0",
        description=rule.description,
    )
    with pytest.raises(ValueError, match="unsupported check"):
        broken.passes(1.0)


def test_ruleset_reports_size_and_quarantine_membership():
    ruleset = RuleSet(
        name="demo",
        row_rules=(
            r.not_null("a"),
            r.not_null("b", quarantine=False, severity=Severity.WARN),
        ),
        dataset_rules=(r.row_count_at_least(1),),
    )
    assert len(ruleset) == 3
    assert [rule.name for rule in ruleset.quarantine_rules()] == ["not_null[a]"]


def test_satisfies_passes_the_expression_through(make_df):
    df = make_df([(4,), (5,)], "value int")
    rule = r.satisfies(
        "even_value",
        F.col("value") % F.lit(2) == F.lit(0),
        "value must be even",
        columns=("value",),
    )
    assert _flags(df, rule) == [True, False]
    assert rule.columns == ("value",)
