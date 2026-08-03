"""The evaluation engine and the quarantine split."""

from __future__ import annotations

from pyspark.sql import functions as F

from tripetl.quality import rules as r
from tripetl.quality.engine import FAILURE_COLUMN, evaluate, split
from tripetl.quality.rules import RuleSet, Severity

SCHEMA = "id int, value int, code int"


def _ruleset(**kwargs) -> RuleSet:
    return RuleSet(
        name="demo",
        row_rules=(
            r.not_null("value", **kwargs),
            r.in_range("value", 0, 100, **kwargs),
            r.accepted_values("code", (1, 2), **kwargs),
        ),
    )


def test_pass_rates_are_counted_per_rule(make_df):
    df = make_df(
        [(1, 10, 1), (2, 20, 2), (3, None, 1), (4, 500, 9)],
        SCHEMA,
    )
    report = evaluate(df, _ruleset())

    assert report.total_rows == 4
    by_name = {result.name: result for result in report.results}

    assert by_name["not_null[value]"].failing_rows == 1
    assert by_name["not_null[value]"].pass_rate == 0.75
    # Both the null and the out-of-range value fail the range check.
    assert by_name["in_range[value]"].failing_rows == 2
    assert by_name["accepted_values[code]"].failing_rows == 1


def test_a_rule_passes_while_it_stays_above_its_threshold(make_df):
    rows = [(i, i, 1) for i in range(99)] + [(99, None, 1)]
    df = make_df(rows, SCHEMA)

    lenient = evaluate(df, RuleSet("x", (r.not_null("value", threshold=0.99),)))
    strict = evaluate(df, RuleSet("x", (r.not_null("value", threshold=1.0),)))

    assert lenient.passed
    assert not strict.passed
    assert strict.errors[0].pass_rate == 0.99


def test_warnings_do_not_make_a_report_fail(make_df):
    df = make_df([(1, None, 1)], SCHEMA)
    report = evaluate(df, RuleSet("x", (r.not_null("value", severity=Severity.WARN),)))

    assert report.passed
    assert len(report.warnings) == 1
    assert report.errors == ()


def test_an_empty_dataset_satisfies_row_rules_but_trips_the_row_count(make_df):
    """Vacuous truth is the honest answer; the count rule is what catches it."""
    df = make_df([], SCHEMA)
    ruleset = RuleSet(
        name="demo",
        row_rules=(r.not_null("value"),),
        dataset_rules=(r.row_count_at_least(1),),
    )
    report = evaluate(df, ruleset)

    by_name = {result.name: result for result in report.results}
    assert report.total_rows == 0
    assert by_name["not_null[value]"].passed
    assert by_name["not_null[value]"].pass_rate == 1.0
    assert not by_name["row_count_at_least"].passed
    assert not report.passed


def test_dataset_rules_detect_duplicates(make_df):
    df = make_df([(1, 1, 1), (1, 2, 1), (2, 3, 1)], SCHEMA)
    report = evaluate(df, RuleSet("x", dataset_rules=(r.unique("id"),)))

    result = report.results[0]
    assert result.kind == "dataset"
    assert result.metric == 1.0
    assert not result.passed


def _spark_jobs_for(spark, df, ruleset, tag: str) -> int:
    """Run an evaluation and count the Spark jobs it triggered."""
    context = spark.sparkContext
    context.setJobGroup(tag, tag)
    try:
        evaluate(df, ruleset)
    finally:
        context.setLocalProperty("spark.jobGroup.id", None)
    return len(context.statusTracker().getJobIdsForGroup(tag))


def test_evaluation_cost_does_not_grow_with_the_number_of_rules(spark, make_df):
    """The single-pass claim in the engine docstring, measured.

    Comparing two rule sets rather than asserting an absolute job count is the
    honest form of the claim -- and the robust one, since adaptive execution is
    free to split a stage. Twelve row rules must cost exactly what one costs;
    if the engine ever regresses to filtering per rule, this fails loudly.
    """
    df = make_df([(index, index, 1) for index in range(50)], SCHEMA)
    dataset_rules = (r.unique("id"), r.row_count_at_least(1))

    few = RuleSet("few", (r.in_range("value", 0, 100),), dataset_rules)
    many = RuleSet(
        "many",
        tuple(r.in_range("value", 0, 100 + index) for index in range(12)),
        dataset_rules,
    )

    assert _spark_jobs_for(spark, df, many, "cost-many") == _spark_jobs_for(
        spark, df, few, "cost-few"
    )


def test_split_records_every_rule_a_row_broke(make_df):
    df = make_df([(1, 10, 1), (2, None, 9)], SCHEMA)
    ruleset = _ruleset()

    clean, quarantined = split(df, ruleset.quarantine_rules())

    assert clean.count() == 1
    assert FAILURE_COLUMN not in clean.columns
    assert clean.collect()[0]["id"] == 1

    bad = quarantined.collect()
    assert len(bad) == 1
    # Not just the first failure: all three, so the row can be diagnosed.
    assert sorted(bad[0][FAILURE_COLUMN]) == [
        "accepted_values[code]",
        "in_range[value]",
        "not_null[value]",
    ]


def test_split_keeps_the_original_columns_on_quarantined_rows(make_df):
    df = make_df([(7, None, 1)], SCHEMA)
    _, quarantined = split(df, (r.not_null("value"),))

    row = quarantined.collect()[0]
    assert row["id"] == 7
    assert row["code"] == 1


def test_split_without_rules_yields_an_empty_typed_quarantine(make_df):
    df = make_df([(1, 1, 1)], SCHEMA)
    clean, quarantined = split(df, ())

    assert clean.count() == 1
    assert quarantined.count() == 0
    # The column must exist even when empty, so a downstream reader of the
    # quarantine dataset sees a stable schema.
    assert FAILURE_COLUMN in quarantined.columns


def test_rows_failing_a_non_quarantine_rule_still_flow_onward(make_df):
    """A WARN-only rule should report without diverting the row."""
    df = make_df([(1, None, 1)], SCHEMA)
    ruleset = RuleSet(
        name="demo",
        row_rules=(r.not_null("value", severity=Severity.WARN, quarantine=False),),
    )
    clean, quarantined = split(df, ruleset.quarantine_rules())

    assert clean.count() == 1
    assert quarantined.count() == 0


def test_a_null_producing_custom_predicate_counts_as_a_failure(make_df):
    """`satisfies` hands null-safety to the caller; the engine must not crash."""
    df = make_df([(1, None, 1)], SCHEMA)
    rule = r.satisfies("raw_comparison", F.col("value") > F.lit(0), "value must be positive")
    report = evaluate(df, RuleSet("x", (rule,)))

    assert report.results[0].failing_rows == 1
    assert not report.passed
