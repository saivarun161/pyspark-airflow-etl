"""Evaluating rule sets, and splitting bad rows out of a DataFrame."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from tripetl.quality.report import QualityReport, RuleResult
from tripetl.quality.rules import Rule, RuleSet

#: Column added to quarantined rows naming the rules they violated.
FAILURE_COLUMN = "_dq_failures"


def evaluate(df: DataFrame, ruleset: RuleSet, *, stage: str | None = None) -> QualityReport:
    """Check every rule in ``ruleset`` against ``df``.

    The whole rule set costs at most two Spark jobs regardless of how many
    rules it holds: one aggregation that counts passing rows for all row rules
    at once, and -- only if the set has any -- a second for the dataset rules.
    The naive alternative, filtering and counting per rule, is one full scan
    per rule and gets slow exactly when you are tempted to add more checks.

    Callers evaluating several rule sets against the same DataFrame should
    ``cache()`` it first; the engine does not, because it cannot know whether
    the caller is about to reuse the data or discard it.
    """
    stage = stage or ruleset.name
    row_rules = ruleset.row_rules

    aggregations = [F.count(F.lit(1)).alias("_total")]
    for index, rule in enumerate(row_rules):
        # coalesce is belt-and-braces: the builders guarantee a non-null
        # predicate, but `satisfies()` hands that guarantee to the caller.
        safe = F.coalesce(rule.predicate, F.lit(False))
        aggregations.append(F.sum(safe.cast("long")).alias(f"_row_{index}"))

    measured = df.agg(*aggregations).first()
    total_rows = int(measured["_total"]) if measured is not None else 0

    results: list[RuleResult] = []
    for index, rule in enumerate(row_rules):
        raw = measured[f"_row_{index}"] if measured is not None else None
        passing = int(raw) if raw is not None else 0
        # An empty dataset satisfies every row rule vacuously. That is the
        # mathematically honest answer; `row_count_at_least` is the rule that
        # exists to catch an empty extract.
        pass_rate = 1.0 if total_rows == 0 else passing / total_rows
        results.append(
            RuleResult(
                name=rule.name,
                description=rule.description,
                severity=rule.severity.value,
                kind="row",
                passed=pass_rate >= rule.threshold,
                columns=rule.columns,
                threshold=rule.threshold,
                pass_rate=pass_rate,
                failing_rows=total_rows - passing,
                total_rows=total_rows,
            )
        )

    if ruleset.dataset_rules:
        dataset_aggs = [
            rule.metric.cast("double").alias(f"_ds_{index}")
            for index, rule in enumerate(ruleset.dataset_rules)
        ]
        dataset_row = df.agg(*dataset_aggs).first()
        for index, dataset_rule in enumerate(ruleset.dataset_rules):
            raw_metric = dataset_row[f"_ds_{index}"] if dataset_row is not None else None
            metric = float(raw_metric) if raw_metric is not None else None
            results.append(
                RuleResult(
                    name=dataset_rule.name,
                    description=dataset_rule.description,
                    severity=dataset_rule.severity.value,
                    kind="dataset",
                    passed=dataset_rule.passes(metric),
                    columns=dataset_rule.columns,
                    metric=metric,
                    total_rows=total_rows,
                )
            )

    return QualityReport(
        stage=stage,
        ruleset=ruleset.name,
        total_rows=total_rows,
        results=tuple(results),
    )


def split(df: DataFrame, rules: tuple[Rule, ...]) -> tuple[DataFrame, DataFrame]:
    """Partition ``df`` into rows that satisfy every rule and rows that do not.

    Quarantined rows keep all their original columns and gain
    :data:`FAILURE_COLUMN`, an array naming every rule they broke -- not just
    the first. Dropping bad rows is easy and quietly destroys the evidence you
    need to fix the source; writing them out with their reasons attached turns
    a data-quality incident into something you can actually investigate.
    """
    if not rules:
        empty = df.limit(0).withColumn(FAILURE_COLUMN, F.array().cast("array<string>"))
        return df, empty

    failures = F.array_compact(
        F.array(
            *[F.when(~F.coalesce(rule.predicate, F.lit(False)), F.lit(rule.name)) for rule in rules]
        )
    )
    tagged = df.withColumn(FAILURE_COLUMN, failures)
    clean = tagged.filter(F.size(F.col(FAILURE_COLUMN)) == 0).drop(FAILURE_COLUMN)
    quarantined = tagged.filter(F.size(F.col(FAILURE_COLUMN)) > 0)
    return clean, quarantined
