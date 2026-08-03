"""The sample generator: reproducibility, cleanliness, and injected defects."""

from __future__ import annotations

import pytest
from pyspark.sql import functions as F

from tripetl.quality.engine import evaluate
from tripetl.quality.rulesets import bronze_ruleset
from tripetl.sources import DEFECTS, generate_sample, read_raw
from tripetl.transforms.clean import clean

ROWS = 600


def test_the_same_seed_reproduces_the_same_rows(spark):
    first = generate_sample(spark, rows=200, seed=7).collect()
    second = generate_sample(spark, rows=200, seed=7).collect()
    assert first == second


def test_a_different_seed_produces_different_rows(spark):
    first = generate_sample(spark, rows=200, seed=7).collect()
    second = generate_sample(spark, rows=200, seed=8).collect()
    assert first != second


def test_a_pristine_sample_passes_every_bronze_rule(spark):
    """The zero-defect setting is what makes this usable as a fixture."""
    df = clean(generate_sample(spark, rows=ROWS, seed=1, dirty_rate=0.0))
    report = evaluate(df, bronze_ruleset())

    assert report.passed, report.to_text()
    assert report.failures == ()
    assert report.total_rows == ROWS


def test_a_pristine_sample_contains_no_duplicates(spark):
    df = clean(generate_sample(spark, rows=ROWS, seed=1, dirty_rate=0.0))
    assert df.select("trip_id").distinct().count() == ROWS


def test_duplicates_are_added_alongside_defects(spark):
    df = generate_sample(spark, rows=ROWS, seed=1, dirty_rate=0.2)
    # rows + rows * dirty_rate / 2
    assert df.count() == ROWS + int(ROWS * 0.2 / 2)


def test_every_declared_defect_actually_appears(spark):
    """Each defect maps to a bronze rule; if one never fires, that rule is untested."""
    df = generate_sample(spark, rows=4_000, seed=3, dirty_rate=0.5)
    counts = df.select(
        F.sum(F.col("passenger_count").isNull().cast("int")).alias("null_passenger_count"),
        F.sum((F.col("fare_amount") < 0).cast("int")).alias("negative_fare"),
        F.sum((F.col("tpep_pickup_datetime") > F.col("tpep_dropoff_datetime")).cast("int")).alias(
            "reversed_timestamps"
        ),
        F.sum((F.col("trip_distance") > 200).cast("int")).alias("impossible_distance"),
        F.sum((F.col("payment_type") == 99).cast("int")).alias("unknown_payment_type"),
        F.sum((F.col("PULocationID") == 999).cast("int")).alias("unknown_zone"),
    ).collect()[0]

    for defect in DEFECTS:
        assert counts[defect] > 0, f"defect {defect} was never injected"


def test_a_dirty_sample_trips_the_bronze_gate(spark):
    df = clean(generate_sample(spark, rows=2_000, seed=5, dirty_rate=0.5))
    report = evaluate(df, bronze_ruleset())

    assert not report.passed
    assert len(report.errors) > 0


def test_generated_rows_match_the_published_schema(spark):
    from tripetl.schema import RAW_TRIP_SCHEMA

    df = generate_sample(spark, rows=10, seed=1)
    assert df.schema == RAW_TRIP_SCHEMA


def test_zero_rows_is_allowed(spark):
    assert generate_sample(spark, rows=0, seed=1).count() == 0


def test_an_out_of_range_dirty_rate_is_rejected(spark):
    with pytest.raises(ValueError, match="dirty_rate"):
        generate_sample(spark, rows=10, seed=1, dirty_rate=1.5)


def test_read_raw_applies_the_declared_schema(spark, tmp_path):
    from tripetl.schema import RAW_TRIP_SCHEMA

    path = str(tmp_path / "raw")
    generated = generate_sample(spark, rows=50, seed=2)
    expected_rows = generated.count()
    generated.write.parquet(path)

    loaded = read_raw(spark, path)
    assert loaded.schema == RAW_TRIP_SCHEMA
    assert loaded.count() == expected_rows
