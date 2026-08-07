"""Deriving the columns the mart actually reports on."""

from __future__ import annotations

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from tripetl.schema import AIRPORT_RATE_CODES, PARTITION_COLUMN
from tripetl.transforms.clean import pickup_date_expression

#: Hour-of-day buckets, as (label, first hour, last hour) inclusive.
TIME_OF_DAY_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("overnight", 0, 5),
    ("morning", 6, 11),
    ("afternoon", 12, 16),
    ("evening", 17, 21),
    ("night", 22, 23),
)


def _safe_divide(numerator: Column, denominator: Column) -> Column:
    """Divide, yielding null rather than infinity when the denominator is 0.

    Spark returns null for integer division by zero but ``Infinity`` for
    doubles, and ``Infinity`` propagates through ``avg()`` to poison an entire
    aggregate group. A null is excluded from the average instead, which is the
    honest treatment of "we could not compute this".
    """
    return F.when(denominator > F.lit(0), numerator / denominator)


def time_of_day_expression(hour: Column) -> Column:
    """Map an hour column onto :data:`TIME_OF_DAY_BUCKETS`."""
    expression = F.when(F.lit(False), F.lit(None).cast("string"))
    for label, first, last in TIME_OF_DAY_BUCKETS:
        expression = expression.when(hour.between(first, last), F.lit(label))
    return expression


def enrich(df: DataFrame) -> DataFrame:
    """Add duration, speed, unit economics and calendar columns.

    Derived values are rounded at source. Two runs of the same job on the same
    input should produce byte-identical output, and unrounded floating point
    accumulated through a shuffle does not reliably give you that.
    """
    duration_min = (F.col("dropoff_at").cast("long") - F.col("pickup_at").cast("long")) / F.lit(
        60.0
    )

    df = df.withColumn("trip_duration_min", F.round(duration_min, 3))

    duration_hours = F.col("trip_duration_min") / F.lit(60.0)
    df = df.withColumn(
        "avg_speed_mph",
        F.round(_safe_divide(F.col("trip_distance"), duration_hours), 2),
    )
    df = df.withColumn(
        "fare_per_mile",
        F.round(_safe_divide(F.col("fare_amount"), F.col("trip_distance")), 2),
    )
    # Expressed against the fare, not the total: tipping on tolls and
    # surcharges is not a thing anyone does, and including them makes the
    # percentage drift with the route rather than the service.
    df = df.withColumn(
        "tip_pct",
        F.round(_safe_divide(F.col("tip_amount") * F.lit(100.0), F.col("fare_amount")), 2),
    )

    # Bronze already derived this as its partition key, so on the real path
    # this recomputes a column that is already correct. It stays because enrich
    # is also called on frames that never went through `clean` -- the fixtures
    # in the transform tests, and anything reading an older warehouse -- and
    # sharing the expression is what guarantees the two agree.
    df = df.withColumn(PARTITION_COLUMN, pickup_date_expression())
    df = df.withColumn("pickup_hour", F.hour(F.col("pickup_at")))
    df = df.withColumn("pickup_day_of_week", F.dayofweek(F.col("pickup_at")))
    df = df.withColumn("time_of_day", time_of_day_expression(F.col("pickup_hour")))

    df = df.withColumn(
        "is_airport_trip",
        F.coalesce(F.col("rate_code_id").isin(list(AIRPORT_RATE_CODES)), F.lit(False))
        | F.coalesce(F.col("airport_fee") > F.lit(0), F.lit(False)),
    )
    return df
