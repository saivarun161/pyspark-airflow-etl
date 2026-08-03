"""The gold layer: daily demand and revenue per pickup zone."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from tripetl.schema import GOLD_SCHEMA

#: The grain of the published mart. One row per zone per day.
GRAIN: tuple[str, ...] = ("pickup_date", "pu_location_id")


def daily_zone_metrics(df: DataFrame) -> DataFrame:
    """Aggregate enriched trips to one row per pickup zone per day.

    ``percentile_approx`` rather than an exact percentile: the exact version
    needs a full sort within each group, and at a realistic month's volume the
    difference between an exact and an approximate median duration is far
    smaller than the measurement error in the meter that produced it.
    """
    aggregated = df.groupBy(*GRAIN).agg(
        F.count(F.lit(1)).alias("trips"),
        F.round(F.sum("total_amount"), 2).alias("total_revenue"),
        F.round(F.avg("trip_distance"), 3).alias("avg_trip_distance"),
        F.round(F.avg("trip_duration_min"), 2).alias("avg_duration_min"),
        F.round(F.percentile_approx("trip_duration_min", 0.5), 2).alias("median_duration_min"),
        F.round(F.avg("tip_pct"), 2).alias("avg_tip_pct"),
        F.round(F.avg(F.col("is_airport_trip").cast("double")), 4).alias("airport_trip_share"),
    )
    # Select by the declared schema so the published column order is fixed by
    # `schema.py` and not by the order the aggregations happen to be written.
    return aggregated.select(*GOLD_SCHEMA.fieldNames()).orderBy(*GRAIN)
