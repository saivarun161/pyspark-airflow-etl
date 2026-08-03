"""Normalising raw trips: names, whitespace, identity, duplicates."""

from __future__ import annotations

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from tripetl.schema import COLUMN_RENAMES

#: Fields that together identify a trip. The feed ships no trip identifier, so
#: one is derived from the properties that cannot coincide by chance: the same
#: vendor recording the same two timestamps, zones, distance and fare.
NATURAL_KEY: tuple[str, ...] = (
    "vendor_id",
    "pickup_at",
    "dropoff_at",
    "pu_location_id",
    "do_location_id",
    "trip_distance",
    "fare_amount",
    "total_amount",
)

#: Stand-in for a null inside the hashed key. Without it, ``concat_ws`` drops
#: nulls and ``(1, null, 3)`` hashes identically to ``(1, 3, null)`` -- two
#: different trips silently becoming one.
_NULL_SENTINEL = "<null>"


def normalize(df: DataFrame) -> DataFrame:
    """Rename vendor columns to snake_case and tidy the string fields."""
    for source, target in COLUMN_RENAMES.items():
        if source in df.columns and source != target:
            df = df.withColumnRenamed(source, target)

    if "store_and_fwd_flag" in df.columns:
        df = df.withColumn("store_and_fwd_flag", F.upper(F.trim(F.col("store_and_fwd_flag"))))
    return df


def trip_id_expression(columns: tuple[str, ...] = NATURAL_KEY) -> Column:
    """The hash expression used as a trip's surrogate key."""
    parts = [F.coalesce(F.col(name).cast("string"), F.lit(_NULL_SENTINEL)) for name in columns]
    return F.sha2(F.concat_ws("|", *parts), 256)


def add_trip_id(df: DataFrame) -> DataFrame:
    """Attach a stable surrogate key derived from the natural key.

    Stable across runs and across re-ingestion of the same month, which is what
    makes the deduplication below idempotent: reprocessing an extract cannot
    produce a second copy of a trip already in the warehouse.
    """
    return df.withColumn("trip_id", trip_id_expression())


def deduplicate(df: DataFrame) -> DataFrame:
    """Collapse repeated trips to one row each.

    The public extracts genuinely contain duplicates, usually where a month
    boundary was re-published. Rows sharing a ``trip_id`` are identical in
    every field that defines a trip, so which copy survives does not matter.
    """
    return df.dropDuplicates(["trip_id"])


def clean(df: DataFrame) -> DataFrame:
    """Full bronze-shaping: normalize, then key."""
    return add_trip_id(normalize(df))
