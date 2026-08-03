"""Explicit schemas for every layer of the warehouse.

Nothing in this pipeline infers a schema. Inference costs an extra pass over
the data, and worse, it lets an upstream change alter column types silently --
a month where every ``passenger_count`` happens to be null arrives as
``StringType`` and the downstream arithmetic starts producing nulls instead of
an error. Declaring the schema turns that into a loud failure at read time.
"""

from __future__ import annotations

from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

#: The public NYC TLC yellow-taxi trip record layout, as published in the
#: monthly parquet extracts. Column names are theirs, including the casing.
RAW_TRIP_SCHEMA = StructType(
    [
        StructField("VendorID", IntegerType(), nullable=True),
        StructField("tpep_pickup_datetime", TimestampType(), nullable=True),
        StructField("tpep_dropoff_datetime", TimestampType(), nullable=True),
        StructField("passenger_count", LongType(), nullable=True),
        StructField("trip_distance", DoubleType(), nullable=True),
        StructField("RatecodeID", LongType(), nullable=True),
        StructField("store_and_fwd_flag", StringType(), nullable=True),
        StructField("PULocationID", IntegerType(), nullable=True),
        StructField("DOLocationID", IntegerType(), nullable=True),
        StructField("payment_type", LongType(), nullable=True),
        StructField("fare_amount", DoubleType(), nullable=True),
        StructField("extra", DoubleType(), nullable=True),
        StructField("mta_tax", DoubleType(), nullable=True),
        StructField("tip_amount", DoubleType(), nullable=True),
        StructField("tolls_amount", DoubleType(), nullable=True),
        StructField("improvement_surcharge", DoubleType(), nullable=True),
        StructField("total_amount", DoubleType(), nullable=True),
        StructField("congestion_surcharge", DoubleType(), nullable=True),
        StructField("airport_fee", DoubleType(), nullable=True),
    ]
)

#: Vendor-supplied names mapped to the snake_case names used from bronze on.
#: Applied once, in :func:`tripetl.transforms.clean.normalize`, so the rest of
#: the codebase never has to remember whether it is ``PULocationID`` or
#: ``pu_location_id``.
COLUMN_RENAMES: dict[str, str] = {
    "VendorID": "vendor_id",
    "tpep_pickup_datetime": "pickup_at",
    "tpep_dropoff_datetime": "dropoff_at",
    "RatecodeID": "rate_code_id",
    "PULocationID": "pu_location_id",
    "DOLocationID": "do_location_id",
    "airport_fee": "airport_fee",
}

#: Columns added by :mod:`tripetl.transforms.enrich`, in the order they appear.
ENRICHED_COLUMNS: tuple[str, ...] = (
    "trip_duration_min",
    "avg_speed_mph",
    "fare_per_mile",
    "tip_pct",
    "pickup_date",
    "pickup_hour",
    "pickup_day_of_week",
    "time_of_day",
    "is_airport_trip",
)

#: The gold table. Declared so a schema change to the published mart is a
#: deliberate edit here rather than a side effect of an aggregation tweak.
GOLD_SCHEMA = StructType(
    [
        StructField("pickup_date", DateType(), nullable=True),
        StructField("pu_location_id", IntegerType(), nullable=True),
        StructField("trips", LongType(), nullable=False),
        StructField("total_revenue", DoubleType(), nullable=True),
        StructField("avg_trip_distance", DoubleType(), nullable=True),
        StructField("avg_duration_min", DoubleType(), nullable=True),
        StructField("median_duration_min", DoubleType(), nullable=True),
        StructField("avg_tip_pct", DoubleType(), nullable=True),
        StructField("airport_trip_share", DoubleType(), nullable=True),
    ]
)

#: NYC TLC payment_type codes. 1 credit card, 2 cash, 3 no charge, 4 dispute,
#: 5 unknown, 6 voided trip.
VALID_PAYMENT_TYPES: tuple[int, ...] = (1, 2, 3, 4, 5, 6)

#: RatecodeID codes. 1 standard, 2 JFK, 3 Newark, 4 Nassau/Westchester,
#: 5 negotiated fare, 6 group ride, 99 null/unknown in some vintages.
VALID_RATE_CODES: tuple[int, ...] = (1, 2, 3, 4, 5, 6)

#: Taxi zone ids run 1..265 in the published lookup table.
MAX_LOCATION_ID = 265

#: Rate codes that denote an airport trip (JFK and Newark respectively).
AIRPORT_RATE_CODES: tuple[int, ...] = (2, 3)
