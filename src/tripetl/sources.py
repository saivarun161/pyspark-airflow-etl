"""Reading raw trips, and generating a realistic sample when there are none.

The sample generator exists so the whole pipeline -- including the gates and
the quarantine path -- can be demonstrated and tested without a network round
trip, an API key, or a 50 MB download in CI. It models the same schema the
public NYC TLC extracts use, and injects the specific defects those extracts
actually contain.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

from pyspark.sql import DataFrame, SparkSession

from tripetl.schema import MAX_LOCATION_ID, RAW_TRIP_SCHEMA

#: Defects the generator knows how to inject, one per bronze rule worth
#: exercising. Named so a test can assert on them individually.
DEFECTS: tuple[str, ...] = (
    "null_passenger_count",
    "negative_fare",
    "reversed_timestamps",
    "impossible_distance",
    "unknown_payment_type",
    "unknown_zone",
)

#: Start of the synthetic week. Timezone-aware on purpose: PySpark converts a
#: *naive* datetime to an instant using the driver's local timezone, not the
#: session timezone, so a generator built on naive datetimes produces different
#: pickup hours on a laptop in New York than on a CI runner in UTC -- and the
#: tests that assert on those hours pass in one place and fail in the other.
SAMPLE_EPOCH = datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC)
SAMPLE_DAYS = 7

_MTA_TAX = 0.5
_IMPROVEMENT_SURCHARGE = 0.3
_CONGESTION_SURCHARGE = 2.5
_AIRPORT_FEE = 1.75


def read_raw(spark: SparkSession, path: str, *, fmt: str = "parquet") -> DataFrame:
    """Read raw trip records with the schema declared rather than inferred.

    Works unchanged against a local path or the public TLC parquet extracts;
    the reader does not care which, so the pipeline is exercised identically in
    a test and against real published data.
    """
    reader = spark.read.schema(RAW_TRIP_SCHEMA)
    if fmt == "csv":
        reader = reader.option("header", "true")
    return reader.format(fmt).load(path)


def generate_sample(
    spark: SparkSession,
    *,
    rows: int = 20_000,
    seed: int = 20260803,
    dirty_rate: float = 0.04,
) -> DataFrame:
    """Build a synthetic raw-trip DataFrame.

    Args:
        rows: Number of clean base rows before duplicates are added.
        seed: Fixes the output. The same seed always yields the same rows.
        dirty_rate: Fraction of rows carrying a defect, spread evenly across
            :data:`DEFECTS`. Duplicate rows are added at half this rate on top.
            At ``0.0`` the output is pristine and passes every gate -- which is
            what makes it useful as a test fixture.
    """
    if not 0.0 <= dirty_rate <= 1.0:
        raise ValueError(f"dirty_rate must be between 0 and 1, got {dirty_rate}")

    rng = random.Random(seed)
    records = [_clean_record(rng) for _ in range(rows)]

    if dirty_rate > 0 and records:
        for index, record in enumerate(records):
            if rng.random() < dirty_rate:
                records[index] = _corrupt(rng, record)

        duplicates = int(rows * dirty_rate / 2)
        records.extend(rng.choice(records) for _ in range(duplicates))

    return spark.createDataFrame(records, RAW_TRIP_SCHEMA)


def _clean_record(rng: random.Random) -> tuple[object, ...]:
    """One plausible, internally consistent trip."""
    pickup = SAMPLE_EPOCH + timedelta(seconds=rng.randrange(SAMPLE_DAYS * 24 * 60 * 60))
    # Trip lengths are heavily right-skewed: mostly short hops, a long tail.
    distance = round(min(rng.expovariate(1 / 2.6) + 0.3, 60.0), 2)
    speed_mph = rng.uniform(7.0, 24.0)
    duration_min = max(1.0, (distance / speed_mph) * 60.0)
    dropoff = pickup + timedelta(minutes=duration_min)

    rate_code = rng.choices([1, 2, 3, 4, 5], weights=[92, 3, 1, 2, 2])[0]
    is_airport = rate_code in (2, 3)
    payment_type = rng.choices([1, 2, 3, 4], weights=[70, 26, 2, 2])[0]

    fare = round(3.0 + 2.6 * distance + 0.45 * duration_min, 2)
    extra = round(rng.choice([0.0, 0.5, 1.0]), 2)
    # Only card payments record a tip; cash tips are invisible to the meter.
    tip = round(fare * rng.uniform(0.0, 0.3), 2) if payment_type == 1 else 0.0
    tolls = round(rng.uniform(6.0, 12.0), 2) if is_airport else 0.0
    airport_fee = _AIRPORT_FEE if is_airport else 0.0
    total = round(
        fare
        + extra
        + _MTA_TAX
        + tip
        + tolls
        + _IMPROVEMENT_SURCHARGE
        + _CONGESTION_SURCHARGE
        + airport_fee,
        2,
    )

    return (
        rng.choice([1, 2]),
        pickup,
        dropoff,
        rng.choices([1, 2, 3, 4, 5, 6], weights=[70, 15, 6, 4, 3, 2])[0],
        distance,
        rate_code,
        "Y" if rng.random() < 0.02 else "N",
        rng.randint(1, MAX_LOCATION_ID),
        rng.randint(1, MAX_LOCATION_ID),
        payment_type,
        fare,
        extra,
        _MTA_TAX,
        tip,
        tolls,
        _IMPROVEMENT_SURCHARGE,
        total,
        _CONGESTION_SURCHARGE,
        airport_fee,
    )


def _corrupt(rng: random.Random, record: tuple[object, ...]) -> tuple[object, ...]:
    """Apply one randomly chosen defect to a record.

    Each defect corresponds to a real failure mode in the published feed:
    meters that report backwards, GPS glitches producing 4000-mile trips,
    payment codes outside the documented set, zone ids from a newer lookup
    table than the one you are joining against.
    """
    fields = list(record)
    defect = rng.choice(DEFECTS)

    if defect == "null_passenger_count":
        fields[3] = None
    elif defect == "negative_fare":
        fields[10] = -abs(float(fields[10]))  # type: ignore[arg-type]
        fields[16] = -abs(float(fields[16]))  # type: ignore[arg-type]
    elif defect == "reversed_timestamps":
        fields[1], fields[2] = fields[2], fields[1]
    elif defect == "impossible_distance":
        fields[4] = round(rng.uniform(500.0, 4_000.0), 2)
    elif defect == "unknown_payment_type":
        fields[9] = 99
    elif defect == "unknown_zone":
        fields[7] = 999
    else:  # pragma: no cover - defensive; DEFECTS and this branch move together
        raise ValueError(f"unhandled defect {defect!r}")

    return tuple(fields)
