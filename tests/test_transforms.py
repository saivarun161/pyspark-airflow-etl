"""Transformations: naming, keying, deduplication, derived values, aggregates."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from tripetl.schema import GOLD_SCHEMA
from tripetl.transforms.aggregate import daily_zone_metrics
from tripetl.transforms.clean import add_trip_id, deduplicate, normalize
from tripetl.transforms.enrich import enrich

TRIP_SCHEMA = (
    "pickup_at timestamp, dropoff_at timestamp, trip_distance double, "
    "fare_amount double, tip_amount double, rate_code_id bigint, airport_fee double"
)

KEY_SCHEMA = (
    "vendor_id int, pickup_at timestamp, dropoff_at timestamp, "
    "pu_location_id int, do_location_id int, trip_distance double, "
    "fare_amount double, total_amount double"
)

# Timezone-aware: a naive datetime is converted using the driver's local zone,
# which would make every hour-of-day assertion below machine-dependent.
PICKUP = datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC)
DROPOFF = datetime(2026, 6, 1, 0, 30, 0, tzinfo=UTC)


# -- clean ------------------------------------------------------------------


def test_normalize_renames_vendor_columns(make_df):
    df = make_df(
        [(1, PICKUP, DROPOFF, 100, 200)],
        "VendorID int, tpep_pickup_datetime timestamp, "
        "tpep_dropoff_datetime timestamp, PULocationID int, DOLocationID int",
    )
    renamed = normalize(df)

    assert set(renamed.columns) == {
        "vendor_id",
        "pickup_at",
        "dropoff_at",
        "pu_location_id",
        "do_location_id",
    }


def test_normalize_tidies_the_store_and_forward_flag(make_df):
    df = make_df([("  n ",), ("Y",)], "store_and_fwd_flag string")
    values = [row[0] for row in normalize(df).collect()]
    assert values == ["N", "Y"]


def test_normalize_tolerates_columns_that_are_already_renamed(make_df):
    df = make_df([(1,)], "vendor_id int")
    assert normalize(df).columns == ["vendor_id"]


def test_trip_id_is_stable_for_identical_trips(make_df):
    rows = [(1, PICKUP, DROPOFF, 10, 20, 5.0, 20.0, 25.0)] * 2
    ids = {row["trip_id"] for row in add_trip_id(make_df(rows, KEY_SCHEMA)).collect()}
    assert len(ids) == 1


def test_trip_id_differs_when_any_key_field_differs(make_df):
    rows = [
        (1, PICKUP, DROPOFF, 10, 20, 5.0, 20.0, 25.0),
        (1, PICKUP, DROPOFF, 10, 20, 5.0, 20.0, 26.0),
    ]
    ids = {row["trip_id"] for row in add_trip_id(make_df(rows, KEY_SCHEMA)).collect()}
    assert len(ids) == 2


def test_nulls_in_the_key_do_not_collide(make_df):
    """Two different trips must not hash alike just because a field is null.

    ``concat_ws`` drops nulls, so without the sentinel these two rows -- which
    differ in *which* zone is missing -- would produce the same key.
    """
    rows = [
        (1, PICKUP, DROPOFF, None, 20, 5.0, 20.0, 25.0),
        (1, PICKUP, DROPOFF, 20, None, 5.0, 20.0, 25.0),
    ]
    ids = {row["trip_id"] for row in add_trip_id(make_df(rows, KEY_SCHEMA)).collect()}
    assert len(ids) == 2


def test_deduplicate_collapses_repeated_trips(make_df):
    rows = [(1, PICKUP, DROPOFF, 10, 20, 5.0, 20.0, 25.0)] * 3 + [
        (2, PICKUP, DROPOFF, 10, 20, 5.0, 20.0, 25.0)
    ]
    deduped = deduplicate(add_trip_id(make_df(rows, KEY_SCHEMA)))
    assert deduped.count() == 2


def test_deduplicate_is_idempotent(make_df):
    rows = [(1, PICKUP, DROPOFF, 10, 20, 5.0, 20.0, 25.0)] * 3
    once = deduplicate(add_trip_id(make_df(rows, KEY_SCHEMA)))
    assert deduplicate(once).count() == once.count() == 1


# -- enrich -----------------------------------------------------------------


@pytest.fixture
def enriched(make_df):
    rows = [
        # 30 minutes, 5 miles, $20 fare, $4 tip -> 10 mph, $4/mile, 20% tip
        (PICKUP, DROPOFF, 5.0, 20.0, 4.0, 1, 0.0),
        # zero distance and zero fare: both unit economics undefined
        (PICKUP, DROPOFF, 0.0, 0.0, 0.0, 1, 0.0),
        # zero duration: speed undefined
        (PICKUP, PICKUP, 3.0, 10.0, 0.0, 1, 0.0),
        # airport trip by rate code, at 18:00
        (
            datetime(2026, 6, 2, 18, 0, 0, tzinfo=UTC),
            datetime(2026, 6, 2, 18, 45, 0, tzinfo=UTC),
            12.0,
            60.0,
            0.0,
            2,
            0.0,
        ),
        # airport trip by fee alone, at 08:00
        (
            datetime(2026, 6, 3, 8, 0, 0, tzinfo=UTC),
            datetime(2026, 6, 3, 8, 20, 0, tzinfo=UTC),
            4.0,
            18.0,
            0.0,
            1,
            1.75,
        ),
    ]
    return enrich(make_df(rows, TRIP_SCHEMA)).collect()


def test_duration_speed_and_unit_economics(enriched):
    row = enriched[0]
    assert row["trip_duration_min"] == 30.0
    assert row["avg_speed_mph"] == 10.0
    assert row["fare_per_mile"] == 4.0
    assert row["tip_pct"] == 20.0


def test_division_by_zero_yields_null_not_infinity(enriched):
    """Infinity would propagate through avg() and poison a whole gold group."""
    zero_distance = enriched[1]
    assert zero_distance["fare_per_mile"] is None
    assert zero_distance["tip_pct"] is None

    zero_duration = enriched[2]
    assert zero_duration["trip_duration_min"] == 0.0
    assert zero_duration["avg_speed_mph"] is None


def test_calendar_columns_come_from_the_pickup(enriched):
    row = enriched[0]
    assert row["pickup_date"] == date(2026, 6, 1)
    assert row["pickup_hour"] == 0
    assert row["time_of_day"] == "overnight"
    # 2026-06-01 is a Monday; Spark's dayofweek is 1-based from Sunday.
    assert row["pickup_day_of_week"] == 2


def test_time_of_day_buckets(enriched):
    assert enriched[3]["time_of_day"] == "evening"
    assert enriched[4]["time_of_day"] == "morning"


def test_airport_trips_are_flagged_by_rate_code_or_by_fee(enriched):
    assert enriched[0]["is_airport_trip"] is False
    assert enriched[3]["is_airport_trip"] is True
    assert enriched[4]["is_airport_trip"] is True


def test_airport_flag_is_never_null_when_inputs_are_missing(make_df):
    df = make_df([(PICKUP, DROPOFF, 1.0, 5.0, 0.0, None, None)], TRIP_SCHEMA)
    assert enrich(df).collect()[0]["is_airport_trip"] is False


# -- aggregate --------------------------------------------------------------

SILVER_SCHEMA = (
    "pickup_date date, pu_location_id int, total_amount double, "
    "trip_distance double, trip_duration_min double, tip_pct double, "
    "is_airport_trip boolean"
)


@pytest.fixture
def gold(make_df):
    rows = [
        (date(2026, 6, 1), 10, 25.0, 5.0, 30.0, 20.0, False),
        (date(2026, 6, 1), 10, 35.0, 7.0, 10.0, 10.0, True),
        (date(2026, 6, 1), 20, 15.0, 2.0, 20.0, None, False),
        (date(2026, 6, 2), 10, 45.0, 9.0, 40.0, 30.0, False),
    ]
    return daily_zone_metrics(make_df(rows, SILVER_SCHEMA)).collect()


def test_gold_grain_is_one_row_per_zone_per_day(gold):
    assert len(gold) == 3
    assert [(row["pickup_date"], row["pu_location_id"]) for row in gold] == [
        (date(2026, 6, 1), 10),
        (date(2026, 6, 1), 20),
        (date(2026, 6, 2), 10),
    ]


def test_gold_metrics_are_computed_per_group(gold):
    first = gold[0]
    assert first["trips"] == 2
    assert first["total_revenue"] == 60.0
    assert first["avg_trip_distance"] == 6.0
    assert first["avg_duration_min"] == 20.0
    assert first["avg_tip_pct"] == 15.0
    assert first["airport_trip_share"] == 0.5


def test_null_derived_values_are_excluded_rather_than_counted_as_zero(gold):
    """The zone whose only trip has no tip percentage reports null, not 0."""
    zone_20 = gold[1]
    assert zone_20["trips"] == 1
    assert zone_20["avg_tip_pct"] is None


def test_gold_matches_the_declared_schema(gold, make_df):
    rows = [(date(2026, 6, 1), 10, 25.0, 5.0, 30.0, 20.0, False)]
    result = daily_zone_metrics(make_df(rows, SILVER_SCHEMA))

    assert result.columns == list(GOLD_SCHEMA.fieldNames())
    actual = {field.name: field.dataType for field in result.schema.fields}
    expected = {field.name: field.dataType for field in GOLD_SCHEMA.fields}
    assert actual == expected
