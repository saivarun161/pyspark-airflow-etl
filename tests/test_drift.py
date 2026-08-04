"""Schema drift detection: names, types, and how each grade is graded.

These are pure-Python tests over :class:`StructType` -- no SparkSession, no
warehouse. The detector is a comparison of two schema objects, and keeping the
tests at that level is what makes them fast enough to cover every kind
exhaustively.
"""

from __future__ import annotations

import json

import pytest
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from tripetl.quality.drift import (
    BLOCKING_KINDS,
    MISMATCHED,
    MISSING,
    RECASED,
    UNEXPECTED,
    WIDENED,
    ColumnDrift,
    SchemaDiff,
    SchemaDriftError,
    compare_schemas,
    enforce_schema,
)

DECLARED = StructType(
    [
        StructField("VendorID", IntegerType()),
        StructField("pickup_at", TimestampType()),
        StructField("passenger_count", LongType()),
        StructField("trip_distance", DoubleType()),
        StructField("payment_type", LongType()),
    ]
)


def _without(schema: StructType, name: str) -> StructType:
    return StructType([field for field in schema.fields if field.name != name])


def _with(schema: StructType, field: StructField) -> StructType:
    return StructType([*schema.fields, field])


def _retyped(schema: StructType, name: str, new_type) -> StructType:
    return StructType(
        [
            StructField(field.name, new_type) if field.name == name else field
            for field in schema.fields
        ]
    )


def _renamed(schema: StructType, old: str, new: str) -> StructType:
    return StructType(
        [
            StructField(new, field.dataType) if field.name == old else field
            for field in schema.fields
        ]
    )


# -- the clean case ---------------------------------------------------------


def test_an_identical_schema_conforms():
    diff = compare_schemas(DECLARED, DECLARED)
    assert diff.conforms
    assert diff.columns == ()
    assert diff.blocking == ()
    assert diff.advisory == ()


def test_column_order_does_not_matter():
    reordered = StructType(list(reversed(DECLARED.fields)))
    assert compare_schemas(reordered, DECLARED).conforms


# -- missing and unexpected -------------------------------------------------


def test_a_missing_column_is_blocking():
    diff = compare_schemas(_without(DECLARED, "trip_distance"), DECLARED)
    assert not diff.conforms
    (drift,) = diff.of_kind(MISSING)
    assert drift.column == "trip_distance"
    assert drift.is_blocking
    assert drift.expected == "double"
    assert drift.actual is None


def test_an_unexpected_column_is_only_advisory():
    stored = _with(DECLARED, StructField("cbd_congestion_fee", DoubleType()))
    diff = compare_schemas(stored, DECLARED)
    assert diff.conforms
    (drift,) = diff.of_kind(UNEXPECTED)
    assert drift.column == "cbd_congestion_fee"
    assert not drift.is_blocking
    assert drift.actual == "double"


def test_a_gained_column_and_a_lost_column_are_reported_separately():
    stored = _with(_without(DECLARED, "payment_type"), StructField("new_col", StringType()))
    diff = compare_schemas(stored, DECLARED)
    assert {d.kind for d in diff.columns} == {MISSING, UNEXPECTED}
    assert len(diff.blocking) == 1
    assert len(diff.advisory) == 1


# -- types ------------------------------------------------------------------


def test_a_safe_upcast_widens_and_does_not_block():
    diff = compare_schemas(_retyped(DECLARED, "trip_distance", IntegerType()), DECLARED)
    (drift,) = diff.of_kind(WIDENED)
    assert drift.column == "trip_distance"
    assert drift.expected == "double"
    assert drift.actual == "int"
    assert not drift.is_blocking
    assert diff.conforms


def test_a_narrowing_type_change_is_mismatched_and_blocks():
    diff = compare_schemas(_retyped(DECLARED, "trip_distance", StringType()), DECLARED)
    (drift,) = diff.of_kind(MISMATCHED)
    assert drift.is_blocking
    assert drift.actual == "string"
    assert not diff.conforms


def test_date_widens_to_the_declared_timestamp():
    from pyspark.sql.types import DateType

    diff = compare_schemas(_retyped(DECLARED, "pickup_at", DateType()), DECLARED)
    (drift,) = diff.of_kind(WIDENED)
    assert not drift.is_blocking


def test_types_can_be_left_unchecked():
    stored = _retyped(DECLARED, "trip_distance", StringType())
    diff = compare_schemas(stored, DECLARED, check_types=False)
    assert diff.conforms
    assert diff.columns == ()
    assert diff.types_checked is False


# -- casing -----------------------------------------------------------------


def test_a_recased_name_is_matched_but_reported():
    diff = compare_schemas(_renamed(DECLARED, "VendorID", "vendorid"), DECLARED)
    (drift,) = diff.of_kind(RECASED)
    assert drift.expected == "VendorID"
    assert drift.actual == "vendorid"
    assert not drift.is_blocking
    # matched, so no MISSING and no UNEXPECTED were emitted for it
    assert diff.of_kind(MISSING) == ()
    assert diff.of_kind(UNEXPECTED) == ()
    assert diff.conforms


def test_case_sensitive_mode_treats_a_recase_as_missing_and_unexpected():
    diff = compare_schemas(
        _renamed(DECLARED, "VendorID", "vendorid"), DECLARED, case_sensitive=True
    )
    assert {d.kind for d in diff.columns} == {MISSING, UNEXPECTED}
    assert not diff.conforms  # the declared VendorID is now missing


def test_a_recased_column_with_a_widened_type_reports_both():
    stored = _retyped(
        _renamed(DECLARED, "trip_distance", "TRIP_DISTANCE"), "TRIP_DISTANCE", IntegerType()
    )
    diff = compare_schemas(stored, DECLARED)
    kinds = {d.kind for d in diff.columns}
    assert kinds == {RECASED, WIDENED}
    assert diff.conforms


# -- the aggregate diff -----------------------------------------------------


def test_blocking_kinds_are_exactly_missing_and_mismatched():
    assert set(BLOCKING_KINDS) == {MISSING, MISMATCHED}


def test_the_diff_counts_columns_on_both_sides():
    stored = _with(DECLARED, StructField("extra", StringType()))
    diff = compare_schemas(stored, DECLARED)
    assert diff.declared_columns == 5
    assert diff.stored_columns == 6


def test_the_diff_round_trips_through_json():
    stored = _retyped(_without(DECLARED, "payment_type"), "trip_distance", StringType())
    diff = compare_schemas(stored, DECLARED, dataset="s3://feed")
    payload = json.loads(diff.to_json())
    assert payload["dataset"] == "s3://feed"
    assert payload["conforms"] is False
    assert payload["blocking_count"] == 2
    assert len(payload["columns"]) == 2


def test_text_names_the_dataset_and_the_verdict():
    diff = compare_schemas(_without(DECLARED, "trip_distance"), DECLARED, dataset="feed-A")
    text = diff.to_text()
    assert "feed-A" in text
    assert "DRIFTED" in text
    assert "trip_distance" in text


def test_text_reports_a_clean_schema_plainly():
    text = compare_schemas(DECLARED, DECLARED, dataset="feed-A").to_text()
    assert "CONFORMS" in text
    assert "no drift" in text


# -- enforcement ------------------------------------------------------------


def test_enforce_returns_the_diff_when_it_conforms():
    diff = compare_schemas(DECLARED, DECLARED)
    assert enforce_schema(diff) is diff


def test_enforce_raises_on_blocking_drift():
    diff = compare_schemas(_without(DECLARED, "trip_distance"), DECLARED, dataset="feed-A")
    with pytest.raises(SchemaDriftError) as excinfo:
        enforce_schema(diff)
    assert excinfo.value.diff is diff
    assert "feed-A" in str(excinfo.value)
    assert "trip_distance" in str(excinfo.value)


def test_advisory_only_drift_does_not_raise():
    stored = _with(DECLARED, StructField("cbd_congestion_fee", DoubleType()))
    diff = compare_schemas(stored, DECLARED)
    assert enforce_schema(diff) is diff


def test_non_strict_enforcement_reports_but_does_not_raise():
    diff = compare_schemas(_without(DECLARED, "trip_distance"), DECLARED)
    assert enforce_schema(diff, strict=False) is diff


def test_a_manual_column_drift_summarises_each_kind():
    assert "absent" in ColumnDrift("c", MISSING, expected="double").summary()
    assert "never declared" in ColumnDrift("c", UNEXPECTED, actual="double").summary()
    assert "BLOCK" in ColumnDrift("c", MISSING, expected="double").summary()
    assert "WARN" in ColumnDrift("c", WIDENED, expected="double", actual="int").summary()


def test_an_empty_diff_conforms_and_prints_cleanly():
    diff = SchemaDiff(dataset="x", declared_columns=0, stored_columns=0)
    assert diff.conforms
    assert "no drift" in diff.to_text()
