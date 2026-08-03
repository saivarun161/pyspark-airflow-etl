"""The rule sets guarding each stage boundary.

Thresholds below 1.0 are a deliberate admission: real vendor feeds always
carry a little junk, and a pipeline that halts on a single malformed row out of
twenty million is a pipeline someone will disable. The thresholds encode how
much junk is normal; crossing one means something changed upstream.
"""

from __future__ import annotations

from pyspark.sql import functions as F

from tripetl.quality import rules as r
from tripetl.quality.rules import RuleSet, Severity
from tripetl.schema import (
    MAX_LOCATION_ID,
    VALID_PAYMENT_TYPES,
    VALID_RATE_CODES,
)

#: Longest trip we will believe, in minutes. A 12-hour taxi ride is a broken
#: meter, not a fare.
MAX_TRIP_MINUTES = 12 * 60

#: Longest trip we will believe, in miles.
MAX_TRIP_MILES = 200.0

#: Fastest average speed we will believe for a street journey, in mph.
MAX_AVG_SPEED_MPH = 90.0

#: Largest single fare we will believe, in dollars.
MAX_FARE = 5_000.0


def bronze_ruleset(*, min_rows: int = 1) -> RuleSet:
    """Structural checks on freshly ingested trips.

    These are the expectations that must hold for a row to be *interpretable*
    at all: the timestamps exist and run forwards, the codes are codes we know,
    the money is not negative.
    """
    return RuleSet(
        name="bronze",
        row_rules=(
            r.not_null("pickup_at"),
            r.not_null("dropoff_at"),
            r.ordered("pickup_at", "dropoff_at", strict=True, threshold=0.98),
            r.non_negative("fare_amount", threshold=0.98),
            r.in_range("fare_amount", 0.0, MAX_FARE, threshold=0.98),
            r.in_range("trip_distance", 0.0, MAX_TRIP_MILES, threshold=0.98),
            r.in_range("pu_location_id", 1, MAX_LOCATION_ID, threshold=0.98),
            r.in_range("do_location_id", 1, MAX_LOCATION_ID, threshold=0.98),
            r.accepted_values("payment_type", VALID_PAYMENT_TYPES, threshold=0.98),
            r.accepted_values(
                "rate_code_id",
                VALID_RATE_CODES,
                allow_null=True,
                severity=Severity.WARN,
                threshold=0.98,
            ),
            # Passenger count is chronically unreliable in the public feed --
            # worth watching, not worth blocking on, and not worth discarding
            # an otherwise-good fare over.
            r.not_null(
                "passenger_count",
                severity=Severity.WARN,
                threshold=0.95,
                quarantine=False,
            ),
            r.non_negative("tip_amount", allow_null=True, threshold=0.99),
            r.non_negative("total_amount", threshold=0.98),
        ),
        dataset_rules=(r.row_count_at_least(min_rows),),
    )


def silver_ruleset(*, min_rows: int = 1) -> RuleSet:
    """Checks on cleaned, deduplicated, enriched trips.

    Bronze asked whether a row could be read. Silver asks whether the values we
    *derived* from it are physically plausible -- a trip that covers nine miles
    in forty seconds parsed fine and is still wrong.
    """
    return RuleSet(
        name="silver",
        row_rules=(
            r.not_null("trip_id"),
            r.in_range("trip_duration_min", 0.0, MAX_TRIP_MINUTES, threshold=0.99),
            r.satisfies(
                name="positive_duration",
                predicate=F.coalesce(F.col("trip_duration_min") > F.lit(0), F.lit(False)),
                description="trip_duration_min must be greater than zero",
                threshold=0.99,
                columns=("trip_duration_min",),
            ),
            r.in_range(
                "avg_speed_mph",
                0.0,
                MAX_AVG_SPEED_MPH,
                allow_null=True,
                threshold=0.99,
            ),
            # A tip above 100% of the fare happens; above 1000% is a data
            # entry error. Null is expected for cash fares, where the tip is
            # never recorded.
            r.in_range(
                "tip_pct",
                0.0,
                1_000.0,
                allow_null=True,
                severity=Severity.WARN,
                threshold=0.99,
            ),
            r.not_null("pickup_date"),
            r.in_range("pickup_hour", 0, 23),
        ),
        dataset_rules=(
            # Deduplication happens immediately upstream of this gate, so a
            # duplicate here means the dedupe key is wrong -- a defect in our
            # code, not in the vendor's feed. Hence no tolerance at all.
            r.unique("trip_id"),
            r.row_count_at_least(min_rows),
        ),
    )


def gold_ruleset(*, min_rows: int = 1) -> RuleSet:
    """Checks on the published mart.

    Cheap, and they catch the aggregation bugs that are otherwise invisible:
    a bad join blowing up trip counts, a share that exceeds 1 because the
    numerator and denominator were grouped differently.
    """
    return RuleSet(
        name="gold",
        row_rules=(
            r.not_null("pickup_date"),
            r.not_null("pu_location_id"),
            r.satisfies(
                name="positive_trips",
                predicate=F.coalesce(F.col("trips") > F.lit(0), F.lit(False)),
                description="every published group must contain at least one trip",
                columns=("trips",),
            ),
            r.non_negative("total_revenue", allow_null=True, threshold=0.99),
            r.in_range("airport_trip_share", 0.0, 1.0, allow_null=True),
            r.in_range(
                "median_duration_min",
                0.0,
                MAX_TRIP_MINUTES,
                allow_null=True,
                threshold=0.99,
            ),
        ),
        dataset_rules=(r.row_count_at_least(min_rows),),
    )
