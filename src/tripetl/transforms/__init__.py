"""DataFrame transformations, split by the layer they produce.

Every function here takes a DataFrame and returns a DataFrame, with no I/O and
no SparkSession of its own. That is what makes them testable against a
five-row fixture instead of a warehouse.
"""

from __future__ import annotations

from tripetl.transforms.aggregate import daily_zone_metrics
from tripetl.transforms.clean import (
    add_pickup_date,
    add_trip_id,
    clean,
    deduplicate,
    normalize,
    pickup_date_expression,
)
from tripetl.transforms.enrich import enrich

__all__ = [
    "add_pickup_date",
    "add_trip_id",
    "clean",
    "daily_zone_metrics",
    "deduplicate",
    "enrich",
    "normalize",
    "pickup_date_expression",
]
