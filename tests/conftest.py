"""Shared fixtures.

One SparkSession is created for the whole test session. Starting a JVM costs
several seconds; doing it per test would make the suite slow enough that people
stop running it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pytest
from pyspark.sql import DataFrame, SparkSession

from tripetl.session import build_session


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    session = build_session(
        "tripetl-tests",
        master="local[2]",
        # Two partitions keeps shuffles from dominating the runtime on the tiny
        # fixtures used here.
        shuffle_partitions=2,
    )
    yield session
    session.stop()


@pytest.fixture
def make_df(spark: SparkSession):
    """Build a DataFrame from rows and a DDL schema string."""

    def _make(rows: Sequence[tuple], schema: str) -> DataFrame:
        return spark.createDataFrame(list(rows), schema)

    return _make
