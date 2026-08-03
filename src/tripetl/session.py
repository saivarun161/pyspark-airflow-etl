"""SparkSession construction, tuned for a laptop and for deterministic tests."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from pyspark.sql import SparkSession

#: Defaults applied to every session this package builds.
#:
#: ``shuffle.partitions`` is the important one: Spark's default of 200 turns a
#: 10k-row local job into 200 near-empty tasks, and the scheduling overhead
#: dominates the actual work.
BASE_CONF: Mapping[str, str] = {
    "spark.sql.shuffle.partitions": "8",
    "spark.sql.adaptive.enabled": "true",
    # Fixing the session time zone keeps timestamp arithmetic identical on a
    # laptop in any locale and on a CI runner in UTC. Without it, tests that
    # assert on derived date columns pass locally and fail in CI.
    "spark.sql.session.timeZone": "UTC",
    # Fail loudly on a bad cast instead of silently writing NULL.
    "spark.sql.ansi.enabled": "false",
    "spark.sql.legacy.timeParserPolicy": "CORRECTED",
}


def build_session(
    app_name: str = "tripetl",
    *,
    master: str = "local[*]",
    shuffle_partitions: int | None = None,
    ui_enabled: bool = False,
    extra_conf: Mapping[str, str] | None = None,
) -> SparkSession:
    """Build (or attach to) a SparkSession.

    The interpreter pinning below is not optional. Spark launches its Python
    workers by resolving ``python3`` on ``PATH``, which is frequently *not* the
    interpreter running the driver -- inside a virtualenv it is usually the
    system Python. The workers then fail to unpickle anything, with an error
    that points at the wrong place entirely. Pinning both ends to
    ``sys.executable`` makes the driver and its workers the same Python.
    """
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    builder = SparkSession.builder.appName(app_name).master(master)
    for key, value in BASE_CONF.items():
        builder = builder.config(key, value)
    builder = builder.config("spark.ui.enabled", "true" if ui_enabled else "false")
    if shuffle_partitions is not None:
        builder = builder.config("spark.sql.shuffle.partitions", str(shuffle_partitions))
    for key, value in (extra_conf or {}).items():
        builder = builder.config(key, value)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


@contextmanager
def session_scope(spark: SparkSession | None = None, **kwargs: object) -> Iterator[SparkSession]:
    """Yield a session, creating a short-lived one only if none is available.

    Stage functions are called two ways: by the CLI, which runs every stage in
    one long-lived session, and by Airflow, where each task is its own process
    and therefore its own Spark application. A stage should not have to know
    which context it is in -- it asks for a scope and only ever stops the
    session it created itself.

    That last part matters more than it looks. ``getOrCreate`` returns any
    session already live in the JVM, so a scope that stopped whatever it was
    handed would tear down its caller's session and take every later job in the
    process down with it.
    """
    if spark is not None:
        yield spark
        return

    borrowed = SparkSession.getActiveSession()
    if borrowed is not None:
        yield borrowed
        return

    owned = build_session(**kwargs)  # type: ignore[arg-type]
    try:
        yield owned
    finally:
        owned.stop()
