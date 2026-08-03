"""PySpark ETL for NYC taxi trips with data-quality gates between layers.

The shape of the thing:

    raw -> [bronze gate] -> bronze -> [silver gate] -> silver -> [gold gate] -> gold
                  |                         |
                  +--> quarantine <---------+

A gate evaluates a declarative :class:`~tripetl.quality.rules.RuleSet` against
the data, writes the report to disk, and raises if a blocking rule falls below
its threshold. Rows that fail a quarantine-eligible rule are written out with
the names of the rules they broke, rather than dropped.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
