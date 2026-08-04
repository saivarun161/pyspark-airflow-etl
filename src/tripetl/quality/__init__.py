"""Declarative data quality: rules, a single-pass engine, and the gate."""

from __future__ import annotations

from tripetl.quality.drift import (
    ColumnDrift,
    SchemaDiff,
    SchemaDriftError,
    compare_schemas,
    enforce_schema,
)
from tripetl.quality.engine import FAILURE_COLUMN, evaluate, split
from tripetl.quality.gate import QualityGateFailed, enforce
from tripetl.quality.report import QualityReport, RuleResult
from tripetl.quality.rules import DatasetRule, Rule, RuleSet, Severity
from tripetl.quality.rulesets import bronze_ruleset, gold_ruleset, silver_ruleset

__all__ = [
    "FAILURE_COLUMN",
    "ColumnDrift",
    "DatasetRule",
    "QualityGateFailed",
    "QualityReport",
    "Rule",
    "RuleResult",
    "RuleSet",
    "SchemaDiff",
    "SchemaDriftError",
    "Severity",
    "bronze_ruleset",
    "compare_schemas",
    "enforce",
    "enforce_schema",
    "evaluate",
    "gold_ruleset",
    "silver_ruleset",
    "split",
]
