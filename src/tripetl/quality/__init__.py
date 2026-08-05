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
from tripetl.quality.history import (
    HistoryEntry,
    ReportHistory,
    RuleTrend,
    TrendReport,
    compare_reports,
    compare_runs,
    new_run_id,
)
from tripetl.quality.report import QualityReport, RuleResult
from tripetl.quality.rules import DatasetRule, Rule, RuleSet, Severity
from tripetl.quality.rulesets import bronze_ruleset, gold_ruleset, silver_ruleset

__all__ = [
    "FAILURE_COLUMN",
    "ColumnDrift",
    "DatasetRule",
    "HistoryEntry",
    "QualityGateFailed",
    "QualityReport",
    "ReportHistory",
    "Rule",
    "RuleResult",
    "RuleSet",
    "RuleTrend",
    "SchemaDiff",
    "SchemaDriftError",
    "Severity",
    "TrendReport",
    "bronze_ruleset",
    "compare_reports",
    "compare_runs",
    "compare_schemas",
    "enforce",
    "enforce_schema",
    "evaluate",
    "gold_ruleset",
    "new_run_id",
    "silver_ruleset",
    "split",
]
