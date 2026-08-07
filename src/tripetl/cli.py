"""Command line entry point: ``tripetl``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import date

from tripetl.config import DEFAULT_PARTITION_BY, DEFAULT_WAREHOUSE, PipelineConfig
from tripetl.quality import engine, rulesets
from tripetl.quality.drift import SchemaDriftError
from tripetl.quality.gate import QualityGateFailed
from tripetl.quality.history import DEFAULT_TOLERANCE, ReportHistory, compare_runs
from tripetl.quality.rules import RuleSet
from tripetl.session import session_scope
from tripetl.sources import generate_sample, raw_schema_diff

#: Returned when a data-quality gate blocks the run. Distinct from 1 so a
#: scheduler can tell "the data was bad" apart from "the job crashed" -- they
#: want different people woken up. Schema drift shares it: the source being
#: wrong wakes the same person as the data being wrong.
EXIT_GATE_FAILED = 2

#: Returned when the arguments cannot describe a run at all -- a window that
#: ends before it starts, a defect rate above 1. Distinct from
#: :data:`EXIT_GATE_FAILED` because nobody needs paging over a typo.
EXIT_USAGE = 1

STAGE_RULESETS = {
    "bronze": rulesets.bronze_ruleset,
    "silver": rulesets.silver_ruleset,
    "gold": rulesets.gold_ruleset,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tripetl",
        description=(
            "PySpark ETL for NYC taxi trips, with data-quality gates between every layer."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_sample_options(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--rows", type=int, default=20_000, help="base rows to generate")
        sub.add_argument("--seed", type=int, default=20260803, help="generator seed")
        sub.add_argument(
            "--dirty-rate",
            type=float,
            default=0.04,
            help="fraction of rows carrying an injected defect (0 for pristine)",
        )

    run = subparsers.add_parser("run", help="run bronze -> silver -> gold")
    run.add_argument(
        "--input",
        help="raw trip data; omit to generate a sample and run entirely offline",
    )
    run.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    run.add_argument("--warehouse", default=DEFAULT_WAREHOUSE, help="output root")
    run.add_argument("--master", default="local[*]", help="Spark master URL")
    run.add_argument(
        "--no-gates",
        action="store_true",
        help="report quality failures without stopping the run",
    )
    run.add_argument(
        "--no-quarantine",
        action="store_true",
        help="carry failing rows forward instead of diverting them",
    )
    run.add_argument(
        "--no-schema-check",
        action="store_true",
        help="skip comparing the input's stored layout to the declared schema",
    )
    # A window plus partitioned writes is what makes a backfill safe: the run
    # rebuilds the days it names and replaces exactly those partitions.
    run.add_argument(
        "--since",
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="process only trips picked up on or after this date",
    )
    run.add_argument(
        "--until",
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="process only trips picked up on or before this date",
    )
    run.add_argument(
        "--no-partitioning",
        action="store_true",
        help="write each layer as one flat dataset, so a run replaces all of it",
    )
    add_sample_options(run)

    sample = subparsers.add_parser("sample", help="write a synthetic raw dataset")
    sample.add_argument("--out", required=True, help="destination path")
    sample.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    sample.add_argument("--master", default="local[*]")
    add_sample_options(sample)

    quality = subparsers.add_parser(
        "quality", help="check an existing dataset without writing anything"
    )
    quality.add_argument("--path", required=True, help="dataset to check")
    quality.add_argument("--stage", required=True, choices=sorted(STAGE_RULESETS))
    quality.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    quality.add_argument("--master", default="local[*]")
    quality.add_argument(
        "--markdown", action="store_true", help="emit a markdown table instead of text"
    )

    schema = subparsers.add_parser(
        "schema", help="compare a raw extract's stored layout to the declared schema"
    )
    schema.add_argument("--path", required=True, help="raw extract to inspect")
    schema.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    schema.add_argument("--master", default="local[*]")
    schema.add_argument("--json", action="store_true", help="emit the diff as JSON")

    trend = subparsers.add_parser(
        "trend", help="compare a stage's two most recent runs from the report history"
    )
    trend.add_argument("--warehouse", default=DEFAULT_WAREHOUSE, help="warehouse root")
    trend.add_argument("--stage", required=True, choices=sorted(STAGE_RULESETS))
    trend.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="how far a pass rate may move before it counts as a change",
    )
    trend.add_argument("--json", action="store_true", help="emit the trend as JSON")
    trend.add_argument(
        "--markdown", action="store_true", help="emit a markdown table instead of text"
    )
    # Opt-in rather than the default the gates use. A threshold is a policy
    # someone signed up to; a tolerance is a guess until a few weeks of history
    # have calibrated it, and a check that blocks on a guess gets switched off.
    trend.add_argument(
        "--fail-on-regression",
        action="store_true",
        help=f"exit {EXIT_GATE_FAILED} when a rule regressed since the previous run",
    )

    show = subparsers.add_parser("show", help="preview a dataset")
    show.add_argument("--path", required=True)
    show.add_argument("--limit", type=int, default=20)
    show.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    show.add_argument("--master", default="local[*]")

    return parser


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        warehouse=args.warehouse,
        input_path=args.input,
        input_format=args.format,
        master=args.master,
        enforce_gates=not args.no_gates,
        quarantine_enabled=not args.no_quarantine,
        check_input_schema=not args.no_schema_check,
        partition_by=() if args.no_partitioning else DEFAULT_PARTITION_BY,
        since=args.since,
        until=args.until,
        sample_rows=args.rows,
        sample_seed=args.seed,
        sample_dirty_rate=args.dirty_rate,
    )


def _read(args: argparse.Namespace, session: object) -> object:
    reader = session.read  # type: ignore[attr-defined]
    if args.format == "csv":
        return reader.option("header", "true").option("inferSchema", "true").csv(args.path)
    return reader.parquet(args.path)


def _command_run(args: argparse.Namespace) -> int:
    from tripetl.pipeline import NothingToPublish, run_pipeline

    # A config that cannot be built is a usage error, not a crash. Reaching
    # Spark first would bury the one useful line under a JVM stack trace.
    try:
        config = _config_from_args(args)
    except ValueError as invalid:
        print(f"tripetl run: {invalid}", file=sys.stderr)
        return EXIT_USAGE

    try:
        result = run_pipeline(config)
    except QualityGateFailed as failure:
        print(failure.report.to_text(), file=sys.stderr)
        print(f"\n{failure}", file=sys.stderr)
        return EXIT_GATE_FAILED
    except SchemaDriftError as failure:
        print(failure.diff.to_text(), file=sys.stderr)
        print(f"\n{failure}", file=sys.stderr)
        return EXIT_GATE_FAILED
    # An empty stage is a data problem too -- every row rejected, or a window
    # naming days the input does not contain -- so it wakes the same person.
    except NothingToPublish as failure:
        print(f"{failure}", file=sys.stderr)
        return EXIT_GATE_FAILED

    print(result.to_text())
    for stage in result.stages:
        print()
        print(stage.report.to_text())
    return 0


def _command_sample(args: argparse.Namespace) -> int:
    with session_scope(app_name="tripetl-sample", master=args.master) as session:
        df = generate_sample(session, rows=args.rows, seed=args.seed, dirty_rate=args.dirty_rate)
        writer = df.write.mode("overwrite")
        if args.format == "csv":
            writer.option("header", "true").csv(args.out)
        else:
            writer.parquet(args.out)
        print(f"wrote {df.count():,} rows to {args.out}")
    return 0


def _command_quality(args: argparse.Namespace) -> int:
    with session_scope(app_name="tripetl-quality", master=args.master) as session:
        ruleset: RuleSet = STAGE_RULESETS[args.stage]()
        report = engine.evaluate(_read(args, session), ruleset, stage=args.stage)
        print(report.to_markdown() if args.markdown else report.to_text())
        return 0 if report.passed else EXIT_GATE_FAILED


def _command_schema(args: argparse.Namespace) -> int:
    with session_scope(app_name="tripetl-schema", master=args.master) as session:
        diff = raw_schema_diff(session, args.path, fmt=args.format)
        print(diff.to_json() if args.json else diff.to_text())
        return 0 if diff.conforms else EXIT_GATE_FAILED


def _command_trend(args: argparse.Namespace) -> int:
    """Read the history off disk and diff the last two runs.

    No SparkSession: the history is JSON written by earlier runs, and starting
    a JVM to read it would make the one command you reach for mid-incident the
    slowest one in the tool.
    """
    config = PipelineConfig(warehouse=args.warehouse)
    trend = compare_runs(ReportHistory(config.history_dir), args.stage, tolerance=args.tolerance)
    if trend is None:
        print(
            f"not enough history for stage {args.stage!r} under {config.history_dir}; "
            "a trend needs two runs",
            file=sys.stderr,
        )
        return 0

    if args.json:
        print(trend.to_json())
    elif args.markdown:
        print(trend.to_markdown())
    else:
        print(trend.to_text())

    return EXIT_GATE_FAILED if args.fail_on_regression and not trend.stable else 0


def _command_show(args: argparse.Namespace) -> int:
    with session_scope(app_name="tripetl-show", master=args.master) as session:
        df = _read(args, session)
        df.printSchema()  # type: ignore[attr-defined]
        df.show(args.limit, truncate=False)  # type: ignore[attr-defined]
    return 0


COMMANDS = {
    "run": _command_run,
    "sample": _command_sample,
    "quality": _command_quality,
    "schema": _command_schema,
    "trend": _command_trend,
    "show": _command_show,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )
    return COMMANDS[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
