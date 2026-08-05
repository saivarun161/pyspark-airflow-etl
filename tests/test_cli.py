"""Argument parsing and command exit codes."""

from __future__ import annotations

from pathlib import Path

import pytest

from tripetl.cli import EXIT_GATE_FAILED, build_parser, main
from tripetl.quality.history import DEFAULT_TOLERANCE
from tripetl.sources import generate_sample
from tripetl.transforms.clean import clean

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def bronze_dataset(tmp_path_factory, spark) -> str:
    """A clean, bronze-shaped dataset on disk for the read-only commands."""
    path = str(tmp_path_factory.mktemp("cli") / "bronze")
    clean(generate_sample(spark, rows=200, seed=4, dirty_rate=0.0)).write.parquet(path)
    return path


def test_a_subcommand_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_run_defaults_are_offline_and_strict():
    args = build_parser().parse_args(["run"])
    assert args.input is None
    assert args.no_gates is False
    assert args.no_quarantine is False
    assert args.dirty_rate == 0.04


def test_run_flags_invert_the_defaults():
    args = build_parser().parse_args(["run", "--no-gates", "--no-quarantine", "--no-schema-check"])
    assert args.no_gates is True
    assert args.no_quarantine is True
    assert args.no_schema_check is True


def test_run_checks_the_input_schema_by_default():
    assert build_parser().parse_args(["run"]).no_schema_check is False


def test_an_unknown_stage_is_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["quality", "--path", "x", "--stage", "platinum"])


def test_sample_writes_parquet(tmp_path, spark, capsys):
    out = tmp_path / "sample"
    assert main(["sample", "--out", str(out), "--rows", "100", "--seed", "2"]) == 0
    assert "wrote" in capsys.readouterr().out
    assert spark.read.parquet(str(out)).count() > 0


def test_sample_writes_csv(tmp_path, spark):
    out = tmp_path / "sample_csv"
    assert (
        main(
            [
                "sample",
                "--out",
                str(out),
                "--rows",
                "50",
                "--dirty-rate",
                "0",
                "--format",
                "csv",
            ]
        )
        == 0
    )
    assert spark.read.option("header", "true").csv(str(out)).count() == 50


def test_run_succeeds_and_reports(tmp_path, capsys):
    exit_code = main(["run", "--warehouse", str(tmp_path / "wh"), "--rows", "300", "--seed", "6"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "pipeline PASSED" in output
    assert "quality report :: gold" in output


def test_run_returns_the_gate_exit_code_on_bad_data(tmp_path, capsys):
    """A distinct code so a scheduler can tell bad data from a crash."""
    exit_code = main(
        [
            "run",
            "--warehouse",
            str(tmp_path / "wh"),
            "--rows",
            "300",
            "--dirty-rate",
            "0.6",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == EXIT_GATE_FAILED
    assert "quality gate failed" in captured.err
    assert not (tmp_path / "wh" / "gold").exists()


def test_run_with_gates_disabled_succeeds_on_bad_data(tmp_path):
    exit_code = main(
        [
            "run",
            "--warehouse",
            str(tmp_path / "wh"),
            "--rows",
            "300",
            "--dirty-rate",
            "0.6",
            "--no-gates",
        ]
    )
    assert exit_code == 0


def test_quality_checks_a_dataset_in_place(bronze_dataset, capsys):
    exit_code = main(["quality", "--path", bronze_dataset, "--stage", "bronze"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "PASSED" in output


def test_quality_can_emit_markdown(bronze_dataset, capsys):
    main(["quality", "--path", bronze_dataset, "--stage", "bronze", "--markdown"])
    assert capsys.readouterr().out.startswith("| rule |")


def test_quality_fails_on_a_dirty_dataset(tmp_path, spark):
    path = str(tmp_path / "dirty")
    clean(generate_sample(spark, rows=400, seed=9, dirty_rate=0.6)).write.parquet(path)

    assert main(["quality", "--path", path, "--stage", "bronze"]) == EXIT_GATE_FAILED


def test_show_previews_a_dataset(bronze_dataset, capsys):
    assert main(["show", "--path", bronze_dataset, "--limit", "3"]) == 0
    assert "pickup_at" in capsys.readouterr().out


def test_run_accepts_an_explicit_input_path(tmp_path, spark):
    raw = str(tmp_path / "raw")
    generate_sample(spark, rows=200, seed=12, dirty_rate=0.0).write.parquet(raw)

    exit_code = main(["run", "--input", raw, "--warehouse", str(tmp_path / "wh")])
    assert exit_code == 0
    assert Path(tmp_path / "wh" / "gold").exists()


def test_schema_reports_a_conforming_extract(tmp_path, spark, capsys):
    raw = str(tmp_path / "raw")
    generate_sample(spark, rows=100, seed=12, dirty_rate=0.0).write.parquet(raw)

    assert main(["schema", "--path", raw]) == 0
    assert "CONFORMS" in capsys.readouterr().out


def test_schema_flags_a_drifted_extract(tmp_path, spark, capsys):
    raw = str(tmp_path / "raw")
    generate_sample(spark, rows=100, seed=12, dirty_rate=0.0).drop("fare_amount").write.parquet(raw)

    assert main(["schema", "--path", raw]) == EXIT_GATE_FAILED
    output = capsys.readouterr().out
    assert "DRIFTED" in output
    assert "fare_amount" in output


def test_schema_can_emit_json(tmp_path, spark, capsys):
    import json

    raw = str(tmp_path / "raw")
    generate_sample(spark, rows=100, seed=12, dirty_rate=0.0).write.parquet(raw)

    assert main(["schema", "--path", raw, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["conforms"] is True


def test_run_blocks_on_a_drifted_input(tmp_path, spark, capsys):
    raw = str(tmp_path / "raw")
    generate_sample(spark, rows=200, seed=12, dirty_rate=0.0).drop("fare_amount").write.parquet(raw)

    exit_code = main(["run", "--input", raw, "--warehouse", str(tmp_path / "wh")])
    captured = capsys.readouterr()

    assert exit_code == EXIT_GATE_FAILED
    assert "schema drift" in captured.err
    assert not (tmp_path / "wh" / "gold").exists()


# -- trend ------------------------------------------------------------------


def test_trend_defaults_to_reporting_without_blocking():
    args = build_parser().parse_args(["trend", "--stage", "bronze"])
    assert args.fail_on_regression is False
    assert args.tolerance == DEFAULT_TOLERANCE


def test_trend_rejects_an_unknown_stage():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["trend", "--stage", "platinum"])


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory) -> str:
    """A warehouse whose second run is materially dirtier than its first."""
    warehouse = str(tmp_path_factory.mktemp("trend") / "wh")
    main(["run", "--warehouse", warehouse, "--rows", "300", "--seed", "41"])
    main(
        [
            "run",
            "--warehouse",
            warehouse,
            "--rows",
            "300",
            "--seed",
            "42",
            "--dirty-rate",
            "0.3",
            "--no-gates",
        ]
    )
    return warehouse


def test_trend_reports_the_movement_between_two_runs(two_runs, capsys):
    assert main(["trend", "--warehouse", two_runs, "--stage", "bronze"]) == 0
    output = capsys.readouterr().out

    assert "quality trend :: bronze" in output
    assert "REGRESSED" in output


def test_trend_only_blocks_when_asked(two_runs):
    exit_code = main(
        ["trend", "--warehouse", two_runs, "--stage", "bronze", "--fail-on-regression"]
    )
    assert exit_code == EXIT_GATE_FAILED


def test_trend_can_emit_json(two_runs, capsys):
    import json

    assert main(["trend", "--warehouse", two_runs, "--stage", "bronze", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "bronze"
    assert payload["regression_count"] >= 1


def test_trend_can_emit_markdown(two_runs, capsys):
    main(["trend", "--warehouse", two_runs, "--stage", "bronze", "--markdown"])
    assert capsys.readouterr().out.startswith("| rule |")


def test_trend_says_so_when_there_is_not_enough_history(tmp_path, capsys):
    warehouse = str(tmp_path / "wh")
    main(["run", "--warehouse", warehouse, "--rows", "200", "--seed", "43"])

    assert main(["trend", "--warehouse", warehouse, "--stage", "bronze"]) == 0
    assert "not enough history" in capsys.readouterr().err


def test_trend_on_an_empty_warehouse_is_not_an_error(tmp_path, capsys):
    assert main(["trend", "--warehouse", str(tmp_path / "nothing"), "--stage", "bronze"]) == 0
    assert "not enough history" in capsys.readouterr().err
