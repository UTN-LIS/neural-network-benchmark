"""Tests for the command-line entry point (``prebenchmark.cli``).

Unit tests seed a small store directly; the smoke test (marked ``smoke``)
exercises the full reduced pipeline ``generate -> run -> compare -> aggregate ->
export`` offline against the synthetic wine_quality fixture.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from prebenchmark import store
from prebenchmark.cli import build_parser, main
from tests.test_analysis import write_export
from tests.test_export import EXPERIMENT, _build_store


def test_cli_docstring_lists_every_registered_subcommand():
    """Mantiene sincronizados el docstring del módulo y el parser real."""
    import prebenchmark.cli as cli_module

    parser = build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")
    registered = set(subparsers.choices)
    module_docstring = cli_module.__doc__ or ""
    missing = sorted(
        command for command in registered if f"``{command}``" not in module_docstring
    )
    assert not missing, f"subcomandos sin documentar en el docstring: {missing}"


def test_cli_without_command_prints_help():
    assert main([]) == 2


def test_cli_unknown_subcommand_exits():
    with pytest.raises(SystemExit):
        main(["not-a-command"])


def test_cli_generate_registers_limited_space(tmp_path: Path):
    root = tmp_path / "results"
    assert main(["generate", "--experiment-id", "exp-gen", "--root", str(root), "--limit", "6"]) == 0
    configs = store.ResultsStore(root).load_configs("exp-gen")
    assert len(configs) == 6
    assert all(config["validation_state"] in {"VALID", "WARNING"} for config in configs)


def test_cli_dataset_flag_defaults_to_wine_quality_and_is_forwarded(tmp_path: Path):
    parser = build_parser()
    args = parser.parse_args(["generate", "--experiment-id", "exp-gen"])
    assert args.dataset == "wine_quality"
    args = parser.parse_args(
        ["run", "--experiment-id", "exp-run", "--dataset", "satimage", "--limit", "1"]
    )
    assert args.dataset == "satimage"

    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--experiment-id", "exp-run", "--dataset", "unknown"])



def test_cli_export_writes_tables(tmp_path: Path):
    root = tmp_path / "results"
    _build_store(root)
    out = tmp_path / "exports"
    code = main(
        [
            "export",
            "--experiment-id",
            EXPERIMENT,
            "--root",
            str(root),
            "--out",
            str(out),
            "--baseline-run-id",
            "run-01",
            "--no-plots",
        ]
    )
    assert code == 0
    assert (out / EXPERIMENT / next((out / EXPERIMENT).glob("master_*.csv")).name).exists()


def test_cli_compare_and_aggregate(tmp_path: Path):
    root = tmp_path / "results"
    _build_store(root)
    assert main(["compare", "--experiment-id", EXPERIMENT, "--root", str(root), "--baseline-run-id", "run-01"]) == 0
    assert main(["aggregate", "--experiment-id", EXPERIMENT, "--root", str(root)]) == 0


def test_cli_analyze_writes_report(tmp_path: Path):
    export_dir = write_export(tmp_path / "exports" / "exp-analysis")
    out = tmp_path / "reports"
    code = main(
        [
            "analyze",
            "--experiment-id",
            "exp-analysis",
            "--export-dir",
            str(export_dir),
            "--out",
            str(out),
            "--no-figures",
        ]
    )
    assert code == 0
    reports = list((out / "exp-analysis").glob("analysis_*.html"))
    assert len(reports) == 1
    assert "Tabla 1a. Calidad por configuración" in reports[0].read_text(encoding="utf-8")


def test_cli_analyze_pdf_flag_writes_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from prebenchmark import pdf

    export_dir = write_export(tmp_path / "exports" / "exp-pdf")
    out = tmp_path / "reports"

    def fake_pdf(html_path, pdf_path, **kwargs):
        path = Path(pdf_path)
        path.write_bytes(b"%PDF-1.4\n")
        return path

    monkeypatch.setattr(pdf, "html_to_pdf", fake_pdf)
    code = main(
        [
            "analyze",
            "--experiment-id",
            "exp-pdf",
            "--export-dir",
            str(export_dir),
            "--out",
            str(out),
            "--no-figures",
            "--pdf",
        ]
    )
    assert code == 0
    assert len(list((out / "exp-pdf").glob("analysis_*.pdf"))) == 1


def test_cli_analyze_pdf_failure_returns_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    from prebenchmark import pdf

    export_dir = write_export(tmp_path / "exports" / "exp-pdf-fail")
    out = tmp_path / "reports"

    def failing_pdf(html_path, pdf_path, **kwargs):
        raise RuntimeError("no browser")

    monkeypatch.setattr(pdf, "html_to_pdf", failing_pdf)
    code = main(
        [
            "analyze",
            "--experiment-id",
            "exp-pdf-fail",
            "--export-dir",
            str(export_dir),
            "--out",
            str(out),
            "--no-figures",
            "--pdf",
        ]
    )
    assert code == 1
    assert "no se pudo generar el PDF" in capsys.readouterr().err


def test_cli_analyze_missing_export_returns_error(tmp_path: Path, capsys):
    code = main(
        [
            "analyze",
            "--experiment-id",
            "exp-missing",
            "--root",
            str(tmp_path / "results"),
        ]
    )
    assert code == 1
    assert "directorio de exportación no encontrado" in capsys.readouterr().err.lower()


def test_cli_compare_without_baseline_returns_error(tmp_path: Path, capsys):
    root = tmp_path / "results"
    _build_store(root)
    assert main(["compare", "--experiment-id", EXPERIMENT, "--root", str(root)]) == 1
    assert "baseline" in capsys.readouterr().err.lower()


def test_cli_run_unknown_config_id_returns_error(tmp_path: Path):
    root = tmp_path / "results"
    code = main(
        ["run", "--experiment-id", "exp-run", "--root", str(root), "--config-id", "cfg-v1-doesnotexist"]
    )
    assert code == 1


def test_cli_run_multi_seed_expands_fixed_seed_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wine_quality_csv
):
    from prebenchmark.seeds import EXPLORATORY_SEEDS

    monkeypatch.setattr(
        "prebenchmark.data.fetch_wine_quality_csv",
        lambda cache_dir=None: Path(wine_quality_csv),
    )
    root = tmp_path / "results"
    experiment = "exp-multi"
    assert main(
        [
            "run",
            "--experiment-id",
            experiment,
            "--root",
            str(root),
            "--limit",
            "1",
            "--max-epochs",
            "1",
            "--multi-seed",
        ]
    ) == 0
    runs = store.ResultsStore(root).query_runs(experiment)
    assert len(runs) == len(EXPLORATORY_SEEDS)
    assert {meta["seed"] for meta in runs} == set(EXPLORATORY_SEEDS)

    from prebenchmark import aggregate

    group = aggregate.aggregate_runs(runs)[0]
    assert group.n_valid == len(EXPLORATORY_SEEDS)
    assert group.metrics["val_loss"].n == len(EXPLORATORY_SEEDS)
    assert group.metrics["val_loss"].mean is not None
    assert group.metrics["val_accuracy"].n == len(EXPLORATORY_SEEDS)
    assert group.metrics["val_accuracy"].mean is not None


def test_cli_run_early_stop_enables_convergence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wine_quality_csv
):
    monkeypatch.setattr(
        "prebenchmark.data.fetch_wine_quality_csv",
        lambda cache_dir=None: Path(wine_quality_csv),
    )
    root = tmp_path / "results"
    experiment = "exp-early"
    assert main(
        [
            "run",
            "--experiment-id",
            experiment,
            "--root",
            str(root),
            "--limit",
            "1",
            "--max-epochs",
            "2",
            "--earlyStop",
        ]
    ) == 0
    runs = store.ResultsStore(root).query_runs(experiment)
    assert len(runs) == 1
    convergence = runs[0]["convergence"]
    assert convergence["enabled"] is True
    # Unspecified fields are resolved to the documented defaults when enabled.
    assert convergence["metric"] == "val_loss"
    assert convergence["patience"] == 20
    assert convergence["restore_best_weights"] is True


def test_cli_run_class_weight_balanced_records_preprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wine_quality_csv
):
    monkeypatch.setattr(
        "prebenchmark.data.fetch_wine_quality_csv",
        lambda cache_dir=None: Path(wine_quality_csv),
    )
    root = tmp_path / "results"
    assert main(
        [
            "run",
            "--experiment-id",
            "exp-cw",
            "--root",
            str(root),
            "--limit",
            "1",
            "--max-epochs",
            "2",
            "--class-weight",
            "balanced",
        ]
    ) == 0
    runs = store.ResultsStore(root).query_runs("exp-cw")
    assert len(runs) == 1
    assert runs[0]["preprocessing"]["class_weight"] == "balanced"
    assert runs[0]["config_snapshot"]["preprocessing"]["class_weight"] == "balanced"


def test_cli_class_weight_changes_config_id_but_not_space_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wine_quality_csv
):
    """``class_weight`` no divide el espacio de 36, pero sí produce otro ``config_id``."""
    from prebenchmark import space
    from prebenchmark.config import ExperimentConfig

    monkeypatch.setattr(
        "prebenchmark.data.fetch_wine_quality_csv",
        lambda cache_dir=None: Path(wine_quality_csv),
    )
    root = tmp_path / "results"
    common = ["--root", str(root), "--limit", "1", "--max-epochs", "1"]
    assert main(["run", "--experiment-id", "exp-plain", *common]) == 0
    assert main(
        ["run", "--experiment-id", "exp-cw", *common, "--class-weight", "balanced"]
    ) == 0

    plain = store.ResultsStore(root).query_runs("exp-plain")[0]
    balanced = store.ResultsStore(root).query_runs("exp-cw")[0]
    assert plain["preprocessing"]["class_weight"] is None
    assert balanced["preprocessing"]["class_weight"] == "balanced"
    assert plain["config_id"] != balanced["config_id"]
    plain_key = space.compute_space_key(ExperimentConfig.from_dict(plain["config_snapshot"]))
    balanced_key = space.compute_space_key(
        ExperimentConfig.from_dict(balanced["config_snapshot"])
    )
    assert plain_key == balanced_key


@pytest.mark.smoke
def test_cli_e2e_reduced_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wine_quality_csv):
    # Keep the pipeline fully offline: never hit UCI.
    monkeypatch.setattr(
        "prebenchmark.data.fetch_wine_quality_csv",
        lambda cache_dir=None: Path(wine_quality_csv),
    )
    root = tmp_path / "results"
    out = tmp_path / "exports"
    experiment = "exp-e2e"

    assert main(["generate", "--experiment-id", experiment, "--root", str(root), "--limit", "2"]) == 0
    assert main(
        ["run", "--experiment-id", experiment, "--root", str(root), "--limit", "2", "--max-epochs", "3"]
    ) == 0

    results = store.ResultsStore(root)
    runs = results.query_runs(experiment)
    assert len(runs) == 2
    assert all(meta.get("execution_state") in {"COMPLETED", "COMPLETED_WITH_WARNINGS"} for meta in runs)

    baseline = runs[0]["run_id"]
    assert main(
        ["compare", "--experiment-id", experiment, "--root", str(root), "--baseline-run-id", baseline]
    ) == 0
    assert main(["aggregate", "--experiment-id", experiment, "--root", str(root)]) == 0
    assert main(
        [
            "export",
            "--experiment-id",
            experiment,
            "--root",
            str(root),
            "--out",
            str(out),
            "--baseline-run-id",
            baseline,
            "--no-plots",
        ]
    ) == 0

    export_dir = out / experiment
    for name in ("master", "baseline_comparison", "aggregates", "epochs", "configs", "decisions", "warnings_summary"):
        assert list(export_dir.glob(f"{name}_*.csv")), f"missing {name} export"

    master = next(export_dir.glob("master_*.csv"))
    rows = list(csv.DictReader(master.open(encoding="utf-8")))
    assert len(rows) == 2
    assert all(row["run_id"] for row in rows)
