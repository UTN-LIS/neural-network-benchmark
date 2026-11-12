"""Tests for the self-contained HTML report (``prebenchmark.report``).

The report must be usable offline: no external CSS, fonts, scripts or images.
These tests assert the HTML contract rather than pixel output.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from prebenchmark import analysis, report
from tests.test_analysis import write_export


@pytest.fixture
def analysis_result(tmp_path: Path):
    export_dir = write_export(tmp_path / "exp-report")
    return analysis.analyze_exports(export_dir)


def test_build_report_writes_self_contained_html(analysis_result, tmp_path: Path):
    out = tmp_path / "reports" / "analysis.html"
    built = report.build_report(
        analysis_result,
        out,
        generated_at="2026-01-01 00:00 UTC",
    )

    assert built.path == out
    assert out.exists()
    assert built.bytes_written > 0
    assert built.figures
    assert not built.warnings

    document = out.read_text(encoding="utf-8")
    assert document.startswith("<!doctype html>")
    assert "Análisis estadístico" in document
    assert "relu | adam | lr=0.01" in document
    assert "tanh | sgd" in document
    assert "Tabla 1a. Calidad por configuración" in document
    assert "data:image/png;base64," in document
    assert "http://" not in document
    assert "https://" not in document
    assert 'src="data:image/png;base64,' in document


def test_report_without_figures_keeps_tables(analysis_result, tmp_path: Path):
    out = tmp_path / "analysis_no_figures.html"
    built = report.build_report(analysis_result, out, include_figures=False)

    assert built.figures == ()
    document = out.read_text(encoding="utf-8")
    assert "data:image/png;base64," not in document
    assert "Tabla 1a. Calidad por configuración" in document
    assert "Comparación inferencial" in document
    assert "Los gráficos se desactivaron" in document


def test_report_escapes_user_controlled_labels(analysis_result, tmp_path: Path):
    result = analysis_result
    result.summary.loc[0, "label"] = "<script>alert('x')</script>"

    document, _, _ = report.render_report(result, include_figures=False)

    assert "<script>alert" not in document
    assert "&lt;script&gt;" in document


def test_report_documents_convergence_definitions(analysis_result, tmp_path: Path):
    document, _, _ = report.render_report(analysis_result, include_figures=False)

    assert "best_epoch" in document
    assert "time_to_best_epoch" in document
    assert "epochs_executed" in document
    assert "Cliff" in document
    assert "Mann-Whitney" in document


def test_report_lists_data_quality_warnings(analysis_result, tmp_path: Path):
    document, _, _ = report.render_report(analysis_result, include_figures=False)

    assert "Notas de calidad de datos" in document
    assert "no están en COMPLETED" in document
