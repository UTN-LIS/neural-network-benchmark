"""Reporte HTML autocontenido para el análisis estadístico de un experimento.

El reporte consume un :class:`prebenchmark.analysis.AnalysisResult` y produce un
único archivo ``.html`` sin dependencias externas: el CSS es inline y cada figura
se embebe como PNG en base64. Se puede abrir con doble clic, compartir por correo
o archivar junto al snapshot exportado.

Las figuras se generan con matplotlib en backend ``Agg``. Si una figura falla, el
reporte se escribe igual y la advertencia queda registrada: los gráficos son un
medio, no un requisito de validez del análisis.
"""

from __future__ import annotations

import base64
import html
import io
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .analysis import AnalysisResult

__all__ = ["ReportResult", "build_report", "render_report"]

_NA = "—"
_CLASS_WEIGHT_COLORS: Mapping[str, str] = {
    "none": "#0b5fa5",
    "balanced": "#d97706",
    "unknown": "#6b7280",
}
_DIMENSION_LABELS: Mapping[str, str] = {
    "activation": "Activación",
    "optimizer": "Optimizador",
    "lr": "Tasa de aprendizaje",
    "init": "Inicialización",
    "class_weight": "Peso de clase",
}
_METRIC_LABELS: Mapping[str, str] = {
    "f1_macro": "F1 macro",
    "accuracy": "Exactitud",
    "balanced_accuracy": "Exactitud balanceada",
    "precision_macro": "Precisión macro",
    "recall_macro": "Recall macro",
    "val_loss": "Pérdida de validación",
    "val_accuracy": "Exactitud de validación",
    "best_epoch": "Mejor época",
    "epochs_executed": "Épocas ejecutadas",
    "time_to_best_epoch": "Tiempo hasta la mejor época (s)",
    "training_time": "Tiempo de entrenamiento (s)",
}


# ---------------------------------------------------------------------------
# Formateo seguro para HTML
# ---------------------------------------------------------------------------


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _number(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return _NA
    if not math.isfinite(numeric):
        return _NA
    if numeric == 0:
        return "0"
    magnitude = abs(numeric)
    if magnitude >= 1000 or magnitude < 0.001:
        return f"{numeric:.{digits}e}"
    return f"{numeric:.{digits}f}"


def _p_value(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return _NA
    if not math.isfinite(numeric):
        return _NA
    if numeric < 0.001:
        return "&lt; 0.001"
    return f"{numeric:.3f}"


def _delta(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return _NA
    if not math.isfinite(numeric):
        return _NA
    return f"{numeric:+.{digits}f}"


def _interval(low: Any, high: Any, digits: int = 3) -> str:
    low_text = _number(low, digits)
    high_text = _number(high, digits)
    if low_text == _NA or high_text == _NA:
        return _NA
    return f"[{low_text}, {high_text}]"


def _badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="badge badge-{_escape(kind)}">{_escape(text)}</span>'


def _short_label(row: Mapping[str, Any]) -> str:
    label = row.get("label")
    if isinstance(label, str) and label.strip():
        return label
    return (
        f"{row.get('activation', '?')} / {row.get('optimizer', '?')} / "
        f"lr={row.get('lr_label', row.get('lr', '?'))} / {row.get('init', '?')} / "
        f"cw={row.get('class_weight', '?')}"
    )


# ---------------------------------------------------------------------------
# Figuras (matplotlib -> base64)
# ---------------------------------------------------------------------------


def _figure_to_base64(figure: Any) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _setup_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    return plt


def _color_for(class_weight: Any) -> str:
    return _CLASS_WEIGHT_COLORS.get(str(class_weight), "#6b7280")


def _ranking_bars(axis: Any, panel: pd.DataFrame, reference: float, *, fontsize: float) -> None:
    ordered = panel.sort_values("f1_macro_mean", ascending=True)
    positions = np.arange(len(ordered))
    values = ordered["f1_macro_mean"].to_numpy(dtype=float)
    errors = ordered["f1_macro_std"].fillna(0.0).to_numpy(dtype=float)
    colors = [_color_for(value) for value in ordered["class_weight"]]
    axis.barh(positions, values, xerr=errors, color=colors, alpha=0.9, capsize=2, error_kw={"elinewidth": 0.8})
    axis.set_yticks(positions)
    axis.set_yticklabels([_short_label(row) for _, row in ordered.iterrows()], fontsize=fontsize)
    axis.axvline(reference, color="#111827", linestyle=":", linewidth=1.0)
    axis.text(reference, len(ordered) - 0.2, " referencia", va="top", fontsize=7, color="#111827")


def _ranking_figure(result: AnalysisResult, plt: Any) -> Any:
    summary = result.summary
    reference = float(summary.iloc[0]["f1_macro_mean"])
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=f"class_weight={label}")
        for label, color in _CLASS_WEIGHT_COLORS.items()
        if label in set(summary["class_weight"].astype(str))
    ]

    if len(summary) > 30:
        ordered = summary.sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)
        half = math.ceil(len(ordered) / 2)
        panels = [ordered.iloc[:half], ordered.iloc[half:]]
        figure, axes = plt.subplots(1, 2, figsize=(7.5, 8.4), sharex=True)
        for index, (axis, panel) in enumerate(zip(axes, panels)):
            _ranking_bars(axis, panel, reference, fontsize=7.0)
            start = index * half + 1
            axis.set_title(f"Puestos {start}–{start + len(panel) - 1}", fontsize=8)
        figure.supxlabel("F1 macro medio entre semillas (barra de error = desvío muestral)", fontsize=9)
        figure.suptitle("Ranking de configuraciones por F1 macro medio", fontsize=10)
        figure.tight_layout(rect=(0, 0.03, 1, 0.96))
        return figure

    figure, axis = plt.subplots(figsize=(9.5, max(4.5, 0.14 * len(summary) + 1.4)))
    _ranking_bars(axis, summary, reference, fontsize=7.5)
    axis.set_xlabel("F1 macro medio entre semillas (barra de error = desvío muestral)")
    axis.set_title("Ranking de configuraciones por F1 macro medio")
    if handles:
        axis.legend(handles=handles, loc="lower right", fontsize=7, frameon=False)
    return figure


def _f1_distribution_figure(result: AnalysisResult, plt: Any) -> Any:
    runs = result.runs[result.runs["valid_run"]]
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    values = pd.to_numeric(runs["f1_macro"], errors="coerce").dropna().to_numpy(dtype=float)
    axes[0].hist(values, bins=min(24, max(6, len(values) // 4)), color="#0b5fa5", alpha=0.85, edgecolor="white")
    if values.size:
        axes[0].axvline(float(np.mean(values)), color="#111827", linestyle="--", linewidth=1.0, label="media")
        axes[0].axvline(float(np.median(values)), color="#d97706", linestyle=":", linewidth=1.2, label="mediana")
        axes[0].legend(fontsize=7, frameon=False)
    axes[0].set_xlabel("F1 macro por corrida")
    axes[0].set_ylabel("Corridas")
    axes[0].set_title("Distribución de F1 macro por corrida")

    for class_weight, subset in runs.groupby("class_weight", dropna=False):
        series = pd.to_numeric(subset["f1_macro"], errors="coerce").dropna().to_numpy(dtype=float)
        if series.size == 0:
            continue
        axes[1].hist(
            series,
            bins=min(20, max(5, series.size // 3)),
            alpha=0.45,
            label=f"cw={class_weight} (n={series.size})",
            color=_color_for(class_weight),
            edgecolor="white",
        )
    axes[1].set_xlabel("F1 macro por corrida")
    axes[1].set_ylabel("Corridas")
    axes[1].set_title("F1 macro por variante de class_weight")
    axes[1].legend(fontsize=7, frameon=False)
    figure.tight_layout()
    return figure


def _dimension_boxplots_figure(result: AnalysisResult, plt: Any) -> Any:
    runs = result.runs[result.runs["valid_run"]]
    dimensions = [dimension for dimension in ("activation", "optimizer", "lr", "init") if dimension in runs.columns]
    if not dimensions:
        return None

    columns = 2
    rows = math.ceil(len(dimensions) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(11, 3.4 * rows), squeeze=False)
    for index, dimension in enumerate(dimensions):
        axis = axes[index // columns][index % columns]
        grouped = (
            runs.assign(_level=runs[dimension].astype(str))
            .groupby("_level")["f1_macro"]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
        )
        data = [
            pd.to_numeric(runs.loc[runs[dimension].astype(str) == level, "f1_macro"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
            for level in grouped.index
        ]
        if not data:
            axis.set_visible(False)
            continue
        box = axis.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
        for patch, level in zip(box["boxes"], grouped.index):
            patch.set_facecolor(_color_for(level) if dimension == "class_weight" else "#93c5fd")
            patch.set_alpha(0.85)
        axis.set_xticks(range(1, len(grouped) + 1))
        axis.set_xticklabels(
            [f"{level}\n(n={int(grouped.loc[level, 'count'])})" for level in grouped.index],
            fontsize=7.5,
        )
        axis.set_ylabel("F1 macro por corrida")
        axis.set_title(f"F1 macro según {_DIMENSION_LABELS.get(dimension, dimension)}")
    for index in range(len(dimensions), rows * columns):
        axes[index // columns][index % columns].set_visible(False)
    figure.tight_layout()
    return figure


def _convergence_scatter_figure(result: AnalysisResult, plt: Any) -> Any:
    runs = result.runs[result.runs["valid_run"]]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    markers = {"adam": "o", "sgd": "s"}
    for class_weight, subset in runs.groupby("class_weight", dropna=False):
        for optimizer, optimizer_subset in subset.groupby("optimizer", dropna=False):
            marker = markers.get(str(optimizer), "^")
            axes[0].scatter(
                pd.to_numeric(optimizer_subset["time_to_best_epoch"], errors="coerce"),
                pd.to_numeric(optimizer_subset["f1_macro"], errors="coerce"),
                s=26,
                alpha=0.75,
                color=_color_for(class_weight),
                marker=marker,
                label=f"cw={class_weight}, {optimizer}",
                edgecolors="white",
                linewidths=0.4,
            )
            axes[1].scatter(
                pd.to_numeric(optimizer_subset["best_epoch"], errors="coerce"),
                pd.to_numeric(optimizer_subset["f1_macro"], errors="coerce"),
                s=26,
                alpha=0.75,
                color=_color_for(class_weight),
                marker=marker,
                edgecolors="white",
                linewidths=0.4,
            )
    axes[0].set_xlabel("Tiempo hasta la mejor época (s, epoch_time acumulado)")
    axes[1].set_xlabel("Mejor época (menor val_loss)")
    evaluation_set = (
        str(result.runs["evaluation_set"].iloc[0])
        if "evaluation_set" in result.runs.columns and not result.runs.empty
        else "evaluation"
    )
    for axis in axes:
        axis.set_ylabel(f"F1 macro (conjunto de {evaluation_set})")
        reference = float(result.summary.iloc[0]["f1_macro_mean"])
        axis.axhline(reference, color="#111827", linestyle=":", linewidth=0.9)
    axes[0].set_title("Calidad vs tiempo de convergencia")
    axes[1].set_title("Calidad vs época de convergencia")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=min(4, len(handles)), fontsize=7, frameon=False)
        figure.tight_layout(rect=(0, 0.08, 1, 1))
    else:
        figure.tight_layout()
    return figure


def _convergence_histograms_figure(result: AnalysisResult, plt: Any) -> Any:
    runs = result.runs[result.runs["valid_run"]]
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    best_epochs = pd.to_numeric(runs["best_epoch"], errors="coerce").dropna().to_numpy(dtype=float)
    axes[0].hist(
        best_epochs,
        bins=min(24, max(6, best_epochs.size // 4)) if best_epochs.size else 6,
        color="#0b5fa5",
        alpha=0.85,
        edgecolor="white",
    )
    if best_epochs.size:
        axes[0].axvline(float(np.median(best_epochs)), color="#d97706", linestyle="--", linewidth=1.0, label="mediana")
        axes[0].legend(fontsize=7, frameon=False)
    axes[0].set_xlabel("Mejor época")
    axes[0].set_ylabel("Corridas")
    axes[0].set_title("Cuándo ocurre la mejor pérdida de validación")

    times = pd.to_numeric(runs["time_to_best_epoch"], errors="coerce").dropna().to_numpy(dtype=float)
    axes[1].hist(
        times,
        bins=min(24, max(6, times.size // 4)) if times.size else 6,
        color="#0b5fa5",
        alpha=0.85,
        edgecolor="white",
    )
    if times.size:
        axes[1].axvline(float(np.median(times)), color="#d97706", linestyle="--", linewidth=1.0, label="mediana")
        axes[1].legend(fontsize=7, frameon=False)
    axes[1].set_xlabel("Tiempo hasta la mejor época (s)")
    axes[1].set_ylabel("Corridas")
    axes[1].set_title("Tiempo de entrenamiento acumulado en la mejor época")
    figure.tight_layout()
    return figure


def _heatmap_figure(result: AnalysisResult, plt: Any) -> Any:
    runs = result.runs[result.runs["valid_run"]]
    if "activation" not in runs.columns or "optimizer" not in runs.columns:
        return None
    variants = sorted(runs["class_weight"].astype(str).unique())
    pivots: dict[str, pd.DataFrame] = {}
    for class_weight in variants:
        subset = runs[runs["class_weight"].astype(str) == class_weight]
        pivot = subset.pivot_table(
            index="activation",
            columns="optimizer",
            values="f1_macro",
            aggfunc="mean",
        )
        if not pivot.empty:
            pivots[class_weight] = pivot
    if not pivots:
        return None

    all_values = np.concatenate(
        [pivot.to_numpy(dtype=float).ravel() for pivot in pivots.values()]
    )
    finite_values = all_values[np.isfinite(all_values)]
    vmin = float(finite_values.min()) if finite_values.size else 0.0
    vmax = float(finite_values.max()) if finite_values.size else 1.0
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    columns = len(pivots)
    figure, axes = plt.subplots(1, columns, figsize=(5.2 * columns, 4.0), squeeze=False)
    for index, (class_weight, pivot) in enumerate(pivots.items()):
        axis = axes[0][index]
        data = pivot.to_numpy(dtype=float)
        image = axis.imshow(data, cmap="Blues", aspect="auto", vmin=vmin, vmax=vmax)
        axis.set_xticks(range(pivot.shape[1]))
        axis.set_xticklabels([str(column) for column in pivot.columns])
        axis.set_yticks(range(pivot.shape[0]))
        axis.set_yticklabels([str(index_value) for index_value in pivot.index])
        axis.set_xlabel("Optimizador")
        axis.set_ylabel("Activación")
        axis.set_title(f"F1 macro medio · class_weight={class_weight}")
        axis.grid(False)
        for row in range(pivot.shape[0]):
            for column in range(pivot.shape[1]):
                value = data[row, column]
                if math.isfinite(value):
                    color = "white" if value > (vmin + vmax) / 2 else "#111827"
                    axis.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=8, color=color)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="F1 macro medio")
    figure.tight_layout()
    return figure


def _convergence_curves_figure(result: AnalysisResult, plt: Any, top: int = 6) -> Any:
    epochs = result.bundle.epochs
    if epochs.empty:
        return None
    runs = result.runs[result.runs["valid_run"]]
    selected = result.summary.head(top)["group_key"].astype(str).tolist()
    if result.reference_group_key not in selected:
        selected.append(result.reference_group_key)
    selected = list(dict.fromkeys(selected))[: max(top, 1) + 1]

    columns = min(3, len(selected))
    rows = math.ceil(len(selected) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(11, 3.0 * rows), squeeze=False)
    for index, group_key in enumerate(selected):
        axis = axes[index // columns][index % columns]
        run_ids = runs.loc[runs["group_key"].astype(str) == group_key, "run_id"].astype(str).tolist()
        subset = epochs[epochs["run_id"].astype(str).isin(run_ids)].copy()
        if subset.empty:
            axis.set_visible(False)
            continue
        for run_id, run_epochs in subset.groupby("run_id"):
            ordered = run_epochs.sort_values("epoch")
            axis.plot(ordered["epoch"], ordered["val_loss"], color="#9ca3af", linewidth=0.7, alpha=0.7)
        aggregated = subset.groupby("epoch")["val_loss"].agg(["mean", "std", "count"]).reset_index()
        axis.plot(aggregated["epoch"], aggregated["mean"], color="#0b5fa5", linewidth=1.4, label="media")
        deviation = aggregated["std"].fillna(0.0).to_numpy(dtype=float)
        axis.fill_between(
            aggregated["epoch"].to_numpy(dtype=float),
            aggregated["mean"].to_numpy(dtype=float) - deviation,
            aggregated["mean"].to_numpy(dtype=float) + deviation,
            color="#0b5fa5",
            alpha=0.18,
        )
        row = result.summary[result.summary["group_key"].astype(str) == group_key]
        label = _short_label(row.iloc[0]) if not row.empty else group_key
        axis.set_title(label, fontsize=8)
        axis.set_xlabel("Época")
        axis.set_ylabel("Pérdida de validación")
    for index in range(len(selected), rows * columns):
        axes[index // columns][index % columns].set_visible(False)
    figure.suptitle("Curvas de pérdida de validación de las mejores configuraciones (media ± desvío, gris = semillas individuales)", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def _forest_panel(axis: Any, panel: pd.DataFrame, *, fontsize: float) -> None:
    ordered = panel.sort_values("diff", ascending=True)
    positions = np.arange(len(ordered))
    diffs = ordered["diff"].to_numpy(dtype=float)
    low = ordered["ci95_low"].to_numpy(dtype=float)
    high = ordered["ci95_high"].to_numpy(dtype=float)
    lower_error = np.nan_to_num(diffs - low, nan=0.0)
    upper_error = np.nan_to_num(high - diffs, nan=0.0)
    colors = [
        "#b91c1c" if (math.isfinite(l) and l > 0) or (math.isfinite(h) and h < 0) else "#6b7280"
        for l, h in zip(low, high)
    ]
    axis.errorbar(
        diffs,
        positions,
        xerr=[lower_error, upper_error],
        fmt="o",
        markersize=3.5,
        color="#111827",
        ecolor=colors,
        elinewidth=1.1,
        capsize=0,
    )
    axis.axvline(0.0, color="#111827", linestyle="--", linewidth=1.0)
    axis.set_yticks(positions)
    axis.set_yticklabels([_short_label(row) for _, row in ordered.iterrows()], fontsize=fontsize)


def _forest_figure(result: AnalysisResult, plt: Any) -> Any:
    comparisons = result.comparisons[result.comparisons["metric"] == "f1_macro"].copy()
    if comparisons.empty:
        return None

    if len(comparisons) > 30:
        ordered = comparisons.sort_values("diff", ascending=False).reset_index(drop=True)
        half = math.ceil(len(ordered) / 2)
        panels = [ordered.iloc[:half], ordered.iloc[half:]]
        figure, axes = plt.subplots(1, 2, figsize=(7.5, 8.4), sharex=True)
        for index, (axis, panel) in enumerate(zip(axes, panels)):
            _forest_panel(axis, panel, fontsize=7.0)
            start = index * half + 1
            axis.set_title(f"Puestos {start}–{start + len(panel) - 1}", fontsize=8)
        figure.supxlabel("Diferencia de F1 macro vs referencia (positivo = mejor que la referencia)", fontsize=9)
        figure.suptitle("Diferencia contra la configuración de referencia (IC 95 % de Welch)", fontsize=10)
        figure.tight_layout(rect=(0, 0.03, 1, 0.96))
        return figure

    figure, axis = plt.subplots(figsize=(9.5, max(3.5, 0.12 * len(comparisons) + 1.2)))
    _forest_panel(axis, comparisons, fontsize=7.5)
    axis.set_xlabel("Diferencia de F1 macro vs referencia (positivo = mejor que la referencia)")
    axis.set_title("Diferencia contra la configuración de referencia (IC 95 % de Welch)")
    return figure


def _render_figures(result: AnalysisResult, *, top_curves: int) -> tuple[dict[str, str], list[str]]:
    """Genera todas las figuras y devuelve ``{clave: base64}`` más advertencias."""
    figures: dict[str, str] = {}
    warnings: list[str] = []
    plt = _setup_matplotlib()
    builders = (
        ("ranking", "Ranking de configuraciones", lambda: _ranking_figure(result, plt)),
        ("f1_distribution", "Distribución de F1", lambda: _f1_distribution_figure(result, plt)),
        ("dimension_boxplots", "Boxplots marginales", lambda: _dimension_boxplots_figure(result, plt)),
        ("convergence_scatter", "Dispersión de convergencia", lambda: _convergence_scatter_figure(result, plt)),
        ("convergence_histograms", "Histogramas de convergencia", lambda: _convergence_histograms_figure(result, plt)),
        ("heatmap", "Heatmap de F1", lambda: _heatmap_figure(result, plt)),
        ("convergence_curves", "Curvas de convergencia", lambda: _convergence_curves_figure(result, plt, top=top_curves)),
        ("forest", "Forest plot de pares", lambda: _forest_figure(result, plt)),
    )
    for key, label, builder in builders:
        try:
            figure = builder()
            if figure is None:
                continue
            figures[key] = _figure_to_base64(figure)
            plt.close(figure)
        except Exception as error:  # noqa: BLE001 - una figura no debe tumbar el reporte
            warnings.append(f"figura '{label}' omitida: {error}")
            try:
                plt.close("all")
            except Exception:  # pragma: no cover - defensivo
                pass
    return figures, warnings


# ---------------------------------------------------------------------------
# Bloques HTML
# ---------------------------------------------------------------------------


def _table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    caption: str | None = None,
    table_class: str = "data",
) -> str:
    head = "".join(f"<th>{header}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    caption_html = f"<caption>{caption}</caption>" if caption else ""
    return (
        f'<div class="table-wrap"><table class="{_escape(table_class)}">{caption_html}'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _summary_rows(result: AnalysisResult, kind: str = "quality") -> list[list[str]]:
    rows: list[list[str]] = []
    for _, row in result.summary.iterrows():
        class_weight = str(row.get("class_weight", ""))
        rank = _escape(row.get("rank_f1", ""))
        label = f"<strong>{_escape(_short_label(row))}</strong>"
        if kind == "quality":
            rows.append(
                [
                    rank,
                    label,
                    _badge(class_weight, "cw-balanced" if class_weight == "balanced" else "cw-none"),
                    f"{int(row.get('n_valid', 0))}/{int(row.get('n_runs', 0))}",
                    _number(row.get("f1_macro_mean"), 4),
                    _number(row.get("f1_macro_std"), 4),
                    _interval(row.get("f1_macro_ci95_low"), row.get("f1_macro_ci95_high"), 4),
                    _number(row.get("val_loss_mean"), 4),
                ]
            )
        else:
            rows.append(
                [
                    rank,
                    label,
                    _number(row.get("best_epoch_mean"), 1),
                    _number(row.get("epochs_executed_mean"), 1),
                    _number(row.get("time_to_best_epoch_mean"), 2),
                    _number(row.get("training_time_mean"), 2),
                ]
            )
    return rows


def _comparison_rows(
    result: AnalysisResult, metric: str, kind: str = "quality"
) -> list[list[str]]:
    subset = result.comparisons[result.comparisons["metric"] == metric]
    precision = 4 if metric == "f1_macro" else 2
    rows: list[list[str]] = []
    for _, row in subset.iterrows():
        try:
            significant = float(row.get("welch_p")) < 0.05
        except (TypeError, ValueError):
            significant = False
        label = _escape(_short_label(row))
        if kind == "quality":
            rows.append(
                [
                    label,
                    _number(row.get("mean_config"), precision),
                    _number(row.get("mean_reference"), precision),
                    _delta(row.get("diff"), precision),
                    _interval(row.get("ci95_low"), row.get("ci95_high"), precision),
                ]
            )
        elif kind == "significance":
            rows.append(
                [
                    label,
                    _p_value(row.get("welch_p")),
                    _p_value(row.get("mannwhitney_p")),
                    _delta(row.get("cliffs_delta"), 3),
                    _badge(str(row.get("effect_size", "n/d")), "neutral"),
                    _badge("sí" if significant else "no", "sig" if significant else "neutral"),
                ]
            )
        else:  # convergence: medias, diferencia, IC y p de Welch
            rows.append(
                [
                    label,
                    _number(row.get("mean_config"), precision),
                    _number(row.get("mean_reference"), precision),
                    _delta(row.get("diff"), precision),
                    _interval(row.get("ci95_low"), row.get("ci95_high"), precision),
                    _p_value(row.get("welch_p")),
                    _badge(str(row.get("effect_size", "n/d")), "neutral"),
                ]
            )
    return rows


def _marginal_rows(result: AnalysisResult, kind: str = "quality") -> list[list[str]]:
    rows: list[list[str]] = []
    for _, row in result.marginals.iterrows():
        dimension = str(row.get("dimension", ""))
        dimension_label = _escape(_DIMENSION_LABELS.get(dimension, dimension))
        level = f"<strong>{_escape(row.get('level', ''))}</strong>"
        if kind == "quality":
            rows.append(
                [
                    dimension_label,
                    level,
                    str(int(row.get("n_runs", 0))),
                    str(int(row.get("n_configs", 0))),
                    _number(row.get("f1_mean"), 4),
                    _number(row.get("f1_std"), 4),
                    _interval(row.get("f1_ci95_low"), row.get("f1_ci95_high"), 4),
                ]
            )
        else:
            rows.append(
                [
                    dimension_label,
                    level,
                    _number(row.get("time_mean"), 2),
                    _number(row.get("best_epoch_mean"), 1),
                    _p_value(row.get("kruskal_p")),
                ]
            )
    return rows


def _figure_block(key: str, figures: Mapping[str, str], title: str, caption: str) -> str:
    if key not in figures:
        return ""
    return (
        f'<figure id="fig-{_escape(key)}">'
        f'<img alt="{_escape(title)}" src="data:image/png;base64,{figures[key]}">'
        f"<figcaption><strong>{_escape(title)}.</strong> {caption}</figcaption>"
        "</figure>"
    )


def _observations(result: AnalysisResult) -> list[str]:
    """Observaciones automáticas, factuales y sin lenguaje causal."""
    summary = result.summary
    if summary.empty:
        return []

    observations: list[str] = []
    best = summary.iloc[0]
    observations.append(
        f"Mejor F1 macro medio: <strong>{_escape(best.get('label', ''))}</strong> con "
        f"{_number(best.get('f1_macro_mean'), 4)} ± {_number(best.get('f1_macro_std'), 4)} "
        f"(n={int(best.get('n_valid', 0))})."
    )

    if "class_weight" in summary.columns:
        for class_weight in sorted(summary["class_weight"].dropna().astype(str).unique()):
            subset = summary[summary["class_weight"].astype(str) == class_weight]
            if subset.empty:
                continue
            row = subset.iloc[0]
            observations.append(
                f"Mejor <code>class_weight={_escape(class_weight)}</code>: "
                f"{_escape(row.get('label', ''))} con {_number(row.get('f1_macro_mean'), 4)} "
                f"(puesto {int(row.get('rank_f1', 0))})."
            )

    timed = summary.dropna(subset=["time_to_best_epoch_mean"])
    if not timed.empty:
        fastest = timed.sort_values("time_to_best_epoch_mean").iloc[0]
        observations.append(
            f"Menor tiempo medio hasta la mejor época: <strong>{_escape(fastest.get('label', ''))}</strong> "
            f"con {_number(fastest.get('time_to_best_epoch_mean'), 2)} s, y puesto "
            f"{int(fastest.get('rank_f1', 0))} en F1."
        )

    valid_runs = result.runs[result.runs["valid_run"]]
    if (
        "max_epochs" in valid_runs.columns
        and valid_runs["max_epochs"].notna().any()
        and valid_runs["best_epoch"].notna().any()
    ):
        at_cap = valid_runs[
            valid_runs["best_epoch"].notna()
            & valid_runs["max_epochs"].notna()
            & (valid_runs["best_epoch"] >= valid_runs["max_epochs"])
        ]
        if not at_cap.empty:
            share = 100.0 * len(at_cap) / len(valid_runs)
            observations.append(
                f"{len(at_cap)} de {len(valid_runs)} corridas válidas ({share:.1f}%) alcanzó el tope de épocas "
                "en su mejor época, por lo que su mejor modelo podría seguir mejorando."
            )

    if not result.marginals.empty:
        ranked = result.marginals.dropna(subset=["kruskal_p"]).sort_values("kruskal_p")
        if not ranked.empty:
            row = ranked.iloc[0]
            dimension = str(row["dimension"])
            levels = result.marginals[result.marginals["dimension"] == dimension].sort_values(
                "f1_mean", ascending=False
            )
            if len(levels) >= 2:
                top = levels.iloc[0]
                bottom = levels.iloc[-1]
                observations.append(
                    f"Mayor separación marginal: <strong>{_escape(_DIMENSION_LABELS.get(dimension, dimension))}</strong> "
                    f"(Kruskal p={_p_value(row.get('kruskal_p'))}), de "
                    f"{_escape(top.get('level', ''))} con {_number(top.get('f1_mean'), 4)} a "
                    f"{_escape(bottom.get('level', ''))} con {_number(bottom.get('f1_mean'), 4)}."
                )

    return observations


def _summary_cards(result: AnalysisResult) -> str:
    best = result.summary.iloc[0]
    cards = [
        ("Configuraciones", f"{len(result.summary)}", f"{len(result.runs)} corridas"),
        (
            "Mejor F1 medio",
            _number(best.get("f1_macro_mean"), 4),
            _escape(_short_label(best)),
        ),
        (
            "Más rápida a la mejor época",
            _number(result.summary["time_to_best_epoch_mean"].min(), 2) + " s",
            "mínimo de las medias por configuración",
        ),
        (
            "Referencia",
            _escape(result.reference_group_key),
            _escape(result.reference_label),
        ),
    ]
    return "".join(
        f'<div class="card"><span class="card-label">{_escape(label)}</span>'
        f'<span class="card-value">{value}</span><span class="card-note">{note}</span></div>'
        for label, value, note in cards
    )


def _css() -> str:
    return """
:root {
  --ink: #1a1a1a;
  --muted: #5a5a5a;
  --accent: #0b5fa5;
  --accent-soft: #eaf2fa;
  --rule: #d8d8d8;
  --code-bg: #f4f6f8;
  --warn-bg: #fff7ed;
  --warn-rule: #fdba74;
}
* { box-sizing: border-box; }
body {
  margin: 0 auto;
  max-width: 1180px;
  padding: 36px 28px 64px;
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.55;
  color: var(--ink);
  background: #ffffff;
}
h1, h2, h3 { color: var(--accent); line-height: 1.25; }
h1 { font-size: 24pt; margin: 0 0 6px; border-bottom: 3px solid var(--accent); padding-bottom: 10px; }
h2 { font-size: 15pt; margin-top: 38px; border-bottom: 1px solid var(--rule); padding-bottom: 6px; }
h3 { font-size: 12pt; margin-top: 24px; }
.subtitle { font-size: 12pt; color: var(--muted); margin: 0 0 4px; }
.meta { font-size: 9pt; color: var(--muted); margin-bottom: 24px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 18px 0 8px; }
.card { border: 1px solid var(--rule); border-radius: 8px; padding: 12px 14px; background: #fbfcfe; }
.card-label { display: block; font-size: 8.5pt; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
.card-value { display: block; font-size: 16pt; font-weight: 600; color: var(--accent); margin: 4px 0 2px; overflow-wrap: anywhere; }
.card-note { display: block; font-size: 8.5pt; color: var(--muted); overflow-wrap: anywhere; }
.note { border-left: 4px solid var(--accent); background: var(--accent-soft); padding: 10px 14px; margin: 16px 0; font-size: 9.5pt; }
.warning { border-left: 4px solid var(--warn-rule); background: var(--warn-bg); padding: 10px 14px; margin: 12px 0; font-size: 9.5pt; }
.warning ul { margin: 6px 0 0 18px; padding: 0; }
.table-wrap { overflow-x: auto; margin: 14px 0 20px; border: 1px solid var(--rule); border-radius: 6px; }
table.data { border-collapse: collapse; width: 100%; font-size: 8.6pt; }
table.data caption { caption-side: top; text-align: left; font-weight: 600; color: var(--muted); padding: 8px 10px; }
table.data th, table.data td { padding: 6px 8px; border-bottom: 1px solid var(--rule); text-align: right; white-space: nowrap; }
table.data th { background: var(--accent-soft); color: #123; position: sticky; top: 0; }
table.config-table th:nth-child(2), table.config-table td:nth-child(2) { text-align: left; white-space: normal; min-width: 230px; }
table.data tbody tr:nth-child(even) { background: #fafbfc; }
table.data tbody tr:hover { background: var(--accent-soft); }
.badge { display: inline-block; padding: 1px 7px; border-radius: 999px; font-size: 7.8pt; font-weight: 600; border: 1px solid transparent; }
.badge-cw-balanced { background: #fef3c7; color: #92400e; border-color: #fcd34d; }
.badge-cw-none { background: #dbeafe; color: #1e3a8a; border-color: #93c5fd; }
.badge-neutral { background: #f3f4f6; color: #374151; border-color: #d1d5db; }
.badge-sig { background: #fee2e2; color: #991b1b; border-color: #fca5a5; }
figure { margin: 22px 0 26px; }
figure img { width: 100%; height: auto; border: 1px solid var(--rule); border-radius: 6px; }
figcaption { font-size: 9pt; color: var(--muted); margin-top: 8px; }
details { margin: 12px 0 22px; border: 1px solid var(--rule); border-radius: 6px; padding: 10px 14px; background: #fcfdfe; }
details > summary { cursor: pointer; font-weight: 600; color: var(--accent); }
code { font-family: "Cascadia Mono", Consolas, monospace; background: var(--code-bg); padding: 1px 5px; border-radius: 3px; font-size: 9pt; }
ul.tight { margin: 8px 0 8px 20px; padding: 0; }
ul.tight li { margin: 4px 0; }
footer { margin-top: 48px; padding-top: 14px; border-top: 1px solid var(--rule); font-size: 8.5pt; color: var(--muted); }
@page { size: A4; margin: 12mm; }
@media print {
  body { max-width: none; padding: 0; font-size: 9.5pt; }
  h1 { font-size: 20pt; }
  h2 { page-break-after: avoid; break-after: avoid; }
  h3 { page-break-after: avoid; break-after: avoid; }
  figure { page-break-inside: avoid; break-inside: avoid; margin: 10px 0 14px; }
  figure img { max-width: 100%; max-height: 8.4in; width: auto; height: auto; display: block; margin: 0 auto; }
  figcaption { font-size: 8pt; }
  table { page-break-inside: auto; }
  table.data { font-size: 8pt; table-layout: fixed; width: 100%; }
  table.data th, table.data td { white-space: normal; overflow-wrap: break-word; padding: 3px 4px; }
  table.config-table th:nth-child(1), table.config-table td:nth-child(1) { width: 6%; }
  table.config-table th:nth-child(2), table.config-table td:nth-child(2) { width: 26%; min-width: 0; }
  table.comparison-table th:nth-child(1), table.comparison-table td:nth-child(1) { width: 32%; }
  table.comparison-table th:nth-child(2), table.comparison-table td:nth-child(2) { width: 14%; min-width: 0; }
  table.marginal-table th:nth-child(1), table.marginal-table td:nth-child(1) { width: 16%; }
  table.marginal-table th:nth-child(2), table.marginal-table td:nth-child(2) { width: 12%; min-width: 0; }
  table.data caption { break-after: avoid; page-break-after: avoid; }
  thead { display: table-header-group; }
  tr { page-break-inside: avoid; }
  .cards { break-inside: avoid; }
  details { display: block; }
  details > summary { display: none; }
}
"""


def _html_document(
    result: AnalysisResult,
    figures: Mapping[str, str],
    *,
    title: str,
    generated_at: str,
    warnings: Sequence[str],
    include_figures: bool = True,
) -> str:
    dataset = str(result.runs["dataset"].iloc[0]) if "dataset" in result.runs.columns else "unknown"
    seeds = sorted(
        {
            int(seed)
            for seed in result.runs["seed"].dropna().unique()
            if float(seed).is_integer()
        }
    )

    warnings_html = ""
    if warnings:
        items = "".join(f"<li>{_escape(warning)}</li>" for warning in warnings)
        warnings_html = f'<div class="warning"><strong>Notas de calidad de datos</strong><ul>{items}</ul></div>'

    observations_html = ""
    observations = _observations(result)
    if observations:
        items = "".join(f"<li>{observation}</li>" for observation in observations)
        observations_html = f'<h2>Observaciones clave</h2><ul class="tight">{items}</ul>'

    summary_quality_table = _table(
        [
            "Puesto",
            "Configuración",
            "class_weight",
            "n válidas/total",
            "F1 medio",
            "F1 desvío",
            "IC 95 % de F1",
            "Pérdida val.",
        ],
        _summary_rows(result, "quality"),
        caption="Tabla 1a. Calidad por configuración, ordenada por F1 macro medio. El IC es el intervalo t del 95 % de la media entre semillas.",
        table_class="data config-table",
    )

    summary_convergence_table = _table(
        [
            "Puesto",
            "Configuración",
            "Mejor época",
            "Épocas ejecutadas",
            "Tiempo hasta la mejor (s)",
            "Tiempo de entrenamiento (s)",
        ],
        _summary_rows(result, "convergence"),
        caption="Tabla 1b. Convergencia y costo por configuración, en el mismo orden que la Tabla 1a.",
        table_class="data config-table",
    )

    f1_quality_table = _table(
        [
            "Configuración",
            "F1 medio",
            "F1 de referencia",
            "Diferencia F1",
            "IC 95 % de la diferencia",
        ],
        _comparison_rows(result, "f1_macro", "quality"),
        caption="Tabla 2a. F1 macro contra la configuración de referencia.",
        table_class="data comparison-table",
    )

    f1_significance_table = _table(
        [
            "Configuración",
            "p de Welch",
            "p de Mann-Whitney",
            "delta de Cliff",
            "Efecto",
            "p &lt; 0.05",
        ],
        _comparison_rows(result, "f1_macro", "significance"),
        caption="Tabla 2b. Significancia y tamaño de efecto, en el mismo orden de filas que la Tabla 2a. La última columna marca p de Welch < 0.05; la significancia no mide relevancia práctica.",
        table_class="data comparison-table",
    )

    convergence_tables = "".join(
        f"<details><summary>{_escape(_METRIC_LABELS.get(metric, metric))} vs referencia</summary>"
        + _table(
            [
                "Configuración",
                "Media",
                "Referencia",
                "Diferencia",
                "IC 95 % de la diferencia",
                "p de Welch",
                "Efecto",
            ],
            _comparison_rows(result, metric, "convergence"),
            caption=f"{_METRIC_LABELS.get(metric, metric)} contra la configuración de referencia.",
            table_class="data comparison-table",
        )
        + "</details>"
        for metric in ("time_to_best_epoch", "best_epoch", "training_time")
    )

    marginal_quality_table = _table(
        [
            "Dimensión",
            "Nivel",
            "Corridas",
            "Config.",
            "F1 medio",
            "F1 desvío",
            "IC 95 % de F1",
        ],
        _marginal_rows(result, "quality"),
        caption="Tabla 3a. Efecto marginal sobre F1 macro. Kruskal-Wallis se calcula entre los niveles de la misma dimensión.",
        table_class="data marginal-table",
    )

    marginal_convergence_table = _table(
        [
            "Dimensión",
            "Nivel",
            "Tiempo hasta la mejor (s)",
            "Mejor época",
            "p de Kruskal",
        ],
        _marginal_rows(result, "convergence"),
        caption="Tabla 3b. Efecto marginal sobre convergencia y costo, en el mismo orden que la Tabla 3a.",
        table_class="data marginal-table",
    )

    figures_html = "".join(
        [
            _figure_block(
                "ranking",
                figures,
                "Figura 1 — Ranking de configuraciones",
                "F1 macro medio por configuración a través de sus semillas. Las barras de error son el desvío muestral; la línea punteada marca la mejor media. El color codifica la variante de class_weight (azul = none, naranja = balanced).",
            ),
            _figure_block(
                "f1_distribution",
                figures,
                "Figura 2 — Distribución de F1",
                "Izquierda: todas las corridas válidas agrupadas. Derecha: las mismas corridas separadas por class_weight. Los histogramas superpuestos muestran si el balanceo desplaza toda la distribución o solo sus colas.",
            ),
            _figure_block(
                "dimension_boxplots",
                figures,
                "Figura 3 — Boxplots marginales",
                "F1 macro por corrida agrupado por cada dimensión experimental. Son vistas marginales: mezclan configuraciones y semillas, por lo que describen tendencias y no efectos causales aislados.",
            ),
            _figure_block(
                "convergence_scatter",
                figures,
                "Figura 4 — Calidad vs convergencia",
                "Cada punto es una corrida. Izquierda: F1 contra el tiempo acumulado hasta la mejor época. Derecha: F1 contra la mejor época. El color es class_weight; la forma del marcador es el optimizador.",
            ),
            _figure_block(
                "convergence_histograms",
                figures,
                "Figura 5 — Histogramas de convergencia",
                "Distribución de la mejor época (menor pérdida de validación) y del tiempo acumulado necesario para alcanzarla. Las colas derechas largas indican configuraciones que siguen mejorando hasta el límite de épocas.",
            ),
            _figure_block(
                "heatmap",
                figures,
                "Figura 6 — Heatmap de activación × optimizador",
                "F1 macro medio para cada celda de activación/optimizador, facetado por class_weight. Ambos paneles comparten una escala de color. Las celdas vacías indican que ninguna corrida cubre esa combinación.",
            ),
            _figure_block(
                "convergence_curves",
                figures,
                "Figura 7 — Curvas de pérdida de validación",
                "Múltiplos pequeños para las mejores configuraciones por F1 medio. La línea azul es la media entre semillas, la banda es ±1 desvío y las líneas grises son las semillas individuales. Las curvas terminan en la época de early stopping de cada corrida.",
            ),
            _figure_block(
                "forest",
                figures,
                "Figura 8 — Forest plot de pares",
                "Diferencia de F1 macro medio contra la referencia, con su intervalo de confianza de Welch al 95 %. Los intervalos que excluyen el cero son los que los datos pueden distinguir de la referencia con este tamaño de muestra.",
            ),
        ]
    )

    no_figures_note = ""
    if include_figures and not figures:
        no_figures_note = (
            '<div class="warning">No se generaron gráficos. Las tablas y estadísticas siguen siendo válidas; '
            "verificá que matplotlib esté instalado y que la exportación contenga corridas válidas.</div>"
        )
    elif not include_figures:
        no_figures_note = (
            '<div class="note">Los gráficos se desactivaron para esta ejecución (<code>--no-figures</code>). '
            "Las tablas y estadísticas siguientes están completas.</div>"
        )

    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(title)}</title>
<style>{_css()}</style>
</head>
<body>
<header>
  <h1>{_escape(title)}</h1>
  <p class="subtitle">Experimento <code>{_escape(result.experiment_id)}</code> · snapshot <code>{_escape(result.bundle.timestamp)}</code></p>
  <p class="meta">Generado {_escape(generated_at)} · dataset <code>{_escape(dataset)}</code> · semillas {_escape(", ".join(str(seed) for seed in seeds))} · directorio de exportación <code>{_escape(result.bundle.export_dir)}</code></p>
</header>

<section class="cards">{_summary_cards(result)}</section>

<div class="note">
  <strong>Cómo leer este reporte.</strong> Cada configuración se resume a través de sus semillas. La
  referencia es <code>{_escape(result.reference_group_key)}</code> ({_escape(result.reference_label)}),
  seleccionada porque: {_escape(result.reference_reason)}. Con {len(seeds)} semillas por configuración, los
  tests inferenciales tienen baja potencia: tratá los p-valores como evidencia, no como prueba, y preferí los tamaños de efecto y
  los intervalos de confianza al decidir qué llevar a la siguiente etapa.
</div>

{observations_html}

{warnings_html}

<h2>1. Configuraciones de esta exportación</h2>
<p>
  La tabla lista las {len(result.summary)} configuraciones con al menos una corrida. <code>n válidas/total</code>
  cuenta las corridas en <code>COMPLETED</code> o <code>COMPLETED_WITH_WARNINGS</code> sobre todas las corridas registradas.
  Las columnas de convergencia se derivan del historial por época:
  <code>best_epoch</code> es la época con menor pérdida de validación,
  <code>epochs_executed</code> es donde el early stopping terminó la corrida, y
  <code>time_to_best</code> es la suma acumulada de <code>epoch_time</code> hasta la mejor época.
</p>
{summary_quality_table}
{summary_convergence_table}

<h2>2. Comparación visual</h2>
{no_figures_note}
{figures_html}

<h2>3. Comparación inferencial contra la referencia</h2>
<p>
  La referencia es la mejor configuración empírica: <strong>{_escape(result.reference_label)}</strong>.
  Las diferencias son <code>config − referencia</code>; los deltas de F1 positivos son mejores, mientras que los deltas de tiempo y
  época positivos significan que la configuración es más lenta o más tardía que la referencia.
</p>
{f1_quality_table}
{f1_significance_table}
<h3>Métricas de convergencia y costo</h3>
{convergence_tables}

<h2>4. Efectos marginales por dimensión</h2>
<p>
  Cada fila agrupa todas las corridas válidas de un nivel de una dimensión. El p-valor de Kruskal-Wallis evalúa si
  los niveles de esa dimensión difieren en F1 macro. Como las configuraciones comparten semillas e hiperparámetros,
  esto es un tamiz descriptivo, no una descomposición factorial.
</p>
{marginal_quality_table}
{marginal_convergence_table}

<h2>5. Método y definiciones</h2>
<ul class="tight">
  <li><strong>Contrato de entrada.</strong> El reporte lee los CSV <code>master</code>, <code>configs</code> y <code>epochs</code> más recientes del directorio de exportación. Nunca lee el store interno de resultados.</li>
  <li><strong>Corridas válidas.</strong> Solo las corridas <code>COMPLETED</code> y <code>COMPLETED_WITH_WARNINGS</code> alimentan las métricas. Las corridas fallidas se cuentan pero se excluyen.</li>
  <li><strong>class_weight.</strong> Este campo no está en <code>master</code>; se une desde <code>configs</code> por <code>config_id</code>. Un valor faltante significa <code>none</code>, nunca un <code>balanced</code> implícito.</li>
  <li><strong>best_epoch.</strong> Época con el mínimo <code>val_loss</code>, consistente con <code>restore_best_weights</code>.</li>
  <li><strong>time_to_best_epoch.</strong> Suma acumulada de <code>epoch_time</code> hasta <code>best_epoch</code>. Excluye el overhead de arranque del proceso, por lo que es comparable entre corridas pero queda levemente por debajo de <code>training_time</code>.</li>
  <li><strong>epochs_executed.</strong> Cantidad de épocas realmente entrenadas antes del early stopping o del tope de épocas.</li>
  <li><strong>Intervalos de confianza.</strong> Intervalo t del 95 % para la media e intervalo de Welch–Satterthwaite para las diferencias.</li>
  <li><strong>Tests.</strong> t de Welch (varianzas desiguales) y Mann-Whitney U bilateral. El tamaño de efecto es el delta de Cliff con umbrales de Romano: despreciable &lt; 0.147, pequeño &lt; 0.33, medio &lt; 0.474, grande en el resto.</li>
  <li><strong>Origen de F1.</strong> <code>f1_macro</code> es la métrica final de evaluación registrada en <code>master</code> con <code>evaluation_set = {_escape(str(result.runs['evaluation_set'].iloc[0]) if 'evaluation_set' in result.runs.columns else 'unknown')}</code>. Este módulo no la recalcula.</li>
</ul>

<h2>6. Reproducibilidad</h2>
<p>
  Reconstruí este reporte desde el mismo snapshot con:
</p>
<p><code>python -m prebenchmark analyze --experiment-id {_escape(result.experiment_id)} --export-dir {_escape(result.bundle.export_dir)}</code></p>
<p>
  El reporte es autocontenido: sin acceso a red, sin CSS externo, sin imágenes externas. Las figuras están
  embebidas como PNG en base64. Los resultados estadísticos dependen solo de los CSV exportados y de la
  selección de referencia descrita arriba.
</p>

<footer>
  Generado por <code>prebenchmark.analyze</code> · {len(result.runs)} corridas · {len(result.summary)} configuraciones ·
  referencia <code>{_escape(result.reference_group_key)}</code>.
</footer>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Punto de entrada público
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportResult:
    """Resultado estructurado de :func:`build_report`."""

    path: Path
    figures: tuple[str, ...] = field(default_factory=tuple)
    bytes_written: int = 0
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "figures": list(self.figures),
            "bytes_written": self.bytes_written,
            "warnings": list(self.warnings),
        }


def render_report(
    result: AnalysisResult,
    *,
    title: str | None = None,
    include_figures: bool = True,
    top_curves: int = 6,
    generated_at: str | None = None,
) -> tuple[str, list[str], list[str]]:
    """Renderiza el documento HTML y devuelve ``(html, figuras, advertencias)``."""
    figures: dict[str, str] = {}
    warnings: list[str] = []
    if include_figures:
        figures, figure_warnings = _render_figures(result, top_curves=top_curves)
        warnings.extend(figure_warnings)
    timestamp = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    document = _html_document(
        result,
        figures,
        title=title or f"Análisis estadístico — {result.experiment_id}",
        generated_at=timestamp,
        warnings=[*result.warnings, *warnings],
        include_figures=include_figures,
    )
    return document, list(figures), warnings


def build_report(
    result: AnalysisResult,
    out_path: str | Path,
    *,
    title: str | None = None,
    include_figures: bool = True,
    top_curves: int = 6,
    generated_at: str | None = None,
) -> ReportResult:
    """Escribe el reporte HTML autocontenido en ``out_path``."""
    document, figures, warnings = render_report(
        result,
        title=title,
        include_figures=include_figures,
        top_curves=top_curves,
        generated_at=generated_at,
    )
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return ReportResult(
        path=path,
        figures=tuple(figures),
        bytes_written=len(document.encode("utf-8")),
        warnings=tuple(warnings),
    )
