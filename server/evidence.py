"""Evidencia empírica de las suites completas, derivada de ``results/exports``.

Decisiones de diseño
--------------------
* **Nada de listas fijas de "familias malas".** El ranking se calcula sobre las
  corridas reales del dataset y solo se marcan las familias del fondo de ese
  ranking. Un umbral absoluto no tendría sentido: el F1 máximo de wine_quality es
  ~0.346 y el de satimage ~0.898.
* **El modo de fallo se infiere de las épocas ejecutadas**, no se hardcodea. Una
  familia con F1 bajo y pocas épocas colapsó temprano; una con F1 bajo y épocas
  pegadas al tope no llegó a converger. Son causas opuestas y la UI las explica
  de forma distinta.
* **Los informes se sirven por API**, no exponiendo el sistema de archivos.
"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

import pandas as pd

COMPLETED_STATES = ("COMPLETED", "COMPLETED_WITH_WARNINGS")
BOTTOM_FAMILIES = 2
SLOW_EPOCH_FACTOR = 2.0
FAST_EPOCH_FACTOR = 0.5

FAILURE_LABELS: dict[str, str] = {
    "collapse": (
        "Converge en pocas épocas a una solución degenerada: la tasa de aprendizaje "
        "es demasiado alta para este problema."
    ),
    "no_converge": (
        "Necesita muchas más épocas que el resto de las familias para estabilizarse: "
        "la tasa de aprendizaje es demasiado baja y el costo por corrida se multiplica."
    ),
    "low_quality": "Rendimiento bajo sin un patrón claro de convergencia.",
}


def _latest(directory: Path, pattern: str) -> Path | None:
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob(pattern))
    return matches[-1] if matches else None


def _num(value: Any, digits: int = 4) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return round(number, digits)


def _failure_mode(epochs: float | None, median_epochs: float) -> str | None:
    """Clasifica el fallo comparando la familia con la mediana del propio dataset.

    Los umbrales absolutos no son portables entre datasets. Usar la mediana de
    épocas de las familias del mismo dataset hace que "lento" y "colapsa rápido"
    signifiquen lo mismo en wine_quality (mediana ~100) y en satimage (~112).
    """
    if epochs is None or median_epochs <= 0:
        return None
    if epochs >= SLOW_EPOCH_FACTOR * median_epochs:
        return "no_converge"
    if epochs <= FAST_EPOCH_FACTOR * median_epochs:
        return "collapse"
    return "low_quality"


def _families(runs: pd.DataFrame) -> list[dict[str, Any]]:
    required = {"optimizer", "lr", "f1_macro", "epochs_executed"}
    if not required.issubset(runs.columns) or runs.empty:
        return []

    usable = runs
    if "execution_state" in runs.columns:
        completed = runs[runs["execution_state"].isin(COMPLETED_STATES)]
        if not completed.empty:
            usable = completed

    aggregations: dict[str, tuple[str, str]] = {
        "n": ("f1_macro", "size"),
        "f1_mean": ("f1_macro", "mean"),
        "epochs_mean": ("epochs_executed", "mean"),
    }
    if "accuracy" in usable.columns:
        aggregations["accuracy_mean"] = ("accuracy", "mean")
    if "training_time" in usable.columns:
        aggregations["time_mean"] = ("training_time", "mean")

    grouped = (
        usable.groupby(["optimizer", "lr"], dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values("f1_mean", ascending=False)
        .reset_index(drop=True)
    )
    median_epochs = float(grouped["epochs_mean"].median()) if not grouped.empty else 0.0

    bottom_from = max(1, len(grouped) - BOTTOM_FAMILIES + 1)
    families: list[dict[str, Any]] = []
    for index, row in grouped.iterrows():
        rank = index + 1
        is_bottom = rank >= bottom_from
        epochs_mean = _num(row["epochs_mean"], 2)
        mode = _failure_mode(epochs_mean, median_epochs) if is_bottom else None
        families.append(
            {
                "label": f"{row['optimizer']} + lr={row['lr']}",
                "optimizer": str(row["optimizer"]),
                "lr": _num(row["lr"], 6),
                "n": int(row["n"]),
                "f1_mean": _num(row["f1_mean"]),
                "accuracy_mean": _num(row.get("accuracy_mean")),
                "epochs_mean": epochs_mean,
                "time_mean": _num(row.get("time_mean"), 2),
                "rank": rank,
                "is_bottom": bool(is_bottom),
                "failure_mode": mode,
                "failure_label": FAILURE_LABELS[mode] if mode else None,
            }
        )
    return families


def _experiment_record(
    experiment_dir: Path,
    reports_root: Path,
) -> dict[str, Any] | None:
    master_path = _latest(experiment_dir, "master_*.csv")
    if master_path is None:
        return None
    try:
        runs = pd.read_csv(master_path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    if runs.empty:
        return None

    experiment_id = experiment_dir.name
    dataset = str(runs["dataset"].dropna().iloc[0]) if "dataset" in runs.columns else experiment_id

    completed = runs
    if "execution_state" in runs.columns:
        maybe = runs[runs["execution_state"].isin(COMPLETED_STATES)]
        if not maybe.empty:
            completed = maybe

    aggregates_path = _latest(experiment_dir, "aggregates_*.csv")
    groups = 0
    if aggregates_path is not None:
        try:
            groups = len(pd.read_csv(aggregates_path))
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            groups = 0
    if groups == 0 and "config_id" in runs.columns:
        groups = int(runs["config_id"].nunique())

    class_weight_variants = 1
    configs_path = _latest(experiment_dir, "configs_*.csv")
    if configs_path is not None:
        try:
            configs = pd.read_csv(configs_path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            configs = pd.DataFrame()
        if "class_weight" in configs.columns:
            class_weight_variants = int(configs["class_weight"].fillna("none").nunique())

    total_seconds = (
        float(completed["training_time"].sum()) if "training_time" in completed.columns else 0.0
    )
    seeds = int(runs["seed"].nunique()) if "seed" in runs.columns else 0

    report_html = _latest(reports_root / experiment_id, "analysis_*.html")
    report_pdf = _latest(reports_root / experiment_id, "analysis_*.pdf")

    return {
        "experiment_id": experiment_id,
        "dataset": dataset,
        "runs": int(len(runs)),
        "groups": groups,
        "seeds": seeds,
        "class_weight_variants": class_weight_variants,
        "hours": _num(total_seconds / 3600, 2),
        "report_url": f"/api/reports/{experiment_id}" if report_html else None,
        "pdf_url": f"/api/reports/{experiment_id}/pdf" if report_pdf else None,
        "families": _families(runs),
    }


def load_experiments(
    exports_root: Path,
    reports_root: Path,
    datasets: Collection[str] | None = None,
) -> list[dict[str, Any]]:
    """Devuelve un registro por experimento exportado, con sus familias.

    ``datasets`` restringe la carga al menú curado de la aplicación. Los
    experimentos de prueba sobre datasets internos siguen en disco, pero no se
    mezclan con la comparación general.
    """
    if not exports_root.is_dir():
        return []
    selected = set(datasets) if datasets is not None else None
    records = []
    for experiment_dir in sorted(exports_root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        record = _experiment_record(experiment_dir, reports_root)
        if record is None:
            continue
        if selected is not None and record["dataset"] not in selected:
            continue
        records.append(record)
    records.sort(key=lambda item: (-item["runs"], item["experiment_id"]))
    return records


def dataset_evidence(dataset: str, experiments: list[dict[str, Any]]) -> dict[str, Any]:
    """Evidencia del dataset, priorizando el experimento más grande (la suite completa)."""
    matches = [item for item in experiments if item["dataset"] == dataset]
    if not matches:
        return {
            "dataset": dataset,
            "available": False,
            "families": [],
            "known_issues": [],
        }
    record = max(matches, key=lambda item: (item["runs"], item["experiment_id"]))
    families = record["families"]
    known_issues = [
        {
            "family": family["label"],
            "failure_mode": family["failure_mode"],
            "message": family["failure_label"],
            "evidence": {
                "f1_mean": family["f1_mean"],
                "epochs_mean": family["epochs_mean"],
                "time_mean": family["time_mean"],
                "runs": family["n"],
            },
        }
        for family in families
        if family["failure_mode"]
    ]
    return {
        "dataset": dataset,
        "available": True,
        "experiment_id": record["experiment_id"],
        "runs": record["runs"],
        "groups": record["groups"],
        "seeds": record["seeds"],
        "class_weight_variants": record["class_weight_variants"],
        "hours": record["hours"],
        "report_url": record["report_url"],
        "pdf_url": record["pdf_url"],
        "families": families,
        "known_issues": known_issues,
    }


def report_file(reports_root: Path, experiment_id: str, suffix: str) -> Path | None:
    """Ruta al informe más reciente de un experimento (``html`` o ``pdf``)."""
    if not experiment_id or ".." in experiment_id or "/" in experiment_id or "\\" in experiment_id:
        return None
    return _latest(reports_root / experiment_id, f"analysis_*.{suffix}")
