"""Punto de entrada delgado de línea de comandos: ``python -m prebenchmark {subcommand}``.

La CLI conecta el pipeline existente -- no reimplementa nada.
Las dependencias pesadas u opcionales (torch mediante :mod:`prebenchmark.runner`) se
importan de forma diferida dentro del manejador ``run``, de modo que el resto de los
subcomandos permanece seguro para importar y libre de red/torch.

Todos los subcomandos aceptan ``--experiment-id`` (obligatorio) y ``--root``
(raíz del store de resultados, por defecto ``results``).

Subcomandos
----------
``generate``  registra el espacio de candidatos (opcionalmente limitado) en un experimento
``run``       ejecuta una o más configuraciones generadas
``compare``   selecciona una línea base y compara cada otra ejecución contra ella
``aggregate`` agrupa configuraciones repetidas y resume su variabilidad
``export``    escribe todas las tablas de resultados como CSV con marca de tiempo en ``results/exports``
``analyze``   genera un reporte estadístico HTML (y opcionalmente PDF) desde un snapshot exportado
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from . import compare, export, space, store
from .config import DEFAULT_DATASET, KNOWN_DATASETS, ExperimentConfig
from .identity import compute_config_id, utc_timestamp

__all__ = ["build_parser", "main"]


def _apply_class_weight(cfg: ExperimentConfig, value: str | None) -> ExperimentConfig:
    """Sustituye ``preprocessing.class_weight`` de la configuración.

    ``None`` deja la configuración tal cual (la que genera el espacio, con el
    ``class_weight`` sin activar); ``"none"`` la desactiva explícitamente y
    ``"balanced"`` activa el balanceo por frecuencias de la clase en el split de
    entrenamiento.
    """
    if value is None:
        return cfg
    resolved = None if value == "none" else value
    return replace(cfg, preprocessing=replace(cfg.preprocessing, class_weight=resolved))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prebenchmark",
        description=(
            "Experimental registration, execution, comparison, export and analysis "
            "for the neural-network benchmark."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="{generate,run,compare,aggregate,export,analyze}")

    def _with_common(sub: argparse.ArgumentParser) -> argparse.ArgumentParser:
        sub.add_argument("--experiment-id", required=True, help="experiment identifier")
        sub.add_argument("--root", default="results", help="results store root (default: results)")
        return sub

    generate = _with_common(
        subparsers.add_parser("generate", help="generate and register candidate configurations")
    )
    generate.add_argument("--limit", type=int, default=None, help="only the first N combinations")
    generate.add_argument(
        "--dataset",
        choices=KNOWN_DATASETS,
        default=DEFAULT_DATASET,
        help=(
            "registered dataset to generate configurations for "
            "(default: wine_quality; the input/output dimensions are derived from it)"
        ),
    )
    generate.add_argument(
        "--class-weight",
        choices=("none", "balanced"),
        default=None,
        help=(
            "override preprocessing.class_weight for the registered configurations "
            "('none' disables balancing, 'balanced' enables it; default: as generated)"
        ),
    )
    generate.set_defaults(handler=_cmd_generate)

    run = _with_common(subparsers.add_parser("run", help="execute generated configurations"))
    run.add_argument("--limit", type=int, default=1, help="number of configurations to run (default: 1)")
    run.add_argument("--config-id", default=None, help="run only the configuration with this config_id")
    run.add_argument(
        "--dataset",
        choices=KNOWN_DATASETS,
        default=DEFAULT_DATASET,
        help=(
            "registered dataset to train on (default: wine_quality; must match "
            "the dataset used by generate)"
        ),
    )
    run.add_argument("--max-epochs", type=int, default=None, help="override max_epochs")
    run.add_argument("--seed", type=int, default=None, help="override the random seed")
    run.add_argument(
        "--multi-seed",
        action="store_true",
        help="run each configuration once per exploratory seed set (5 fixed seeds)",
    )
    run.add_argument(
        "--earlyStop",
        dest="early_stop",
        action="store_true",
        help=(
            "enable the convergence criterion on generated configurations "
            "(ConvergenceSpec.enabled=True; unspecified fields use the documented defaults)"
        ),
    )
    run.add_argument(
        "--class-weight",
        choices=("none", "balanced"),
        default=None,
        help=(
            "override preprocessing.class_weight for the executed configurations "
            "('none' disables balancing, 'balanced' enables it; default: as generated)"
        ),
    )
    run.set_defaults(handler=_cmd_run)

    for name, help_text in (
        ("compare", "compare runs against a selected baseline"),
        ("aggregate", "aggregate repeated configurations"),
        ("export", "write result tables as CSVs"),
    ):
        sub = _with_common(subparsers.add_parser(name, help=help_text))
        sub.add_argument("--baseline-run-id", default=None, help="explicit baseline run id")
        sub.add_argument("--baseline-config-id", default=None, help="explicit baseline config id")
        sub.add_argument(
            "--documented-default",
            action="store_true",
            help="opt into the documented default baseline (ReLU + Adam + lr 0.001 + He)",
        )
        if name == "aggregate":
            sub.add_argument("--grouping", choices=("config", "config_id"), default="config")
            sub.add_argument("--min-repetitions", type=int, default=3)
        if name == "export":
            sub.add_argument("--out", default=None, help="raíz de exportaciones (por defecto: <root>/exports)")
            sub.add_argument("--no-plots", action="store_true", help="skip optional plots")

    subparsers.choices["compare"].set_defaults(handler=_cmd_compare)
    subparsers.choices["aggregate"].set_defaults(handler=_cmd_aggregate)
    subparsers.choices["export"].set_defaults(handler=_cmd_export)

    analyze = _with_common(
        subparsers.add_parser(
            "analyze",
            help="genera un reporte estadístico autocontenido desde los CSV exportados",
        )
    )
    analyze.add_argument(
        "--export-dir",
        default=None,
        help="directorio explícito del snapshot exportado (por defecto: <exports>/<experiment-id>)",
    )
    analyze.add_argument(
        "--exports",
        default=None,
        help="raíz de exportaciones (por defecto: <root>/exports)",
    )
    analyze.add_argument(
        "--out",
        default=None,
        help="raíz de reportes (por defecto: <root>/reports)",
    )
    analyze.add_argument(
        "--reference-config-id",
        default=None,
        help="group_key de referencia o config_id miembro (por defecto: mayor F1 macro medio)",
    )
    analyze.add_argument(
        "--top-curves",
        type=int,
        default=6,
        help="cantidad de configuraciones graficadas en el panel de curvas de convergencia (por defecto: 6)",
    )
    analyze.add_argument("--title", default=None, help="título del reporte (sobrescribe el valor por defecto)")
    analyze.add_argument("--no-figures", action="store_true", help="escribe solo tablas y estadísticas")
    analyze.add_argument(
        "--pdf",
        action="store_true",
        help="también genera un PDF a partir del HTML con Chrome/Edge headless",
    )
    analyze.add_argument(
        "--browser",
        default=None,
        help="ruta explícita a Chrome/Edge para --pdf (por defecto: autodetección)",
    )
    analyze.set_defaults(handler=_cmd_analyze)
    return parser


def _cmd_generate(args: argparse.Namespace) -> int:
    results = store.ResultsStore(args.root)
    configs = space.generate_configs(space.default_space(args.dataset))
    if args.limit is not None:
        configs = configs[: args.limit]
    configs = [_apply_class_weight(cfg, args.class_weight) for cfg in configs]
    counts: dict[str, int] = {}
    for cfg in configs:
        validation = space.validate_config(cfg)
        counts[validation.state] = counts.get(validation.state, 0) + 1
        results.register_config(args.experiment_id, cfg, validation)
    print(f"registered {len(configs)} configuration(s) into '{args.experiment_id}': {counts}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from . import runner  # diferido: torch solo se necesita para entrenar realmente
    from .seeds import EXPLORATORY_SEEDS

    results = store.ResultsStore(args.root)
    configs = space.generate_configs(space.default_space(args.dataset))
    if args.config_id is not None:
        configs = [cfg for cfg in configs if compute_config_id(cfg) == args.config_id]
        if not configs:
            raise ValueError(f"no generated configuration has config_id '{args.config_id}'")
    else:
        configs = configs[: args.limit]

    seeds: list[int | None]
    if args.multi_seed:
        seeds = list(EXPLORATORY_SEEDS)
    else:
        seeds = [args.seed]

    executed = 0
    for base in configs:
        if args.early_stop:
            base = replace(base, convergence=replace(base.convergence, enabled=True))
        base = _apply_class_weight(base, args.class_weight)
        for seed in seeds:
            overrides: dict[str, object] = {}
            if args.max_epochs is not None:
                overrides["max_epochs"] = args.max_epochs
            if seed is not None:
                overrides["seed"] = seed
            cfg = replace(base, **overrides) if overrides else base
            result = runner.run_experiment(cfg, experiment_id=args.experiment_id, store=results)
            executed += 1
            print(
                f"run {result['run_id']} config {result['config_id']} "
                f"seed {cfg.seed} state {result['state']} epochs {result.get('epochs_executed')}"
            )
    print(f"executed {executed} run(s) for '{args.experiment_id}'")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    results = store.ResultsStore(args.root)
    runs = results.query_runs(args.experiment_id)
    selection = compare.select_baseline(
        runs,
        baseline_run_id=args.baseline_run_id,
        baseline_config_id=args.baseline_config_id,
        use_documented_default=args.documented_default,
    )
    candidates = [meta for meta in runs if meta.get("run_id") != selection.run_id]
    table = compare.compare_to_baseline(selection, candidates)
    export.persist_baseline(results, args.experiment_id, selection)
    print(f"baseline {selection.run_id} ({selection.config_id}): {selection.rationale}")
    for row in table.rows:
        print(f"  {row.run_id} state={row.state} type={row.comparison_type} comparable={row.comparable}")
    return 0


def _cmd_aggregate(args: argparse.Namespace) -> int:
    from . import aggregate

    results = store.ResultsStore(args.root)
    runs = results.query_runs(args.experiment_id)
    groups = aggregate.aggregate_runs(
        runs, grouping=args.grouping, min_repetitions=args.min_repetitions
    )
    for group in groups:
        print(
            f"group {group.group_key} config {group.config_id} "
            f"n_valid={group.n_valid}/{group.n_total} "
            f"insufficient={group.insufficient_repetitions}"
        )
    print(f"{len(groups)} repetition group(s) for '{args.experiment_id}'")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    results = store.ResultsStore(args.root)
    result = export.export_all(
        results,
        args.experiment_id,
        args.out,
        baseline_run_id=args.baseline_run_id,
        baseline_config_id=args.baseline_config_id,
        use_documented_default=args.documented_default,
        include_plots=not args.no_plots,
    )
    print(f"exported {len(result.files)} table(s) to {result.export_dir}")
    for name, path in result.files.items():
        print(f"  {name}: {Path(path).name} ({result.row_counts.get(name, 0)} rows)")
    for warning in result.warnings:
        print(f"  advertencia: {warning}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from . import analysis, report

    if args.export_dir is not None:
        export_dir = Path(args.export_dir)
    else:
        exports_root = Path(args.exports) if args.exports is not None else Path(args.root) / "exports"
        export_dir = exports_root / args.experiment_id

    result = analysis.analyze_exports(
        export_dir,
        reference_config_id=args.reference_config_id,
    )
    reports_root = Path(args.out) if args.out is not None else Path(args.root) / "reports"
    out_path = reports_root / args.experiment_id / f"analysis_{utc_timestamp()}.html"
    built = report.build_report(
        result,
        out_path,
        title=args.title,
        include_figures=not args.no_figures,
        top_curves=args.top_curves,
    )

    print(f"analizadas {len(result.runs)} corrida(s) en {len(result.summary)} configuración(es)")
    print(f"referencia: {result.reference_group_key} ({result.reference_label})")
    print(f"reporte: {built.path} ({built.bytes_written} bytes, {len(built.figures)} figura(s))")
    if args.pdf:
        from . import pdf

        pdf_path = built.path.with_suffix(".pdf")
        try:
            generated = pdf.html_to_pdf(built.path, pdf_path, browser=args.browser)
        except (FileNotFoundError, RuntimeError) as error:
            print(f"error: no se pudo generar el PDF: {error}", file=sys.stderr)
            return 1
        print(f"pdf: {generated} ({generated.stat().st_size} bytes)")
    for warning in result.warnings:
        print(f"  nota: {warning}")
    for warning in built.warnings:
        print(f"  advertencia: {warning}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    # Los mensajes nuevos del subcomando ``analyze`` están en español; forzar
    # UTF-8 evita mojibake en consolas de Windows configuradas en cp1252.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (OSError, ValueError):  # pragma: no cover - stream no reconfigurable
                pass
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) is None:
        parser.print_help()
        return 2
    try:
        return int(args.handler(args))
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - se ejecuta mediante python -m
    raise SystemExit(main())
