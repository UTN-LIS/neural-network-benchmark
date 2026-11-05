# Handoff — estado del proyecto al 2026-10-29

Este archivo guarda el punto exacto donde quedamos para poder retomar el trabajo
sin reconstruir el contexto desde cero.

## Qué se implementó en esta etapa

### Análisis estadístico
- `prebenchmark/analysis.py`
  - Lee el snapshot exportado más reciente: `master_*.csv`, `configs_*.csv`,
    `epochs_*.csv`; `aggregates_*.csv` es opcional.
  - Nunca lee el store interno de corridas.
  - Une `class_weight` desde `configs` por `config_id`.
  - Deriva convergencia desde `epochs`:
    - `best_epoch`: época con menor `val_loss`.
    - `time_to_best_epoch`: suma acumulada de `epoch_time` hasta `best_epoch`.
    - `epochs_executed`: épocas entrenadas hasta el early stopping o el tope.
  - Agrupa por `group_key` (configuración sin semilla), no por `config_id`.
  - Calcula media, desvío, IC 95 %, Welch, Mann-Whitney, delta de Cliff y
    Kruskal-Wallis.
  - Las corridas `FAILED` se cuentan pero no alimentan las métricas.

- `prebenchmark/report.py`
  - Genera un HTML autocontenido: CSS inline y 8 figuras PNG en base64.
  - Tablas divididas para impresión: 1a/1b, 2a/2b y 3a/3b.
  - Ranking y forest plot en dos paneles cuando hay muchas configuraciones.
  - CSS de impresión A4, encabezados de tabla repetidos y figuras con su
    epígrafe.
  - Si una figura falla, el HTML se genera igual.

- `prebenchmark/pdf.py`
  - Convierte el HTML a PDF con Chrome o Edge headless
    (`--print-to-pdf`).
  - Autodetecta el navegador en Windows, macOS y Linux.
  - Permite una ruta explícita con `--browser`.
  - Es opcional: si no hay navegador, el HTML sigue siendo válido.

### CLI
- Nuevo subcomando `analyze`:
  ```bash
  python -m prebenchmark analyze --experiment-id exp-sweep-01
  python -m prebenchmark analyze --experiment-id exp-sweep-01 --pdf
  ```
- Opciones principales:
  - `--export-dir`
  - `--exports`
  - `--out`
  - `--reference-config-id`
  - `--top-curves`
  - `--title`
  - `--no-figures`
  - `--pdf`
  - `--browser`
- El docstring de `cli.py` ahora lista los seis subcomandos:
  `generate`, `run`, `compare`, `aggregate`, `export`, `analyze`.
- Se agregó un test de integridad que falla si el docstring deja de listar un
  subcomando registrado en el parser.

### Tests
- `tests/test_analysis.py`
- `tests/test_report.py`
- `tests/test_pdf.py`
- `tests/test_cli.py` (incluye el test de integridad del docstring)

Última corrida completa: **245 tests pasan**.

## Reporte estadístico generado

Último reporte de `exp-sweep-01`:

```text
results/reports/exp-sweep-01/analysis_20261029T214944Z.html
results/reports/exp-sweep-01/analysis_20261029T214944Z.pdf
```

- 72 configuraciones, 360 corridas, 5 semillas.
- 8 figuras.
- PDF A4 de 19 páginas.
- Referencia empírica: `grp-v1-7d12b4d9b658`
  (`tanh | adam | lr=0.01 | he | cw=balanced`).

No se agregaron conclusiones estadísticas ni análisis de gráficos al documento
de explicación; eso queda pendiente para la próxima etapa.

## Documento de explicación actualizado

Archivos:

```text
Explicacion-Proyecto-Prebenchmark.html
Explicacion-Proyecto-Prebenchmark.pdf
```

Cambios realizados:
- Pipeline de 5 a 6 etapas: `generate → run → compare → aggregate → export → analyze`.
- Nuevos módulos:
  - 4.15 `analysis.py`
  - 4.16 `report.py`
  - 4.17 `pdf.py`
- Mapa de capas actualizado.
- Sección 9.1 con la estructura actual:
  - `results/experiments/` (corridas)
  - `results/exports/` (CSV derivados)
  - `results/reports/` (HTML/PDF derivados)
- Nueva sección 9.4 sobre reportes; la antigua 9.4 pasó a 9.5.
- Nueva sección 10.7 `analyze`; se reindexaron 10.8, 10.9 y 10.10.
- Flujos de ejemplo, errores frecuentes y referencia rápida actualizados con
  `analyze`.
- Conteo de módulos: 16 → 17.
- PDF regenerado: 21 páginas, ~1.24 MB.

## Decisiones técnicas que no hay que olvidar

1. **`config_id` incluye la semilla.**
   - No sirve para agrupar repeticiones.
   - La unidad correcta es `group_key` (configuración sin semilla).
   - En `exp-sweep-01`: 72 grupos = 36 combinaciones × 2 `class_weight` × 5
     semillas.

2. **`class_weight` no está en `master_*.csv`.**
   - Se une desde `configs_*.csv` por `config_id`.
   - Un valor faltante significa `none`, nunca `balanced` implícito.

3. **Convergencia derivada, no inventada.**
   - `best_epoch` = argmin `val_loss`.
   - `time_to_best_epoch` = suma de `epoch_time` hasta `best_epoch`.
   - `epochs_executed` = época de early stopping / tope.

4. **El análisis consume exports, no el store.**
   - Esto permite reproducir el reporte sin volver a entrenar.
   - Requiere `master`, `configs` y `epochs`.

5. **La referencia es empírica por defecto.**
   - Si no se pasa `--reference-config-id`, se usa la mejor F1 media.
   - El reporte documenta esa elección.

6. **El PDF no es una dependencia de Python.**
   - Usa Chrome/Edge headless.
   - Si no hay navegador, el HTML se genera igual y la CLI devuelve código 1.

## Estructura de resultados y `.gitignore`

Estado actual de `.gitignore` para resultados:

```gitignore
results/experiments/
```

Consecuencias:
- `results/experiments/` queda ignorado (corridas crudas).
- `results/exports/` queda versionable.
- `results/reports/` queda versionable.

Tamaños aproximados:
- `results/exports/exp-sweep-01`: ~8.5 MB (el `epochs_*.csv` pesa ~7.9 MB).
- `results/reports/exp-sweep-01`: ~3.4 MB (HTML + PDF).

Si se quiere una regla más robusta para el futuro:

```gitignore
results/*
!results/exports/
!results/reports/
```

Eso ignora cualquier carpeta nueva dentro de `results/` salvo `exports` y
`reports`.

## Estado de Git al cierre

Último `git status` mostró solamente:

```text
M Explicacion-Proyecto-Prebenchmark.html
M Explicacion-Proyecto-Prebenchmark.pdf
```

El código de `analysis.py`, `report.py`, `pdf.py`, los tests y el cambio de
`.gitignore` ya no figuraban como pendientes; probablemente ya fueron
commiteados. Verificar con:

```bash
git log --oneline -5
git status
```

## Cómo regenerar todo

### Reporte estadístico

```bash
python -m prebenchmark analyze --experiment-id exp-sweep-01 --pdf
```

Si las corridas están en `results/experiments`:

```bash
python -m prebenchmark analyze --experiment-id exp-sweep-01 \
  --root results/experiments --pdf
```

### PDF del documento de explicación

```powershell
.\generar-pdf.ps1
```

O con rutas explícitas:

```powershell
.\generar-pdf.ps1 `
  -Html "Explicacion-Proyecto-Prebenchmark.html" `
  -Pdf  "Explicacion-Proyecto-Prebenchmark.pdf"
```

### Tests

```bash
python -m pytest -q
```

## Engram

- El handoff quedó guardado como observación **#281** (`architecture`) en el proyecto
  `neural-network-benchmark`.
- OpenCode tiene el **plugin** de Engram cargado, pero **no el MCP server** registrado:
  `opencode mcp list` devuelve `No MCP servers configured`.
- Por eso las tools `mem_save`, `mem_search`, etc. no están disponibles en la sesión,
  aunque el binario y la base de Engram funcionen.
- Para habilitarlas:

  ```powershell
  opencode mcp add engram --global -- "C:\Users\agust\AppData\Local\engram\bin\engram.exe" mcp --tools=agent
  ```

  Después hay que reiniciar OpenCode para que cargue el MCP server.

## Próximos pasos sugeridos

1. Revisar los resultados estadísticos y los gráficos del reporte de
   `exp-sweep-01`.
2. Decidir si se agrega una sección de resultados/análisis al documento de
   explicación (se dejó afuera a propósito en esta etapa).
3. Commitear el HTML y el PDF de explicación actualizados.
4. Decidir si se versionan `results/exports` y `results/reports`; si sí,
   revisar el peso del `epochs_*.csv`.
5. Si se quiere, mover este handoff a Engram cuando la herramienta esté
   disponible.

## Archivos relevantes

```text
prebenchmark/analysis.py
prebenchmark/report.py
prebenchmark/pdf.py
prebenchmark/cli.py
tests/test_analysis.py
tests/test_report.py
tests/test_pdf.py
tests/test_cli.py
Explicacion-Proyecto-Prebenchmark.html
Explicacion-Proyecto-Prebenchmark.pdf
results/reports/exp-sweep-01/analysis_20261029T214944Z.html
results/reports/exp-sweep-01/analysis_20261029T214944Z.pdf
results/exports/exp-sweep-01/
results/experiments/exp-sweep-01/
```

## Actualización 2026-11-05 — registro de datasets y `exp-sweep-02-satimage`

### Motivación

`exp-sweep-01` (wine_quality) tenía un techo estructural de F1 macro: la clase 9
tiene 5 ejemplos en todo el dataset y la clase 3 tiene 20 (ratio 439,6x). Un solo
error en el test de la clase 9 la lleva a F1=0 y resta 1/7 al macro. El mejor
promedio por configuración fue 0,303. El balanceo de clases mejoró 29/36
configuraciones, pero solo +0,008 en media y empeoró 7 (hasta −0,122): no fue un
salto significativo.

### Cambios de código

- `prebenchmark/data.py`: registro `DatasetDefinition` / `DatasetInfo` con
  `wine_quality`, `satimage`, `vehicle`, `waveform_5000`, `segment` y `digits`.
  La carga, el target, la caché CSV y la huella SHA-256 son genéricos.
  `dataset_base_config()` deriva `input_dim`/`output_dim` del dataset real.
- `prebenchmark/config.py`: `DEFAULT_DATASET` y `KNOWN_DATASETS`.
- `prebenchmark/space.py`: `default_space(dataset)` usa el dataset pedido;
  `wine_quality` conserva los valores históricos sin tocar red ni disco.
- `prebenchmark/cli.py`: `--dataset` en `generate` y `run`.
- `prebenchmark/compare.py`: `--documented-default` ahora selecciona por las
  cuatro dimensiones (ReLU + Adam + lr 0.001 + He), no por `config_id` completo.
  Esto lo hace agnóstico al dataset, la arquitectura y el early stopping.
- Tests: 252 pasan (el test probabilístico de unicidad de `run_id` puede fallar
  esporádicamente; es previo a este cambio).

### Dataset elegido: satimage

Statlog (Landsat Satellite), OpenML v1. 6.430 muestras, 36 features (4 bandas ×
9 píxeles de un vecindario 3×3), 6 clases (1,2,3,4,5,7; la clase 6 no tiene
ejemplos). La columna objetivo normalizada es `target` (0..5). Arquitectura:
36 → 32 → 16 → 6, 1.814 parámetros. Las capas ocultas siguen fijas en [32, 16].

### Resultados de `exp-sweep-02-satimage`

Protocolo: 36 configuraciones × 5 semillas, `--earlyStop`, `class_weight=none`,
máx. 500 épocas. 180 corridas, 1,07 h.

- F1 macro: media 0,845; rango 0,653–0,898.
- Mejor: `tanh | sgd | lr=0.1 | xavier` → 0,8852 ± 0,0054.
- Segundo: `relu | adam | lr=0.001 | xavier` → 0,8841 ± 0,0128.
- Peor: `sigmoid | sgd | lr=0.001 | he` → 0,6903 ± 0,0090.
- El top 14 es una meseta estadísticamente indistinguible (p > 0,05 frente al
  mejor). Las diferencias significativas aparecen contra las familias malas:
  `adam + lr=0.1` (0,73–0,80) y `sgd + lr=0.001` (0,69–0,85).
- `adam + lr=0.001` es consistentemente fuerte; `sgd + lr=0.1` también.
  `init` casi no tiene efecto marginal (he 0,8457 vs xavier 0,8452).
- Reporte final:
  `results/reports/exp-sweep-02-satimage/analysis_20261105T203942Z.html`
  y `.pdf` (8 figuras).
- Export final: `results/exports/exp-sweep-02-satimage/` con un único snapshot
  (`*_20261105T203928Z.*`): master 180 filas, configs 180 filas, aggregates 36.

### Reproducir

```bash
python -m prebenchmark generate --experiment-id exp-sweep-02-satimage --dataset satimage --class-weight none
python -m prebenchmark run --experiment-id exp-sweep-02-satimage --dataset satimage --limit 36 --seed 42 --earlyStop --class-weight none
python -m prebenchmark run --experiment-id exp-sweep-02-satimage --dataset satimage --limit 36 --seed 123 --earlyStop --class-weight none
python -m prebenchmark run --experiment-id exp-sweep-02-satimage --dataset satimage --limit 36 --seed 7 --earlyStop --class-weight none
python -m prebenchmark run --experiment-id exp-sweep-02-satimage --dataset satimage --limit 36 --seed 2025 --earlyStop --class-weight none
python -m prebenchmark run --experiment-id exp-sweep-02-satimage --dataset satimage --limit 36 --seed 999 --earlyStop --class-weight none
python -m prebenchmark compare --experiment-id exp-sweep-02-satimage --documented-default
python -m prebenchmark aggregate --experiment-id exp-sweep-02-satimage --grouping config
python -m prebenchmark export --experiment-id exp-sweep-02-satimage
python -m prebenchmark analyze --experiment-id exp-sweep-02-satimage --pdf
```

### Nota

`generate` registró 36 candidatos sin `--earlyStop`; `run --earlyStop` los mutó y
creó 36 `config_id` distintos. Esos candidatos nunca se ejecutaron y agregaban
ruido al export (216 filas de configs y una advertencia en el informe). Se podó
`configs.jsonl` para dejar solo las 180 configuraciones ejecutadas (36 × 5
semillas); el archivo original quedó como
`results/exp-sweep-02-satimage/configs.jsonl.bak-20261105`.

