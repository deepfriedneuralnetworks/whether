# WeatherNarrate

Earth-2 + Nemotron weather narrative CLI.

`weathernarrate` runs NVIDIA Earth-2 weather model workflows, extracts local uncertainty statistics from ensembles/samples, and generates natural-language summaries with Nemotron.

## Modes

| Subcommand | Earth-2 model | Resolution | Coverage | Time step |
|---|---|---|---|---|
| `forecast` | Earth2 (DLWP / FCN / FCN3) or ECMWF IFS ENS | ~25 km | Global | 6h |
| `storm` | StormCast ensemble | ~3 km | Central US domain | 1h |
| `zoom` | CorrDiff Taiwan | ~3 km | Taiwan | snapshot |

Alias entry points are also installed:

- `weatherforecast`
- `weatherstorm`
- `weatherzoom`

## Architecture

```
forecast / storm / zoom
  -> model inference (mode-specific)
  -> extract/interpolate local data
  -> compute mean/spread/min/max across ensemble/sample
  -> format structured summary
  -> Nemotron narrative
  -> charts + JSON output
```

## Calibrated Uncertainty Communication

Forecast summaries now include variable-level calibration bands using spread/sigma ratios:

- ratio `< 1σ` -> `high confidence`
- ratio `1-2σ` -> `moderate confidence, notable uncertainty`
- ratio `> 2σ` -> `low confidence, highly uncertain`

These bands are injected into the Nemotron prompt and structured summary so narrative wording is tied to calibrated uncertainty, not just raw spread values.

## Quickstart

1. Create environment and install:

```bash
uv sync
```

2. Dry-run global forecast (no Earth-2 inference required):

```bash
uv run weathernarrate forecast "Cambridge, MA" --dry-run --no-narrate
```
Default `forecast` settings use ECMWF open-data ensemble with 6 members and a 36-hour future window (`--steps 6`, 6-hour spacing).

2b. ECMWF ensemble forecast (same unified `forecast` command):

```bash
uv run weathernarrate forecast "Cambridge, MA" --provider ecmwf --members 10 --steps 8 --no-narrate
```

ECMWF downloads are cached by default under `.cache/weathernarrate/ecmwf`.
Use `--no-cache` to force fresh downloads, or `--cache-dir <path>` to customize location.

3. Dry-run StormCast mode:

```bash
uv run weathernarrate storm "Denver, CO" --dry-run --no-narrate
```

4. Dry-run CorrDiff mode:

```bash
uv run weathernarrate zoom "Taipei" --dry-run --no-narrate
```

5. Enable Nemotron narrative:

```bash
export OPENROUTER_API_KEY=your_key
uv run weathernarrate forecast "Cambridge, MA" --dry-run
```

6. Generate the narrow-vs-wide uncertainty language demo:

```bash
export OPENROUTER_API_KEY=your_key
uv run python examples/uncertainty_demo.py --api-key "$OPENROUTER_API_KEY" --output examples/uncertainty_demo.md
```

This creates `examples/uncertainty_demo.md` with two cases and two Nemotron narratives so you can verify confidence language shifts with spread.

## Real Earth-2 installs

Install optional extras based on mode:

```bash
uv pip install 'earth2studio[dlwp]'
uv pip install 'earth2studio[fcn]'
uv pip install 'earth2studio[fcn3]'
uv pip install ecmwf-opendata cfgrib eccodes
uv pip install 'earth2studio[stormcast]' pyproj
uv pip install 'earth2studio[corrdiff]'
uv pip install cartopy  # optional: map boundaries in forecast dashboard
```

## Current status

- `forecast`: wired with Earth-2 ensemble path, ECMWF open-data ensemble path, and dry-run fallback.
- `storm`: wired with StormCast path + dry-run fallback.
- `zoom`: scaffolded and fully dry-run capable; real CorrDiff custom loop is left as a follow-up integration in `weathernarrate/forecast.py`.

## Output artifacts

Each run writes:

- Zarr model output (or synthetic output in dry-run)
- Structured JSON summary
- Mode-specific charts in `./outputs`
