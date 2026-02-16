# WeatherNarrate

Weather forecasting CLI that combines ensemble weather outputs with Nemotron narratives.

`weathernarrate` supports three modes (`forecast`, `storm`, `zoom`) and produces:
- structured local summary with uncertainty stats
- optional Nemotron narrative
- charts + JSON artifacts

## Current Implementation Status

| Command | Backend | Status |
|---|---|---|
| `forecast` | ECMWF IFS ENS (default) or Earth-2 DLWP/FCN/FCN3 | Working |
| `storm` | Earth-2 StormCast | Working (`--dry-run` available; real run requires heavy deps/GPU) |
| `zoom` | CorrDiff Taiwan | `--dry-run` only (real inference scaffold not yet wired) |

Alias entry points are installed:
- `weatherforecast`
- `weatherstorm`
- `weatherzoom`

## Forecast Behavior (Important)

- Default provider: `ecmwf`
- Default members: `6`
- Default steps: `6` (6-hour spacing, 36h horizon)
- Timeline window includes:
  - one forecast point immediately before "now" (context)
  - all forecast points from "now" through horizon
- Forecast dashboard includes a dashed vertical "now" line on timeline panels.
- Map panel is location-centered regional context (~3000 km radius), not full-globe.

## Uncertainty Calibration

Narrative confidence language is tied to spread/sigma ratio bands:
- `< 1 sigma`: `high confidence`
- `1-2 sigma`: `moderate confidence, notable uncertainty`
- `> 2 sigma`: `low confidence, highly uncertain`

Current sigma references are static per variable (not dynamically estimated from local climatology).

## Install

```bash
uv sync
```

Optional dependencies:
- ECMWF forecast backend:
```bash
uv pip install ecmwf-opendata cfgrib eccodes
```
- Map boundaries/coastlines on forecast panel:
```bash
uv pip install cartopy
```
- StormCast mode:
```bash
uv pip install 'earth2studio[stormcast]' pyproj
```
- CorrDiff mode (still scaffolded for real runs):
```bash
uv pip install 'earth2studio[corrdiff]'
```

## Usage

Forecast (default ECMWF):
```bash
uv run weathernarrate forecast "NYC"
```

Forecast without narrative:
```bash
uv run weathernarrate forecast "NYC" --no-narrate
```

Forecast dry-run (no real model/data download):
```bash
uv run weathernarrate forecast "NYC" --dry-run --no-narrate
```

Earth-2 forecast backend:
```bash
uv run weathernarrate forecast "NYC" --provider earth2 --model dlwp
```

StormCast dry-run:
```bash
uv run weathernarrate storm "Denver, CO" --dry-run --no-narrate
```

CorrDiff dry-run:
```bash
uv run weathernarrate zoom "Taipei" --dry-run --no-narrate
```

## Nemotron Narrative

Set API key:
```bash
export OPENROUTER_API_KEY=your_key
```

Then run any command without `--no-narrate`.

Narrative prompt currently enforces sections in this order:
1. TL;DR
2. Description
3. Timeline
4. Uncertainty
5. Practical Advice

## Outputs

Per run, the CLI writes:
- model output dataset (`.zarr`)
- structured summary JSON (`*_summary.json`)
- charts (forecast dashboard or mode-specific plots)

Default output directory: `./outputs`
