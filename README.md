# WeatherNarrate

Turn raw AI weather model output into plain-language forecasts with calibrated and interpretable uncertainty — powered by NVIDIA Earth-2, ECMWF, and Nemotron.

One command gets you an ensemble forecast, a 4-panel dashboard, probability-of-impact stats, and a Nemotron-generated narrative that says "bring an umbrella" instead of "tp ensemble mean 2.3 mm. 

## What It Does

```
weathernarrate forecast "San Francisco"
```

1. Pulls an ensemble forecast (ECMWF IFS ENS by default, or Earth-2 DLWP/FCN/FCN3)
2. Extracts point data at your location with timezone-aware local times
3. Computes ensemble statistics, impact probabilities (PoP, freeze risk, wind thresholds), and calibrated uncertainty bands
4. Generates a 4-panel dashboard (regional map, temperature, wind, precipitation) with a "now" marker
5. Sends the structured summary to Nemotron, which returns a narrative with confidence language tied to ensemble spread

The uncertainty calibration is the key piece: ensemble spread is compared against per-variable climatological sigma references, producing confidence labels (`high`, `moderate`, `low`) that directly control narrative tone. Narrow spread gets confident language. Wide spread gets hedged language and scenario descriptions.

## Three Modes

| Command | What It Runs | Resolution |
|---|---|---|
| `forecast` | ECMWF IFS ENS (default) or Earth-2 DLWP/FCN/FCN3 | ~25 km global |
| `storm` | Earth-2 StormCast (HRRR emulator) | ~3 km Central US |
| `zoom` | Earth-2 CorrDiff generative downscaling | ~3 km Taiwan |

All three support `--dry-run` with spatially coherent synthetic data for demo/dev without GPU or downloads.

## Quickstart

```bash
uv sync
```

Dry-run (no API keys, no downloads, no GPU):
```bash
uv run weathernarrate forecast "NYC" --dry-run --no-narrate
```

Full run with Nemotron narrative:
```bash
export OPENROUTER_API_KEY=your_key
uv run weathernarrate forecast "NYC"
```

## Optional Dependencies

Install only what you need:

```bash
uv pip install ecmwf-opendata cfgrib eccodes   # ECMWF backend (default provider)
uv pip install cartopy                          # coastlines + borders on map panel
uv pip install 'earth2studio[stormcast]' pyproj # StormCast mode
uv pip install 'earth2studio[corrdiff]'         # CorrDiff mode
```

## Uncertainty Calibration

Ensemble spread is normalized against static per-variable sigma references:

| Ratio | Label | Narrative Effect |
|---|---|---|
| < 1 sigma | high confidence | Confident language, emphasize member agreement |
| 1-2 sigma | moderate confidence | Hedged language, bounded ranges |
| > 2 sigma | low confidence | Explicit disagreement, plausible scenarios |

Impact probabilities are computed directly from ensemble members:
- Precipitation: P(total >= 0.1 mm), P(>= 1 mm), P(>= 5 mm), first-rain window
- Temperature: P(t2m <= 0 C), P(t2m <= -5 C)
- Wind: P(speed >= 20/30/40 mph)

## Narrative Structure

Nemotron outputs follow a fixed 5-section format:

1. **TL;DR** — one sentence
2. **Description** — pattern overview with current local time and forecast window
3. **Timeline** — local-time progression with concrete values at 4+ checkpoints
4. **Uncertainty** — confidence labels tied to calibration bands, where members disagree most
5. **Practical Advice** — dress, umbrella, commute timing, outdoor plans

## Outputs

Each run produces:
- `forecast_dashboard.png` — 4-panel chart (or mode-specific plots for storm/zoom)
- `*_summary.json` — structured data with all ensemble stats and probabilities
- `*.zarr` — raw model output dataset

Default output directory: `./outputs`

## Examples

```bash
# ECMWF ensemble, 6 members, 36h horizon
uv run weathernarrate forecast "Tokyo" --members 6 --steps 6

# Earth-2 FCN3 with precipitation diagnostics
uv run weathernarrate forecast "London" --provider earth2 --model fcn3 --diagnostics

# StormCast hourly storm tracking
uv run weathernarrate storm "Denver, CO" --steps 12 --members 4

# CorrDiff Taiwan downscaling snapshot
uv run weathernarrate zoom "Taipei" --samples 8 --dry-run
```
