# Whether

Turn raw weather model outputs into plain-language forecasts with calibrated and interpretable uncertainty — powered by NVIDIA Earth-2, ECMWF, and Nemotron.

One command gets you an ensemble forecast, a 4-panel dashboard, probability-of-impact stats, and a Nemotron-generated narrative that says "bring an umbrella" instead of "tp ensemble mean 2.3 mm. 

## Demo

Running: ```whether forecast "San Francisco"```
![Example of forecast in SF](outputs/forecast_dashboard.png)

```
**1. TL;DR**
A cold front brings near-certain rain and decreasing temperatures to the San Francisco area from late tonight through Thursday morning.

**2. Description**
As of 11:10 PM PST on February 15, 2026, a cold front is approaching the San Francisco area (37.75, -122.50). This forecast is valid from 10:00 PM PST on February 15 to 10:00 AM PST on February 17. Expect a dominant pattern of cooling temperatures and precipitation, with the entire ensemble (7 members) indicating rain onset by 4:00 AM PST on February 16.

**3. Timeline**
* **2026-02-15 22:00 PST (start of window)**:
	+ Temperature: around 12.3°C (54.1°F)
	+ Wind: approximately 3.25 mph
	+ Precipitation: 0 mm (0 in), dry
* **2026-02-16 02:00 PST (near-current, pre-rain)**:
	+ Temperature: around 11.5°C (52.7°F), decreasing
	+ Wind: steady, around 5 mph
	+ Precipitation: 0 mm (0 in), still dry
* **2026-02-16 04:00 PST (rain onset, all members agree)**:
	+ Temperature: around 10.5°C (50.9°F), cooling
	+ Wind: picking up, around 10 mph
	+ Precipitation: first rain, amounts highly variable (range: 0.35 in to 2.46 in)
* **2026-02-16 16:00 PST (peak rain uncertainty)**:
	+ Temperature: around 9.5°C (49.1°F), continued cooling
	+ Wind: stronger, up to 20 mph (with 43% chance of reaching this speed)
	+ Precipitation: intense, but highly uncertain totals (spread: 8.71 mm)
* **2026-02-17 10:00 PST (end of window)**:
	+ Temperature: near 10.1°C (50.2°F)
	+ Wind: easing, around 15 mph
	+ Precipitation: tapering off, exact amounts uncertain

**4. Uncertainty**
* **Temperature (t2m)**: **High confidence**, with a spread of 0.40°C, representing only 0.16 sigma of the climatological uncertainty.
* **Precipitation (tp)**: **Low confidence**, highly uncertain, due to a large spread of 8.71 mm, equivalent to 4.35 sigma of the climatological uncertainty. Largest member disagreement is in precipitation totals.
* **Wind Speed (wind10m_speed)**: **High confidence**, with a spread of 2.97 mph, representing 0.59 sigma of the climatological uncertainty.

**5. Practical Advice**
* **Dress**: Wear layers for significantly cooler temperatures by Thursday morning.
* **Umbrella**: Carry one from late tonight through Thursday morning due to near-certain rain.
* **Commute Timing**: Plan for potentially hazardous road conditions and reduced visibility during the Thursday morning commute.
* **Outdoor Plans**: Postpone outdoor activities until after the forecast window (Thursday afternoon or later) to avoid rain and cooler temperatures.
```
## What It Does

```
whether forecast "San Francisco"
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
| `forecast` | ECMWF IFS ENS (default) or NVIDIA's Earth-2 DLWP/FCN/FCN3 | ~25 km global |
| `storm` | Earth-2 StormCast (HRRR emulator) | ~3 km Central US |
| `zoom` | Earth-2 CorrDiff generative downscaling | ~3 km Taiwan |

All three support `--dry-run` with spatially coherent synthetic data for demo/dev without GPU or downloads.

## Quickstart

```bash
uv sync
```

Dry-run (no API keys, no downloads, no GPU):
```bash
uv run whether forecast "NYC" --dry-run --no-narrate
```

Full run with Nemotron narrative:
```bash
export OPENROUTER_API_KEY=your_key
uv run whether forecast "NYC"
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
uv run whether forecast "Tokyo" --members 6 --steps 6

# Earth-2 FCN3 with precipitation diagnostics
uv run whether forecast "London" --provider earth2 --model fcn3 --diagnostics

# StormCast hourly storm tracking
uv run whether storm "Denver, CO" --steps 12 --members 4

# CorrDiff Taiwan downscaling snapshot
uv run whether zoom "Taipei" --samples 8 --dry-run
```
