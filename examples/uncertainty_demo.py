from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

from whether.narrate import build_structured_summary, generate_narrative


def _timeline(start_utc: datetime, points: int = 4) -> list[str]:
    return [
        (start_utc + timedelta(hours=6 * i)).strftime("%Y-%m-%dT%H:%M:%S")
        for i in range(points)
    ]


def _build_case(*, spread_multiplier: float, label: str) -> dict:
    t0 = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    timeline = _timeline(t0, points=4)

    t2m_mean = [272.0, 271.0, 270.5, 270.0]
    t2m_spread = [0.8 * spread_multiplier, 0.9 * spread_multiplier, 1.0 * spread_multiplier, 1.1 * spread_multiplier]
    t2m_min = [m - s for m, s in zip(t2m_mean, t2m_spread)]
    t2m_max = [m + s for m, s in zip(t2m_mean, t2m_spread)]

    tp_mean = [0.0005, 0.0008, 0.0012, 0.0015]  # meters
    tp_spread = [0.0002 * spread_multiplier, 0.0002 * spread_multiplier, 0.0003 * spread_multiplier, 0.0004 * spread_multiplier]
    tp_min = [max(0.0, m - s) for m, s in zip(tp_mean, tp_spread)]
    tp_max = [m + s for m, s in zip(tp_mean, tp_spread)]

    u10_mean = [3.0, 3.2, 3.4, 3.0]
    u10_spread = [0.6 * spread_multiplier, 0.7 * spread_multiplier, 0.8 * spread_multiplier, 0.9 * spread_multiplier]
    u10_min = [m - s for m, s in zip(u10_mean, u10_spread)]
    u10_max = [m + s for m, s in zip(u10_mean, u10_spread)]

    return {
        "mode": "forecast",
        "member_dim": "ensemble",
        "n_members": 6,
        "timezone": "America/New_York",
        "current_time_utc": t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "forecast_start_utc": timeline[0],
        "forecast_end_utc": timeline[-1],
        "current_time_local": t0.astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "forecast_start_local": timeline[0].replace("T", " ") + " ET",
        "forecast_end_local": timeline[-1].replace("T", " ") + " ET",
        "timeline": timeline,
        "location": {
            "requested_lat": 42.37,
            "requested_lon": -71.10,
            "selected_lat": 42.37,
            "selected_lon": -71.10,
            "selection_method": "bilinear_interpolation",
            "label": label,
        },
        "variables": {
            "t2m": {
                "units": "K",
                "mean": t2m_mean,
                "spread": t2m_spread,
                "min": t2m_min,
                "max": t2m_max,
                "members": [[m] * 6 for m in t2m_mean],
            },
            "tp": {
                "units": "m",
                "mean": tp_mean,
                "spread": tp_spread,
                "min": tp_min,
                "max": tp_max,
                "members": [[m] * 6 for m in tp_mean],
            },
            "u10m": {
                "units": "m s**-1",
                "mean": u10_mean,
                "spread": u10_spread,
                "min": u10_min,
                "max": u10_max,
                "members": [[m] * 6 for m in u10_mean],
            },
            "v10m": {
                "units": "m s**-1",
                "mean": [1.8, 2.0, 2.1, 2.0],
                "spread": [0.5 * spread_multiplier, 0.6 * spread_multiplier, 0.7 * spread_multiplier, 0.8 * spread_multiplier],
                "min": [1.0, 1.0, 1.0, 0.8],
                "max": [2.6, 3.0, 3.2, 3.1],
                "members": [[m] * 6 for m in [1.8, 2.0, 2.1, 2.0]],
            },
        },
    }


def _render_case(case_name: str, extracted: dict, api_key: str) -> tuple[str, str]:
    summary = build_structured_summary(extracted)
    narrative, err = generate_narrative(
        extracted,
        mode="forecast",
        model_name="ecmwf-ifs-ens",
        provider="ecmwf",
        api_key=api_key,
    )
    if err:
        raise RuntimeError(f"{case_name} narrative generation failed: {err}")
    return summary, narrative or ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate calibrated uncertainty demo narratives.")
    parser.add_argument("--api-key", required=True, help="OpenRouter API key")
    parser.add_argument("--output", default="examples/uncertainty_demo.md", help="Output markdown path")
    args = parser.parse_args()

    narrow = _build_case(spread_multiplier=0.6, label="narrow-spread synthetic case")
    wide = _build_case(spread_multiplier=2.4, label="wide-spread synthetic case")

    narrow_summary, narrow_narrative = _render_case("narrow", narrow, args.api_key)
    wide_summary, wide_narrative = _render_case("wide", wide, args.api_key)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(
            [
                "# Uncertainty Calibration Demo",
                "",
                "This demo uses two synthetic summaries with the same weather signal but different ensemble spread.",
                "The narrative should shift from high-confidence wording (narrow spread) to low-confidence wording (wide spread).",
                "",
                "## Narrow Spread Case",
                "",
                "### Structured Summary",
                "```text",
                narrow_summary,
                "```",
                "",
                "### Nemotron Narrative",
                "```text",
                narrow_narrative,
                "```",
                "",
                "## Wide Spread Case",
                "",
                "### Structured Summary",
                "```text",
                wide_summary,
                "```",
                "",
                "### Nemotron Narrative",
                "```text",
                wide_narrative,
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote demo to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
