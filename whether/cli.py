from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from .charts import generate_charts
from .extract import extract_at_location, trim_extracted_to_future
from .forecast import run_corrdiff, run_ecmwf_ensemble, run_global_ensemble, run_stormcast
from .geocode import resolve_lat_lon, validate_corrdiff_domain, validate_stormcast_domain
from .narrate import build_structured_summary, generate_narrative

console = Console()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lat", type=float, help="Latitude")
    parser.add_argument("--lon", type=float, help="Longitude")
    parser.add_argument("--no-narrate", action="store_true", help="Skip Nemotron narrative generation")
    parser.add_argument("--output", default="./outputs", help="Output directory for charts and model outputs")
    parser.add_argument("--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic demo data (no model inference)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="whether", description="Earth-2 + Nemotron Weather Narrative CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_forecast = sub.add_parser("forecast", help="Global ensemble forecast (Earth2 or ECMWF)")
    p_forecast.add_argument("location", nargs="?", help='Location string, e.g. "NYC"')
    p_forecast.add_argument(
        "--provider",
        choices=["earth2", "ecmwf"],
        default="ecmwf",
        help="Forecast backend provider",
    )
    p_forecast.add_argument(
        "--model",
        choices=["dlwp", "fcn", "fcn3"],
        default="dlwp",
        help="Earth2 model (used when --provider earth2)",
    )
    p_forecast.add_argument(
        "--steps",
        type=_positive_int,
        default=6,
        help="Future horizon in 6h steps (default: 6 = 36h ahead, future-only)",
    )
    p_forecast.add_argument("--members", type=_positive_int, default=6, help="Number of ensemble members")
    p_forecast.add_argument(
        "--diagnostics",
        action="store_true",
        help="Chain PrecipitationAFNO diagnostics (requires --model fcn or --model fcn3)",
    )
    p_forecast.add_argument(
        "--ecmwf-source",
        choices=["aws", "ecmwf", "azure", "google"],
        default="aws",
        help="ECMWF open-data source (used when --provider ecmwf)",
    )
    p_forecast.add_argument(
        "--cache-dir",
        default=".cache/whether/ecmwf",
        help="Cache directory for ECMWF GRIB files (used when --provider ecmwf)",
    )
    p_forecast.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable ECMWF download cache (used when --provider ecmwf)",
    )
    p_forecast.add_argument("--time", help="Init time (ISO); default latest cycle")
    _add_common_flags(p_forecast)

    p_storm = sub.add_parser("storm", help="StormCast regional storm forecast")
    p_storm.add_argument("location", nargs="?", help='Location string, e.g. "Denver, CO"')
    p_storm.add_argument("--steps", type=_positive_int, default=6, help="Number of 1h lead steps")
    p_storm.add_argument("--members", type=_positive_int, default=3, help="Number of ensemble members")
    p_storm.add_argument("--deterministic", action="store_true", help="Run a single deterministic sample")
    p_storm.add_argument("--time", help="Init time (ISO); default latest cycle")
    _add_common_flags(p_storm)

    p_zoom = sub.add_parser("zoom", help="CorrDiff Taiwan generative downscaling")
    p_zoom.add_argument("location", nargs="?", help='Location string, e.g. "Taipei"')
    p_zoom.add_argument("--samples", type=_positive_int, default=4, help="Number of CorrDiff samples")
    p_zoom.add_argument("--time", help="Snapshot time (ISO); default latest cycle")
    _add_common_flags(p_zoom)

    return parser


def _resolve_target(args: argparse.Namespace) -> tuple[float, float, str]:
    return resolve_lat_lon(getattr(args, "location", None), args.lat, args.lon)


def _print_warnings(warnings: list[str]) -> None:
    for warning in warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")


def _print_structured(extracted: dict[str, Any]) -> None:
    summary = build_structured_summary(extracted)
    console.print(Panel(summary, title="Structured Forecast Summary", expand=False))


def _write_json_summary(extracted: dict[str, Any], output_dir: Path, mode: str) -> Path:
    path = output_dir / f"{mode}_summary.json"
    path.write_text(json.dumps(extracted, indent=2), encoding="utf-8")
    return path


def _write_narrative_text(narrative: str, output_dir: Path, mode: str) -> Path:
    path = output_dir / f"{mode}_narrative.txt"
    path.write_text(narrative, encoding="utf-8")
    return path


def _run_forecast(args: argparse.Namespace) -> int:
    lat, lon, label = _resolve_target(args)
    output_dir = Path(args.output)
    now_utc = datetime.now(UTC)
    future_hours = max(int(args.steps), 1) * 6

    console.print(f"Location: {label} -> ({lat:.3f}, {lon:.3f})")
    if args.provider == "ecmwf":
        if args.model != "dlwp":
            console.print("[yellow]Warning:[/yellow] --model is ignored when --provider ecmwf.")
        if args.diagnostics:
            console.print("[yellow]Warning:[/yellow] --diagnostics is ignored for --provider ecmwf.")
        run = run_ecmwf_ensemble(
            lat=lat,
            lon=lon,
            output_dir=output_dir,
            steps=args.steps,
            members=args.members,
            time=args.time,
            dry_run=args.dry_run,
            source=args.ecmwf_source,
            cache_dir=Path(args.cache_dir),
            use_cache=(not args.no_cache),
            now_utc=now_utc,
            future_hours=future_hours,
        )
    else:
        run = run_global_ensemble(
            lat=lat,
            lon=lon,
            output_dir=output_dir,
            model_name=args.model,
            steps=args.steps,
            members=args.members,
            diagnostics=args.diagnostics,
            time=args.time,
            dry_run=args.dry_run,
            now_utc=now_utc,
            future_hours=future_hours,
        )
    _print_warnings(run.warnings)

    extracted = extract_at_location(run.dataset, lat, lon, mode="forecast")
    extracted = trim_extracted_to_future(extracted, now_utc=now_utc, future_hours=future_hours)
    charts = generate_charts(extracted, output_dir, mode="forecast", dataset=run.dataset, lat=lat, lon=lon)
    json_path = _write_json_summary(extracted, output_dir, "forecast")

    _print_structured(extracted)

    if args.no_narrate:
        console.print("Narration skipped (--no-narrate).")
    else:
        narrative, err = generate_narrative(
            extracted,
            mode="forecast",
            model_name=run.model_name,
            api_key=args.api_key,
            provider=args.provider,
        )
        if err:
            console.print(f"[yellow]Nemotron fallback:[/yellow] {err}")
        else:
            narrative_text = narrative or ""
            console.print(Panel(narrative_text, title="Nemotron Narrative", expand=False))
            narrative_path = _write_narrative_text(narrative_text, output_dir, "forecast")
            console.print(f"Narrative TXT: {narrative_path}")

    console.print(f"Model output: {run.output_path}")
    console.print(f"Structured JSON: {json_path}")
    for p in charts:
        console.print(f"Chart: {p}")
    return 0


def _run_storm(args: argparse.Namespace) -> int:
    lat, lon, label = _resolve_target(args)
    validate_stormcast_domain(lat, lon)

    output_dir = Path(args.output)
    console.print(f"Location: {label} -> ({lat:.3f}, {lon:.3f})")

    run = run_stormcast(
        lat=lat,
        lon=lon,
        output_dir=output_dir,
        steps=args.steps,
        members=args.members,
        deterministic_mode=args.deterministic,
        time=args.time,
        dry_run=args.dry_run,
    )
    _print_warnings(run.warnings)

    extracted = extract_at_location(run.dataset, lat, lon, mode="storm")
    charts = generate_charts(extracted, output_dir, mode="storm")
    json_path = _write_json_summary(extracted, output_dir, "storm")

    _print_structured(extracted)

    if args.no_narrate:
        console.print("Narration skipped (--no-narrate).")
    else:
        narrative, err = generate_narrative(
            extracted,
            mode="storm",
            model_name="stormcast",
            api_key=args.api_key,
        )
        if err:
            console.print(f"[yellow]Nemotron fallback:[/yellow] {err}")
        else:
            narrative_text = narrative or ""
            console.print(Panel(narrative_text, title="Nemotron Narrative", expand=False))
            narrative_path = _write_narrative_text(narrative_text, output_dir, "storm")
            console.print(f"Narrative TXT: {narrative_path}")

    console.print(f"Model output: {run.output_path}")
    console.print(f"Structured JSON: {json_path}")
    for p in charts:
        console.print(f"Chart: {p}")
    return 0


def _run_zoom(args: argparse.Namespace) -> int:
    lat, lon, label = _resolve_target(args)
    validate_corrdiff_domain(lat, lon)

    output_dir = Path(args.output)
    console.print(f"Location: {label} -> ({lat:.3f}, {lon:.3f})")

    run = run_corrdiff(
        lat=lat,
        lon=lon,
        output_dir=output_dir,
        samples=args.samples,
        time=args.time,
        dry_run=args.dry_run,
    )
    _print_warnings(run.warnings)

    extracted = extract_at_location(run.dataset, lat, lon, mode="zoom")
    charts = generate_charts(extracted, output_dir, mode="zoom")
    json_path = _write_json_summary(extracted, output_dir, "zoom")

    _print_structured(extracted)

    if args.no_narrate:
        console.print("Narration skipped (--no-narrate).")
    else:
        narrative, err = generate_narrative(
            extracted,
            mode="zoom",
            model_name="corrdiff-taiwan",
            api_key=args.api_key,
        )
        if err:
            console.print(f"[yellow]Nemotron fallback:[/yellow] {err}")
        else:
            narrative_text = narrative or ""
            console.print(Panel(narrative_text, title="Nemotron Narrative", expand=False))
            narrative_path = _write_narrative_text(narrative_text, output_dir, "zoom")
            console.print(f"Narrative TXT: {narrative_path}")

    console.print(f"Model output: {run.output_path}")
    console.print(f"Structured JSON: {json_path}")
    for p in charts:
        console.print(f"Chart: {p}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "forecast":
            return _run_forecast(args)
        if args.command == "storm":
            return _run_storm(args)
        if args.command == "zoom":
            return _run_zoom(args)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        return 1

    parser.print_help()
    return 1


def forecast_main() -> int:
    return main(["forecast", *sys.argv[1:]])


def storm_main() -> int:
    return main(["storm", *sys.argv[1:]])


def zoom_main() -> int:
    return main(["zoom", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
