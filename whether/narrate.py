from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import requests

from .prompts import build_system_prompt
from .timefmt import fmt_time_label
from .uncertainty import calibrate_uncertainty_band
from .units import is_temperature_variable, mm_to_inches, precip_to_mm, to_celsius_if_needed, wind_to_mph

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = os.getenv("WHETHER_NEMOTRON_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")


def _c_to_f(c: float) -> float:
    return (c * 9.0 / 5.0) + 32.0


def _to_utc_datetime(value: str | np.datetime64) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except ValueError:
        dt64 = np.datetime64(text).astype("datetime64[s]")
        epoch = int(dt64.astype(np.int64))
        return datetime.fromtimestamp(epoch, tz=UTC)


def _to_local_time_label(value: str, timezone_name: str) -> str:
    try:
        zone = ZoneInfo(timezone_name)
    except Exception:  # noqa: BLE001
        zone = ZoneInfo("UTC")
    dt_utc = _to_utc_datetime(value)
    return dt_utc.astimezone(zone).strftime("%Y-%m-%d %H:%M %Z")


def _format_probability(probability: float) -> str:
    return f"{probability * 100.0:.0f}%"


def _as_member_matrix(stats: dict[str, list[float]]) -> np.ndarray | None:
    arr = np.asarray(stats.get("members", []), dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return None
    return arr


def _precip_window_stats(tp_stats: dict[str, list[float]]) -> tuple[np.ndarray, np.ndarray] | None:
    tp_members = _as_member_matrix(tp_stats)
    if tp_members is None:
        return None
    tp_mm, _ = precip_to_mm(tp_members, units=str(tp_stats.get("units", "")))
    if tp_mm.ndim != 2 or tp_mm.size == 0:
        return None

    if tp_mm.shape[0] > 1:
        monotonic_fraction = float(np.mean(np.diff(tp_mm, axis=0) >= -1e-6))
    else:
        monotonic_fraction = 0.0
    cumulative_like = tp_mm.shape[0] > 1 and monotonic_fraction >= 0.9

    if cumulative_like:
        step_mm = np.diff(tp_mm, axis=0, prepend=tp_mm[0:1, :])
        step_mm[0, :] = 0.0
        total_mm = np.clip(tp_mm[-1, :] - tp_mm[0, :], a_min=0.0, a_max=None)
    else:
        step_mm = np.clip(tp_mm, a_min=0.0, a_max=None)
        total_mm = np.nansum(step_mm, axis=0)
    return total_mm, step_mm


def _forecast_probability_lines(
    vars_: dict[str, dict[str, list[float]]],
    timeline: list[str],
    timezone_name: str,
) -> list[str]:
    lines: list[str] = []

    tp_stats = vars_.get("tp")
    if tp_stats is not None:
        precip = _precip_window_stats(tp_stats)
        if precip is not None:
            total_mm, step_mm = precip
            pop_01 = float(np.mean(total_mm >= 0.1))
            pop_1 = float(np.mean(total_mm >= 1.0))
            pop_5 = float(np.mean(total_mm >= 5.0))
            lines.append(
                "Precipitation probabilities (window total): "
                f"P(tp>=0.1 mm)={_format_probability(pop_01)}, "
                f"P(tp>=1 mm)={_format_probability(pop_1)}, "
                f"P(tp>=5 mm)={_format_probability(pop_5)}."
            )

            first_hits: list[int] = []
            for member_idx in range(step_mm.shape[1]):
                hit = np.where(step_mm[:, member_idx] >= 0.1)[0]
                if hit.size:
                    first_hits.append(int(hit[0]))
            if first_hits and timeline:
                first_idx = int(min(first_hits))
                last_idx = int(max(first_hits))
                first_local = _to_local_time_label(timeline[first_idx], timezone_name)
                last_local = _to_local_time_label(timeline[last_idx], timezone_name)
                lines.append(
                    "First precipitation window (tp>=0.1 mm step): "
                    f"{first_local} to {last_local} ({len(first_hits)}/{step_mm.shape[1]} members)."
                )
            else:
                lines.append("First precipitation window (tp>=0.1 mm step): no member reaches threshold.")

    sf_stats = vars_.get("sf")
    if sf_stats is not None:
        snowfall = _precip_window_stats(sf_stats)
        if snowfall is not None:
            snow_total_mm, snow_step_mm = snowfall
            snow_total_in = mm_to_inches(snow_total_mm)
            snow_depth_est_in = snow_total_in * 10.0  # simple 10:1 SLR estimate
            lines.append(
                "Snowfall (from sf, liquid-equivalent): "
                f"range {float(np.min(snow_total_mm)):.2f} to {float(np.max(snow_total_mm)):.2f} mm SWE "
                f"({float(np.min(snow_total_in)):.2f} to {float(np.max(snow_total_in)):.2f} in SWE); "
                f"estimated snow depth ~{float(np.min(snow_depth_est_in)):.1f} to {float(np.max(snow_depth_est_in)):.1f} in (10:1 SLR)."
            )

            p2 = float(np.mean(snow_depth_est_in >= 2.0))
            p6 = float(np.mean(snow_depth_est_in >= 6.0))
            p12 = float(np.mean(snow_depth_est_in >= 12.0))
            lines.append(
                "Snow accumulation probabilities (estimated, 10:1 SLR): "
                f"P(snow>=2 in)={_format_probability(p2)}, "
                f"P(snow>=6 in)={_format_probability(p6)}, "
                f"P(snow>=12 in)={_format_probability(p12)}."
            )

            snow_hits: list[int] = []
            for member_idx in range(snow_step_mm.shape[1]):
                hit = np.where(snow_step_mm[:, member_idx] >= 0.1)[0]
                if hit.size:
                    snow_hits.append(int(hit[0]))
            if snow_hits and timeline:
                first_idx = int(min(snow_hits))
                last_idx = int(max(snow_hits))
                first_local = _to_local_time_label(timeline[first_idx], timezone_name)
                last_local = _to_local_time_label(timeline[last_idx], timezone_name)
                lines.append(
                    "First snow window (sf>=0.1 mm SWE step): "
                    f"{first_local} to {last_local} ({len(snow_hits)}/{snow_step_mm.shape[1]} members)."
                )
            else:
                lines.append("First snow window (sf>=0.1 mm SWE step): no member reaches threshold.")

    t2m_stats = vars_.get("t2m")
    if t2m_stats is not None:
        t2m_members = _as_member_matrix(t2m_stats)
        if t2m_members is not None:
            t2m_c = to_celsius_if_needed(
                t2m_members,
                units=str(t2m_stats.get("units", "")),
                var_name="t2m",
            )
            member_min_c = np.nanmin(t2m_c, axis=0)
            freeze_0 = float(np.mean(member_min_c <= 0.0))
            freeze_5 = float(np.mean(member_min_c <= -5.0))
            lines.append(
                "Freeze risk (any time in window): "
                f"P(t2m<=0C)={_format_probability(freeze_0)}, "
                f"P(t2m<=-5C)={_format_probability(freeze_5)}."
            )

    u_stats = vars_.get("u10m")
    v_stats = vars_.get("v10m")
    if u_stats is not None and v_stats is not None:
        u_members = _as_member_matrix(u_stats)
        v_members = _as_member_matrix(v_stats)
        if u_members is not None and v_members is not None and u_members.shape == v_members.shape:
            u_mph, _ = wind_to_mph(u_members, units=str(u_stats.get("units", "")))
            v_mph, _ = wind_to_mph(v_members, units=str(v_stats.get("units", "")))
            speed_mph = np.sqrt((u_mph**2) + (v_mph**2))
            member_peak = np.nanmax(speed_mph, axis=0)
            p20 = float(np.mean(member_peak >= 20.0))
            p30 = float(np.mean(member_peak >= 30.0))
            p40 = float(np.mean(member_peak >= 40.0))
            lines.append(
                "Wind impact risk (any time in window): "
                f"P(speed>=20 mph)={_format_probability(p20)}, "
                f"P(speed>=30 mph)={_format_probability(p30)}, "
                f"P(speed>=40 mph)={_format_probability(p40)}."
            )

    return lines


def _one_line_series(name: str, stats: dict[str, list[float]]) -> str:
    mean = [float(v) for v in stats["mean"]]
    spread = [float(v) for v in stats["spread"]]
    min_values = [float(v) for v in stats["min"]]
    max_values = [float(v) for v in stats["max"]]
    units = str(stats.get("units", ""))

    if is_temperature_variable(name):
        mean_c = to_celsius_if_needed(np.asarray(mean, dtype=float), units=units, var_name=name).tolist()
        min_c = to_celsius_if_needed(np.asarray(min_values, dtype=float), units=units, var_name=name).tolist()
        max_c = to_celsius_if_needed(np.asarray(max_values, dtype=float), units=units, var_name=name).tolist()
        avg_s = sum(spread) / max(len(spread), 1)
        if name == "t2m":
            first_c = float(mean_c[0])
            last_c = float(mean_c[-1])
            first_f = _c_to_f(first_c)
            last_f = _c_to_f(last_c)
            return (
                f"{name}: starts near {first_c:.1f} C ({first_f:.1f} F), "
                f"ends near {last_c:.1f} C ({last_f:.1f} F); "
                f"typical spread {avg_s:.2f} C."
            )
        min_c_all = min(min_c)
        max_c_all = max(max_c)
        min_f = _c_to_f(min_c_all)
        max_f = _c_to_f(max_c_all)
        return (
            f"{name}: range {min_c_all:.2f} C ({min_f:.2f} F) to "
            f"{max_c_all:.2f} C ({max_f:.2f} F); typical spread {avg_s:.2f} C."
        )

    if name in {"tp", "sf", "sd"}:
        mean_mm, precip_unit = precip_to_mm(np.asarray(mean, dtype=float), units=units)
        min_mm, _ = precip_to_mm(np.asarray(min_values, dtype=float), units=units)
        max_mm, _ = precip_to_mm(np.asarray(max_values, dtype=float), units=units)
        spread_mm, _ = precip_to_mm(np.asarray(spread, dtype=float), units=units)
        min_in = mm_to_inches(min_mm)
        max_in = mm_to_inches(max_mm)
        avg_s = float(np.mean(spread_mm)) if spread_mm.size else 0.0
        label = name
        if name == "sf":
            label = "sf (snowfall water-equivalent)"
        elif name == "sd":
            label = "sd (snow depth water-equivalent)"
        return (
            f"{label}: range {float(np.min(min_mm)):.2f} {precip_unit} ({float(np.min(min_in)):.2f} in) to "
            f"{float(np.max(max_mm)):.2f} {precip_unit} ({float(np.max(max_in)):.2f} in); typical spread {avg_s:.2f} {precip_unit}."
        )

    if name in {"u10m", "v10m"}:
        min_mph, unit = wind_to_mph(np.asarray(min_values, dtype=float), units=units)
        max_mph, _ = wind_to_mph(np.asarray(max_values, dtype=float), units=units)
        spread_mph, _ = wind_to_mph(np.asarray(spread, dtype=float), units=units)
        avg_s = float(np.mean(spread_mph)) if spread_mph.size else 0.0
        return (
            f"{name}: range {float(np.min(min_mph)):.2f} to {float(np.max(max_mph)):.2f} {unit}; "
            f"typical spread {avg_s:.2f} {unit}."
        )

    min_v = min(min_values)
    max_v = max(max_values)
    avg_s = sum(spread) / max(len(spread), 1)
    return f"{name}: range {min_v:.2f} to {max_v:.2f}; typical spread {avg_s:.2f}."


def _confidence_label_from_ratio(ratio_sigma: float) -> str:
    if ratio_sigma < 1.0:
        return "high confidence"
    if ratio_sigma <= 2.0:
        return "moderate confidence, notable uncertainty"
    return "low confidence, highly uncertain"


def _wind_speed_summary(
    u_stats: dict[str, list[float]],
    v_stats: dict[str, list[float]],
) -> tuple[str, str] | None:
    u_units = str(u_stats.get("units", ""))
    v_units = str(v_stats.get("units", ""))
    u_members_raw = np.asarray(u_stats.get("members", []), dtype=float)
    v_members_raw = np.asarray(v_stats.get("members", []), dtype=float)
    if u_members_raw.shape != v_members_raw.shape or u_members_raw.ndim != 2:
        return None

    u_mph, _ = wind_to_mph(u_members_raw, units=u_units)
    v_mph, _ = wind_to_mph(v_members_raw, units=v_units)
    wind_speed_mph = np.sqrt((u_mph**2) + (v_mph**2))
    spread_mph = np.std(wind_speed_mph, axis=1)
    min_mph = np.min(wind_speed_mph, axis=1)
    max_mph = np.max(wind_speed_mph, axis=1)

    line = (
        f"wind10m_speed: range {float(np.min(min_mph)):.2f} to {float(np.max(max_mph)):.2f} mph; "
        f"typical spread {float(np.mean(spread_mph)):.2f} mph."
    )

    sigma_ref_mph = 5.0
    ratio = float(np.mean(spread_mph)) / sigma_ref_mph if sigma_ref_mph > 0 else 0.0
    confidence = _confidence_label_from_ratio(ratio)
    calib_line = (
        f"wind10m_speed: spread {float(np.mean(spread_mph)):.2f} mph, "
        f"climatology sigma {sigma_ref_mph:.2f} mph, ratio {ratio:.2f} sigma -> {confidence}"
    )
    return line, calib_line


def _build_step_table(extracted: dict[str, Any], timezone_name: str) -> str:
    """Build a per-timestep data table for the Nemotron prompt."""
    timeline = extracted.get("timeline", [])
    vars_ = extracted.get("variables", {})
    if not timeline:
        return ""

    rows: list[str] = ["", "Step-by-step data (mean ± spread [range]):"]

    # Temperature in Celsius (K spread == C spread for deltas)
    t2m_c = t2m_lo = t2m_hi = t2m_spread = None
    if "t2m" in vars_:
        s = vars_["t2m"]
        u = str(s.get("units", ""))
        t2m_c = to_celsius_if_needed(np.asarray(s["mean"]), units=u, var_name="t2m")
        t2m_lo = to_celsius_if_needed(np.asarray(s["min"]), units=u, var_name="t2m")
        t2m_hi = to_celsius_if_needed(np.asarray(s["max"]), units=u, var_name="t2m")
        t2m_spread = np.asarray(s.get("spread", []), dtype=float)

    # Combined wind speed from u/v
    w_avg = w_lo = w_hi = w_spread = None
    if "u10m" in vars_ and "v10m" in vars_:
        um = _as_member_matrix(vars_["u10m"])
        vm = _as_member_matrix(vars_["v10m"])
        if um is not None and vm is not None and um.shape == vm.shape:
            u_mph, _ = wind_to_mph(um, units=str(vars_["u10m"].get("units", "")))
            v_mph, _ = wind_to_mph(vm, units=str(vars_["v10m"].get("units", "")))
            spd = np.sqrt(u_mph**2 + v_mph**2)
            w_avg = np.mean(spd, axis=1)
            w_lo = np.min(spd, axis=1)
            w_hi = np.max(spd, axis=1)
            w_spread = np.std(spd, axis=1)

    # Per-step precipitation (handles cumulative->step conversion)
    tp_avg = tp_hi = tp_spread = None
    if "tp" in vars_:
        res = _precip_window_stats(vars_["tp"])
        if res is not None:
            _, step = res
            tp_avg = np.mean(step, axis=1)
            tp_hi = np.max(step, axis=1)
            tp_spread = np.std(step, axis=1)

    # Per-step snowfall
    sf_avg = sf_hi = sf_spread = None
    if "sf" in vars_:
        res = _precip_window_stats(vars_["sf"])
        if res is not None:
            _, step = res
            sf_avg = np.mean(step, axis=1)
            sf_hi = np.max(step, axis=1)
            sf_spread = np.std(step, axis=1)

    for i, ts in enumerate(timeline):
        local = _to_local_time_label(ts, timezone_name)
        parts: list[str] = []

        if t2m_c is not None and i < len(t2m_c):
            c = float(t2m_c[i])
            f = _c_to_f(c)
            lo = float(t2m_lo[i])
            hi = float(t2m_hi[i])
            sp = float(t2m_spread[i]) if t2m_spread is not None and i < len(t2m_spread) else 0.0
            parts.append(f"t2m {c:.1f}±{sp:.1f}C ({f:.1f}F) [{lo:.1f} to {hi:.1f}C]")

        if w_avg is not None and i < len(w_avg):
            sp = float(w_spread[i]) if w_spread is not None and i < len(w_spread) else 0.0
            parts.append(
                f"wind {float(w_avg[i]):.1f}±{sp:.1f} mph [{float(w_lo[i]):.1f}-{float(w_hi[i]):.1f}]"
            )

        if tp_avg is not None and i < len(tp_avg):
            p = float(tp_avg[i])
            px = float(tp_hi[i])
            p_in = float(mm_to_inches(np.asarray(p)))
            px_in = float(mm_to_inches(np.asarray(px)))
            sp = float(tp_spread[i]) if tp_spread is not None and i < len(tp_spread) else 0.0
            parts.append(f"precip {p:.2f}±{sp:.2f}mm ({p_in:.2f}in) [max {px:.2f}mm ({px_in:.2f}in)]")

        if sf_avg is not None and i < len(sf_avg):
            s = float(sf_avg[i])
            sx = float(sf_hi[i])
            sp = float(sf_spread[i]) if sf_spread is not None and i < len(sf_spread) else 0.0
            parts.append(f"snow {s:.2f}±{sp:.2f}mm SWE [max {sx:.2f}mm]")

        row = " | ".join(parts) if parts else "(no data)"
        rows.append(f"  {local}: {row}")

    return "\n".join(rows)


def build_structured_summary(extracted: dict[str, Any]) -> str:
    mode = extracted["mode"]
    member_dim = extracted["member_dim"]
    n_members = extracted["n_members"]
    timeline = extracted["timeline"]
    loc = extracted["location"]
    vars_ = extracted["variables"]
    timezone_name = str(extracted.get("timezone", "UTC"))
    current_local = fmt_time_label(extracted.get("current_time_local") or extracted.get("current_time_utc"))
    start_local = fmt_time_label(extracted.get("forecast_start_local") or extracted.get("forecast_start_utc"))
    end_local = fmt_time_label(extracted.get("forecast_end_local") or extracted.get("forecast_end_utc"))

    location_query = loc.get("query", "")
    location_label = loc.get("label", "")
    location_name = location_query or location_label or f"({loc['requested_lat']:.2f}, {loc['requested_lon']:.2f})"

    lines = [
        f"Mode: {mode}",
        f"Members ({member_dim}): {n_members}",
        f"Location: {location_name}",
        f"Location timezone: {timezone_name}",
        f"Current time (local): {current_local}",
        f"Forecast valid window (local): {start_local} -> {end_local}",
        (
            f"Coordinates: requested ({loc['requested_lat']:.2f}, {loc['requested_lon']:.2f}), "
            f"selected gridpoint ({loc['selected_lat']:.2f}, {loc['selected_lon']:.2f})"
        ),
        f"Timeline points: {len(timeline)}",
    ]

    wind_summary = None
    if "u10m" in vars_ and "v10m" in vars_:
        wind_summary = _wind_speed_summary(vars_["u10m"], vars_["v10m"])

    calibration_lines: list[str] = []
    for name, stats in vars_.items():
        if wind_summary is not None and name in {"u10m", "v10m"}:
            continue
        lines.append(_one_line_series(name, stats))
        band = calibrate_uncertainty_band(name, stats.get("spread", []), str(stats.get("units", "")))
        if band is not None:
            calibration_lines.append(
                (
                    f"{band.variable}: spread {band.spread_value:.2f} {band.spread_units}, "
                    f"climatology sigma {band.sigma_value:.2f} {band.sigma_units}, "
                    f"ratio {band.ratio_sigma:.2f} sigma -> {band.confidence_label}"
                )
            )

    if wind_summary is not None:
        wind_line, wind_calib_line = wind_summary
        lines.append(wind_line)
        calibration_lines.append(wind_calib_line)

    if mode == "forecast":
        probability_lines = _forecast_probability_lines(vars_, timeline, timezone_name)
        if probability_lines:
            lines.append("Impact probabilities:")
            lines.extend(probability_lines)

    if calibration_lines:
        lines.append("Uncertainty calibration bands:")
        lines.extend(calibration_lines)

    return "\n".join(lines)


def generate_narrative(
    extracted: dict[str, Any],
    *,
    mode: str,
    model_name: str,
    api_key: str | None,
    provider: str = "earth2",
    explicit_model: str | None = None,
) -> tuple[str | None, str | None]:
    """Generate natural-language weather narrative via OpenRouter/Nemotron.

    Returns (narrative, error_message). Exactly one entry is non-None.
    """
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        return None, "No OpenRouter API key found. Set --api-key or OPENROUTER_API_KEY."

    n_members = int(extracted["n_members"])
    timeline_points = int(len(extracted["timeline"]))
    if mode in {"forecast", "storm"}:
        # Earth2Studio forecasts include initialization at lead_time=0.
        n_steps = max(timeline_points - 1, 1)
    else:
        n_steps = timeline_points
    n_samples = n_members

    system_prompt = build_system_prompt(
        mode,
        model_name=model_name,
        n_members=n_members,
        n_steps=n_steps,
        n_samples=n_samples,
        provider=provider,
    )
    structured = build_structured_summary(extracted)
    timezone_name = str(extracted.get("timezone", "UTC"))
    step_table = _build_step_table(extracted, timezone_name)
    loc = extracted.get("location", {})
    location_name = loc.get("query") or loc.get("label") or "the requested location"

    payload = {
        "model": explicit_model or DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Write the weather narrative for {location_name}. "
                    "Cite the step-by-step values directly in the Timeline.\n\n"
                    f"{structured}"
                    f"{step_table}"
                ),
            },
        ],
        "temperature": 0.3,
    }

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/",
        "X-Title": "Whether",
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=90)
    except requests.RequestException as exc:
        return None, f"Nemotron API request failed: {exc}"

    if response.status_code >= 400:
        message = response.text
        try:
            obj = response.json()
            message = json.dumps(obj, indent=2)
        except ValueError:
            pass
        return None, f"Nemotron API error ({response.status_code}): {message}"

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return str(content).strip(), None
    except Exception as exc:  # noqa: BLE001
        return None, f"Nemotron response parsing failed: {exc}"


__all__ = ["build_structured_summary", "generate_narrative"]
