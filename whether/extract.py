from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import xarray as xr

from .geocode import normalize_lon_180, normalize_lon_360

try:
    from timezonefinder import TimezoneFinder

    _TZF = TimezoneFinder()
except Exception:  # noqa: BLE001
    _TZF = None


def _split_variable_dataset(ds: xr.Dataset) -> dict[str, xr.DataArray]:
    if len(ds.data_vars) == 1:
        name, da = next(iter(ds.data_vars.items()))
        if "variable" in da.dims and "variable" in da.coords:
            out: dict[str, xr.DataArray] = {}
            for v in da.coords["variable"].values:
                v_name = str(v)
                out[v_name] = da.sel(variable=v).drop_vars("variable", errors="ignore")
            return out
        return {name: da}

    return {name: da for name, da in ds.data_vars.items()}


def _as_datetime64(value: Any) -> np.datetime64:
    if isinstance(value, np.datetime64):
        return value
    if isinstance(value, datetime):
        return np.datetime64(value)
    return np.datetime64(str(value))


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
        # Handle nanosecond-precision strings like 2026-02-16T00:00:00.000000000
        if "." in text:
            base, frac = text.split(".", 1)
            frac = (frac + "000000")[:6]
            parsed = datetime.fromisoformat(f"{base}.{frac}")
            return parsed.replace(tzinfo=UTC)
        dt64 = np.datetime64(text).astype("datetime64[s]")
        epoch = int(dt64.astype(np.int64))
        return datetime.fromtimestamp(epoch, tz=UTC)


def _timezone_name_for_location(lat: float, lon: float) -> str:
    if _TZF is None:
        return "UTC"
    tz_name = _TZF.timezone_at(lat=float(lat), lng=float(lon))
    if tz_name:
        return tz_name
    return "UTC"


def _format_local_time(value: str | None, tz_name: str) -> str | None:
    if not value:
        return None
    dt_utc = _to_utc_datetime(value)
    try:
        zone = ZoneInfo(tz_name)
    except Exception:  # noqa: BLE001
        zone = ZoneInfo("UTC")
    return dt_utc.astimezone(zone).strftime("%Y-%m-%d %H:%M %Z")


def _safe_datetime64(value: str) -> np.datetime64 | None:
    try:
        return np.datetime64(str(value))
    except Exception:  # noqa: BLE001
        return None


def _timeline_from_coords(da: xr.DataArray) -> tuple[list[str], int]:
    if "time" in da.coords and da.coords["time"].size > 0:
        time_values = da.coords["time"].values
        if np.ndim(time_values) == 0:
            t0 = _as_datetime64(time_values)
        else:
            t0 = _as_datetime64(time_values[0])
    else:
        t0 = np.datetime64("now")

    if "lead_time" in da.coords and da.coords["lead_time"].size > 0:
        lead = da.coords["lead_time"].values
        times = [str(t0 + dt) for dt in lead]
        return times, int(len(lead))

    return [str(t0)], 1


def _ensure_member_dim(da: xr.DataArray, member_dim: str) -> xr.DataArray:
    if member_dim in da.dims:
        return da
    return da.expand_dims({member_dim: [0]})


def _canonical_member_lead(da: xr.DataArray, member_dim: str) -> tuple[np.ndarray, list[str]]:
    # Keep scalar coordinates (notably `time`) so timeline labels remain tied
    # to model init time instead of falling back to wall-clock time.
    da = da.squeeze(drop=False)
    da = _ensure_member_dim(da, member_dim)

    if "lead_time" in da.dims:
        order = ["lead_time", member_dim]
    else:
        da = da.expand_dims({"lead_time": [np.timedelta64(0, "h")]})
        order = ["lead_time", member_dim]

    array = da.transpose(*order).values
    timeline, _ = _timeline_from_coords(da)
    return np.asarray(array), timeline


def _pick_regular_point(
    da: xr.DataArray,
    lat: float,
    lon: float,
    *,
    prefer_interp: bool = False,
) -> tuple[xr.DataArray, float, float, str]:
    lon_values = da.coords.get("lon")
    if lon_values is None:
        raise ValueError("Expected lon coordinate.")

    lon_candidate = normalize_lon_360(lon) if float(np.nanmax(lon_values.values)) > 180.0 else normalize_lon_180(lon)

    if prefer_interp:
        try:
            interpolated = da.interp(lat=float(lat), lon=float(lon_candidate), method="linear")
            if not np.isnan(np.asarray(interpolated.values)).all():
                return (
                    interpolated,
                    float(lat),
                    float(normalize_lon_180(lon_candidate)),
                    "bilinear_interpolation",
                )
        except Exception:  # noqa: BLE001
            pass

    selected = da.sel(lat=lat, lon=lon_candidate, method="nearest")
    sel_lat = float(selected.coords["lat"].values)
    sel_lon = float(selected.coords["lon"].values)
    return selected, sel_lat, normalize_lon_180(sel_lon), "nearest_gridpoint"


def _pick_2d_grid_point(da: xr.DataArray, lat: float, lon: float) -> tuple[xr.DataArray, float, float, str]:
    if "lat" not in da.coords or "lon" not in da.coords:
        raise ValueError("Expected 2D lat/lon coordinates for storm/zoom extraction.")

    lat2 = np.asarray(da.coords["lat"].values)
    lon2 = np.asarray(da.coords["lon"].values)
    lon_target = normalize_lon_180(lon)
    dist = (lat2 - lat) ** 2 + (normalize_lon_180(lon2) - lon_target) ** 2
    iy, ix = np.unravel_index(np.argmin(dist), dist.shape)

    y_dim, x_dim = da.coords["lat"].dims
    selected = da.isel({y_dim: int(iy), x_dim: int(ix)})
    sel_lat = float(lat2[iy, ix])
    sel_lon = float(normalize_lon_180(lon2[iy, ix]))
    return selected, sel_lat, sel_lon, "nearest_gridpoint"


def _pick_storm_point(da: xr.DataArray, lat: float, lon: float) -> tuple[xr.DataArray, float, float, str]:
    if "HRRR_Y" in da.dims and "HRRR_X" in da.dims and "lat" in da.coords and "lon" in da.coords:
        return _pick_2d_grid_point(da, lat, lon)

    if "lat" in da.dims and "lon" in da.dims:
        return _pick_regular_point(da, lat, lon, prefer_interp=False)

    raise ValueError("Storm dataset does not expose recognizable coordinates for point extraction.")


def _stats(member_values: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": np.mean(member_values, axis=1).tolist(),
        "spread": np.std(member_values, axis=1).tolist(),
        "min": np.min(member_values, axis=1).tolist(),
        "max": np.max(member_values, axis=1).tolist(),
    }


def extract_at_location(ds: xr.Dataset, lat: float, lon: float, mode: str) -> dict[str, Any]:
    """Extract forecast/sample data at nearest location and compute ensemble statistics."""
    vars_map = _split_variable_dataset(ds)

    member_dim = "sample" if mode == "zoom" else "ensemble"
    timeline: list[str] = []
    out_vars: dict[str, Any] = {}
    selected_lat: float | None = None
    selected_lon: float | None = None
    selection_method: str | None = None

    for var_name, da in vars_map.items():
        if mode == "forecast":
            if "lat" not in da.coords or "lon" not in da.coords:
                continue
            picked, sel_lat, sel_lon, method = _pick_regular_point(da, lat, lon, prefer_interp=True)
        elif mode == "storm":
            picked, sel_lat, sel_lon, method = _pick_storm_point(da, lat, lon)
        elif mode == "zoom":
            if "lat" in da.coords and "lon" in da.coords and len(da.coords["lat"].dims) == 2:
                picked, sel_lat, sel_lon, method = _pick_2d_grid_point(da, lat, lon)
            else:
                picked, sel_lat, sel_lon, method = _pick_regular_point(da, lat, lon, prefer_interp=True)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        values, var_timeline = _canonical_member_lead(picked, member_dim=member_dim)
        if not timeline:
            timeline = var_timeline
            selected_lat, selected_lon = sel_lat, sel_lon
            selection_method = method

        out_vars[var_name] = {
            "units": str(da.attrs.get("units", "")),
            "members": values.tolist(),
            **_stats(values),
        }

    if not out_vars:
        raise ValueError("No extractable variables found at requested location.")

    n_members = len(next(iter(out_vars.values()))["members"][0])
    current_time_utc = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    forecast_start_utc = timeline[0] if timeline else None
    forecast_end_utc = timeline[-1] if timeline else None
    tz_name = _timezone_name_for_location(lat, lon)
    current_time_local = _format_local_time(current_time_utc, tz_name)
    forecast_start_local = _format_local_time(forecast_start_utc, tz_name)
    forecast_end_local = _format_local_time(forecast_end_utc, tz_name)

    return {
        "mode": mode,
        "member_dim": member_dim,
        "n_members": n_members,
        "timezone": tz_name,
        "current_time_utc": current_time_utc,
        "forecast_start_utc": forecast_start_utc,
        "forecast_end_utc": forecast_end_utc,
        "current_time_local": current_time_local,
        "forecast_start_local": forecast_start_local,
        "forecast_end_local": forecast_end_local,
        "timeline": timeline,
        "location": {
            "requested_lat": float(lat),
            "requested_lon": normalize_lon_180(float(lon)),
            "selected_lat": float(selected_lat if selected_lat is not None else lat),
            "selected_lon": float(selected_lon if selected_lon is not None else lon),
            "selection_method": selection_method or "nearest_gridpoint",
        },
        "variables": out_vars,
    }


def trim_extracted_to_future(
    extracted: dict[str, Any],
    *,
    now_utc: datetime,
    future_hours: int,
) -> dict[str, Any]:
    """Keep nearest pre-now forecast point and all points in [now, now+future_hours]."""
    if extracted.get("mode") != "forecast":
        return extracted

    timeline = [str(t) for t in extracted.get("timeline", [])]
    if not timeline:
        return extracted

    now64 = np.datetime64(now_utc.astimezone(UTC).replace(tzinfo=None))
    end64 = now64 + np.timedelta64(max(int(future_hours), 1), "h")

    parsed_times: list[tuple[int, np.datetime64]] = []
    for i, ts in enumerate(timeline):
        t64 = _safe_datetime64(ts)
        if t64 is None:
            continue
        parsed_times.append((i, t64))

    future_indices = [i for i, t64 in parsed_times if now64 <= t64 <= end64]
    left_candidates = [i for i, t64 in parsed_times if t64 < now64]
    left_idx = max(left_candidates) if left_candidates else None

    valid_indices: list[int] = []
    if left_idx is not None:
        valid_indices.append(left_idx)
    valid_indices.extend(future_indices)
    valid_indices = sorted(set(valid_indices))

    if not valid_indices:
        if parsed_times:
            nearest_idx = min(parsed_times, key=lambda item: abs((item[1] - now64) / np.timedelta64(1, "s")))[0]
            valid_indices = [nearest_idx]
        else:
            valid_indices = [len(timeline) - 1]

    def _slice_series(values: list[Any]) -> list[Any]:
        return [values[i] for i in valid_indices if i < len(values)]

    trimmed = dict(extracted)
    trimmed_timeline = [timeline[i] for i in valid_indices]
    trimmed_vars: dict[str, Any] = {}
    for name, stats in extracted.get("variables", {}).items():
        out_stats = dict(stats)
        for key in ("members", "mean", "spread", "min", "max"):
            if key in out_stats and isinstance(out_stats[key], list):
                out_stats[key] = _slice_series(out_stats[key])
        trimmed_vars[name] = out_stats

    trimmed["timeline"] = trimmed_timeline
    trimmed["variables"] = trimmed_vars
    trimmed["forecast_start_utc"] = trimmed_timeline[0] if trimmed_timeline else None
    trimmed["forecast_end_utc"] = trimmed_timeline[-1] if trimmed_timeline else None
    tz_name = str(trimmed.get("timezone", "UTC"))
    trimmed["current_time_local"] = _format_local_time(trimmed.get("current_time_utc"), tz_name)
    trimmed["forecast_start_local"] = _format_local_time(trimmed.get("forecast_start_utc"), tz_name)
    trimmed["forecast_end_local"] = _format_local_time(trimmed.get("forecast_end_utc"), tz_name)
    return trimmed


__all__ = ["extract_at_location", "trim_extracted_to_future"]
