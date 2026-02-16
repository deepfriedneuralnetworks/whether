from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
import shutil
import time as pytime
from typing import Literal

import numpy as np
import xarray as xr

from .geocode import normalize_lon_180, normalize_lon_360

Mode = Literal["forecast", "storm", "zoom"]


@dataclass
class ForecastRun:
    mode: Mode
    model_name: str
    dataset: xr.Dataset
    output_path: Path
    init_time: datetime
    member_dim: str
    warnings: list[str] = field(default_factory=list)


def _floor_cycle(now: datetime, hours: int) -> datetime:
    if hours <= 0:
        raise ValueError("Cycle hour interval must be >= 1.")
    now = now.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    floored_hour = now.hour - (now.hour % hours)
    return now.replace(hour=floored_hour)


def _init_time_or_default(time: str | None, cycle_hours: int) -> datetime:
    if time:
        parsed = datetime.fromisoformat(time.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return _floor_cycle(datetime.now(UTC), cycle_hours)


def _output_path(output_dir: Path, prefix: str, init_time: datetime, run_tag: str | None = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = init_time.strftime("%Y%m%dT%H%M%SZ")
    if run_tag:
        return output_dir / f"{prefix}_{stamp}_{run_tag}.zarr"
    return output_dir / f"{prefix}_{stamp}.zarr"


def _half_width_for_radius_km(lat: float, radius_km: float = 500.0, margin_deg: float = 1.0) -> float:
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / max(111.0 * max(np.cos(np.deg2rad(abs(lat))), 0.15), 1e-6)
    return min(max(lat_delta, lon_delta) + margin_deg, 30.0)


def _compute_step_window(
    *,
    init_time: datetime,
    base_step_hours: int,
    default_steps: int,
    now_utc: datetime | None,
    future_hours: int | None,
) -> tuple[int, int]:
    if now_utc is None or future_hours is None:
        return 0, max(int(default_steps), 1)

    elapsed_hours = (now_utc.astimezone(UTC) - init_time.astimezone(UTC)).total_seconds() / 3600.0
    elapsed_steps = elapsed_hours / float(base_step_hours)
    # Keep one point before "now" so charts and narratives can show context
    # to the left of current time, then all requested future lead times.
    start_idx = max(0, int(np.floor(elapsed_steps)) - 1)
    end_idx = max(start_idx + 1, int(np.ceil((elapsed_hours + float(future_hours)) / float(base_step_hours))))
    return start_idx, end_idx


def _valid_cache_file(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _download_ecmwf_grib(
    client,
    *,
    target: Path,
    use_cache: bool,
    warnings: list[str],
    cache_label: str,
    request: dict,
) -> None:
    if use_cache and _valid_cache_file(target):
        warnings.append(f"Using cached ECMWF {cache_label}: {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)

    if use_cache:
        temp_target = target.parent / f"{target.name}.tmp.{os.getpid()}.{pytime.time_ns()}"
    else:
        temp_target = target

    try:
        client.retrieve(**request, target=str(temp_target))
        if not _valid_cache_file(temp_target):
            raise RuntimeError(f"ECMWF download produced an empty file: {temp_target}")
        if use_cache:
            temp_target.replace(target)
    except Exception:
        if temp_target.exists():
            temp_target.unlink(missing_ok=True)
        raise


def _lat_lon_window(lat: float, lon: float, resolution: float, half_width_deg: float) -> tuple[np.ndarray, np.ndarray]:
    lat_start = round((lat - half_width_deg) / resolution) * resolution
    lat_end = round((lat + half_width_deg) / resolution) * resolution

    lats = np.arange(lat_start, lat_end + resolution / 2.0, resolution, dtype=np.float64)
    lon_center = float(normalize_lon_180(lon))
    n_lon = int(round(half_width_deg / resolution))
    lon_offsets = np.arange(-n_lon, n_lon + 1, dtype=np.float64) * resolution
    lons = np.array([float(normalize_lon_180(lon_center + off)) for off in lon_offsets], dtype=np.float64)
    lons = np.sort(np.unique(np.round(lons, 10)))
    lats = np.clip(lats, -90.0, 90.0)
    return lats, lons


def _mock_global_dataset(
    *,
    lat: float,
    lon: float,
    model_name: str,
    steps: int,
    members: int,
    init_time: datetime,
) -> xr.Dataset:
    lead_hours = np.arange(0, (steps + 1) * 6, 6, dtype=np.int32)
    n_leads = int(len(lead_hours))
    lead_time = np.array([np.timedelta64(int(h), "h") for h in lead_hours])
    lats, lons = _lat_lon_window(lat, lon, resolution=0.25, half_width_deg=_half_width_for_radius_km(lat, radius_km=500.0))

    rng = np.random.default_rng(abs(hash((round(lat, 3), round(lon, 3), model_name))) % (2**32))

    model_vars = {
        "dlwp": ["t2m", "z500", "t850"],
        "fcn": ["t2m", "u10m", "v10m", "tp"],
        "fcn3": ["t2m", "u10m", "v10m", "tp"],
        "ecmwf": ["t2m", "u10m", "v10m", "tp"],
    }
    variables = model_vars.get(model_name, model_vars["dlwp"])

    coords = {
        "time": np.array([np.datetime64(init_time.replace(tzinfo=None))]),
        "lead_time": lead_time,
        "ensemble": np.arange(members, dtype=np.int32),
        "lat": lats,
        "lon": lons,
    }
    ds = xr.Dataset(coords=coords)

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    lon_center = float(normalize_lon_180(lon))
    lon_delta = ((lon_grid - lon_center + 180.0) % 360.0) - 180.0
    lat_delta = lat_grid - float(lat)
    background_wave = (
        0.9 * np.sin(np.deg2rad(lat_grid * 3.2 + lon_delta * 1.8))
        + 0.7 * np.cos(np.deg2rad(lat_delta * 5.0 - lon_grid * 1.4))
    )

    for name in variables:
        shape = (1, n_leads, members, lats.size, lons.size)
        arr = np.zeros(shape, dtype=np.float32)

        if name == "t2m":
            base = (
                10.5
                - 0.14 * lat_delta
                + 0.05 * lon_delta
                + 1.8 * background_wave
                + 1.2 * np.exp(-(((lat_delta + 2.0) / 2.4) ** 2 + ((lon_delta - 3.0) / 3.8) ** 2))
                - 1.0 * np.exp(-(((lat_delta - 1.0) / 2.8) ** 2 + ((lon_delta + 3.5) / 3.0) ** 2))
            )
            member_offset = rng.normal(0.0, 0.7, size=(members, 1, 1))
            member_texture = rng.normal(0.0, 0.15, size=(members, lats.size, lons.size))
            for lead_idx in range(n_leads):
                phase = 0.0 if n_leads == 1 else float(lead_idx) / float(n_leads - 1)
                moving_feature = 1.4 * np.exp(
                    -(
                        ((lat_delta - (-1.5 + 3.0 * phase)) / 2.6) ** 2
                        + ((lon_delta - (-4.0 + 8.0 * phase)) / 3.6) ** 2
                    )
                )
                trend = 0.6 + 2.0 * phase
                step_noise = rng.normal(0.0, 0.08, size=(members, lats.size, lons.size))
                arr[0, lead_idx, :, :, :] = (
                    base[np.newaxis, :, :] + moving_feature[np.newaxis, :, :] + trend + member_offset + member_texture + step_noise
                )
        elif name == "tcwv":
            baseline = 25.0 + 5.0 * np.cos(np.deg2rad(lat_grid))
            trend = np.linspace(-1.0, 1.5, n_leads, dtype=np.float32).reshape(1, n_leads, 1, 1, 1)
            member_noise = rng.normal(0.0, 0.9, size=(1, n_leads, members, 1, 1))
            arr = baseline.reshape(1, 1, 1, lats.size, lons.size) + trend + member_noise
        elif name == "z500":
            baseline = 5600.0 + 20.0 * np.sin(np.deg2rad(lat_grid))
            trend = np.linspace(10.0, -10.0, n_leads, dtype=np.float32).reshape(1, n_leads, 1, 1, 1)
            member_noise = rng.normal(0.0, 6.0, size=(1, n_leads, members, 1, 1))
            arr = baseline.reshape(1, 1, 1, lats.size, lons.size) + trend + member_noise
        elif name == "t850":
            baseline = 5.0 - 0.3 * np.abs(lat - lat_grid)
            trend = np.linspace(0.5, 2.0, n_leads, dtype=np.float32).reshape(1, n_leads, 1, 1, 1)
            member_noise = rng.normal(0.0, 1.0, size=(1, n_leads, members, 1, 1))
            arr = baseline.reshape(1, 1, 1, lats.size, lons.size) + trend + member_noise
        elif name == "u10m":
            baseline = 2.4 + 1.1 * np.sin(np.deg2rad(lat_grid * 4.0)) + 0.8 * np.cos(np.deg2rad(lon_delta * 3.2))
            trend = np.linspace(-0.3, 1.0, n_leads, dtype=np.float32).reshape(1, n_leads, 1, 1, 1)
            member_noise = rng.normal(0.0, 0.65, size=(1, n_leads, members, 1, 1))
            spatial = 0.5 * np.sin(np.deg2rad((lat_delta * 5.5) + (lon_delta * 2.5)))
            arr = baseline.reshape(1, 1, 1, lats.size, lons.size) + trend + member_noise + spatial.reshape(
                1, 1, 1, lats.size, lons.size
            )
        elif name == "v10m":
            baseline = -0.8 + 1.0 * np.cos(np.deg2rad(lat_grid * 3.2)) + 0.9 * np.sin(np.deg2rad(lon_delta * 4.0))
            trend = np.linspace(0.4, -0.3, n_leads, dtype=np.float32).reshape(1, n_leads, 1, 1, 1)
            member_noise = rng.normal(0.0, 0.65, size=(1, n_leads, members, 1, 1))
            spatial = 0.4 * np.cos(np.deg2rad((lat_delta * 4.2) - (lon_delta * 3.2)))
            arr = baseline.reshape(1, 1, 1, lats.size, lons.size) + trend + member_noise + spatial.reshape(
                1, 1, 1, lats.size, lons.size
            )
        elif name == "sp":
            baseline = 101300.0 + 150.0 * np.sin(np.deg2rad(lat_grid))
            trend = np.linspace(0.0, -250.0, n_leads, dtype=np.float32).reshape(1, n_leads, 1, 1, 1)
            member_noise = rng.normal(0.0, 35.0, size=(1, n_leads, members, 1, 1))
            arr = baseline.reshape(1, 1, 1, lats.size, lons.size) + trend + member_noise
        elif name == "tp":
            cumulative = np.zeros((members, lats.size, lons.size), dtype=np.float32)
            member_wet_bias = np.clip(rng.normal(1.0, 0.22, size=(members, 1, 1)), 0.4, 1.8)
            for lead_idx in range(n_leads):
                phase = 0.0 if n_leads == 1 else float(lead_idx) / float(n_leads - 1)
                cell_lat = -1.8 + 3.2 * np.sin(phase * np.pi * 1.2)
                cell_lon = -5.0 + 9.0 * phase
                convective_core = np.exp(-(((lat_delta - cell_lat) / 2.2) ** 2 + ((lon_delta - cell_lon) / 2.8) ** 2))
                stratiform = np.clip(0.22 + 0.16 * background_wave, a_min=0.0, a_max=None)
                step_inc = np.clip(
                    (0.55 * convective_core + stratiform)[np.newaxis, :, :] * member_wet_bias
                    + np.abs(rng.normal(0.0, 0.08, size=(members, lats.size, lons.size))),
                    a_min=0.0,
                    a_max=None,
                )
                if lead_idx == 0:
                    cumulative[:, :, :] = 0.0
                else:
                    cumulative += step_inc.astype(np.float32)
                arr[0, lead_idx, :, :, :] = cumulative

        ds[name] = (("time", "lead_time", "ensemble", "lat", "lon"), arr.astype(np.float32))

    ds.attrs.update({"mode": "forecast", "model_name": model_name, "mock": "true"})
    return ds


def _mock_storm_dataset(
    *,
    lat: float,
    lon: float,
    steps: int,
    members: int,
    init_time: datetime,
) -> xr.Dataset:
    ny, nx = 96, 128
    y = np.arange(ny, dtype=np.int32)
    x = np.arange(nx, dtype=np.int32)

    lead_hours = np.arange(0, steps + 1, 1, dtype=np.int32)
    n_leads = int(len(lead_hours))
    lead_time = np.array([np.timedelta64(int(h), "h") for h in lead_hours])

    lat_vals = lat + (y - (ny // 2)) * 0.03
    lon_vals = normalize_lon_180(lon) + (x - (nx // 2)) * 0.03
    lat2d, lon2d = np.meshgrid(lat_vals, lon_vals, indexing="ij")

    rng = np.random.default_rng(abs(hash((round(lat, 3), round(lon, 3), steps, members, "storm"))) % (2**32))

    coords = {
        "time": np.array([np.datetime64(init_time.replace(tzinfo=None))]),
        "lead_time": lead_time,
        "ensemble": np.arange(members, dtype=np.int32),
        "HRRR_Y": y,
        "HRRR_X": x,
    }

    ds = xr.Dataset(coords=coords)
    ds = ds.assign_coords(lat=(("HRRR_Y", "HRRR_X"), lat2d), lon=(("HRRR_Y", "HRRR_X"), lon2d))

    shape = (1, n_leads, members, ny, nx)

    spatial_wave = np.sin(np.deg2rad(lat2d * 8.0)) + np.cos(np.deg2rad(lon2d * 8.0))
    for name in ["t2m", "u10m", "v10m", "refc", "sp"]:
        if name == "t2m":
            arr = (
                18.0
                + 2.0 * spatial_wave.reshape(1, 1, 1, ny, nx)
                + np.linspace(0.0, -1.8, n_leads).reshape(1, n_leads, 1, 1, 1)
                + rng.normal(0.0, 1.2, size=shape)
            )
        elif name == "u10m":
            arr = (
                4.5
                + 1.5 * spatial_wave.reshape(1, 1, 1, ny, nx)
                + np.linspace(0.0, 1.0, n_leads).reshape(1, n_leads, 1, 1, 1)
                + rng.normal(0.0, 0.9, size=shape)
            )
        elif name == "v10m":
            arr = (
                2.5
                + 1.1 * spatial_wave.reshape(1, 1, 1, ny, nx)
                + np.linspace(0.3, -0.7, n_leads).reshape(1, n_leads, 1, 1, 1)
                + rng.normal(0.0, 0.9, size=shape)
            )
        elif name == "refc":
            arr = (
                28.0
                + 12.0 * np.maximum(0.0, spatial_wave).reshape(1, 1, 1, ny, nx)
                + np.linspace(0.0, 12.0, n_leads).reshape(1, n_leads, 1, 1, 1)
                + np.abs(rng.normal(0.0, 5.0, size=shape))
            )
        elif name == "sp":
            arr = (
                101000.0
                + 90.0 * spatial_wave.reshape(1, 1, 1, ny, nx)
                + np.linspace(0.0, -180.0, n_leads).reshape(1, n_leads, 1, 1, 1)
                + rng.normal(0.0, 30.0, size=shape)
            )

        ds[name] = (("time", "lead_time", "ensemble", "HRRR_Y", "HRRR_X"), arr.astype(np.float32))

    ds.attrs.update({"mode": "storm", "model_name": "stormcast", "mock": "true"})
    return ds


def _mock_zoom_dataset(
    *,
    samples: int,
    init_time: datetime,
) -> xr.Dataset:
    ny, nx = 128, 128
    y = np.arange(ny, dtype=np.int32)
    x = np.arange(nx, dtype=np.int32)

    lat_vals = np.linspace(21.5, 25.5, ny)
    lon_vals = np.linspace(119.2, 122.6, nx)
    lat2d, lon2d = np.meshgrid(lat_vals, lon_vals, indexing="ij")

    rng = np.random.default_rng(abs(hash((samples, "corrdiff"))) % (2**32))

    coords = {
        "time": np.array([np.datetime64(init_time.replace(tzinfo=None))]),
        "sample": np.arange(samples, dtype=np.int32),
        "y": y,
        "x": x,
    }
    ds = xr.Dataset(coords=coords)
    ds = ds.assign_coords(lat=(("y", "x"), lat2d), lon=(("y", "x"), lon2d))

    shape = (1, samples, ny, nx)
    base = np.sin(np.deg2rad((lat2d - 22.0) * 10.0)) + np.cos(np.deg2rad((lon2d - 120.0) * 12.0))

    t2m = 24.0 + 2.2 * base.reshape(1, 1, ny, nx) + rng.normal(0.0, 0.8, size=shape)
    u10m = 3.5 + 1.0 * base.reshape(1, 1, ny, nx) + rng.normal(0.0, 0.6, size=shape)
    v10m = 1.8 + 1.0 * base.reshape(1, 1, ny, nx) + rng.normal(0.0, 0.6, size=shape)
    tp = np.maximum(0.0, 2.0 + 3.5 * np.maximum(0.0, base).reshape(1, 1, ny, nx) + rng.normal(0.0, 1.0, size=shape))

    ds["t2m"] = (("time", "sample", "y", "x"), t2m.astype(np.float32))
    ds["u10m"] = (("time", "sample", "y", "x"), u10m.astype(np.float32))
    ds["v10m"] = (("time", "sample", "y", "x"), v10m.astype(np.float32))
    ds["tp"] = (("time", "sample", "y", "x"), tp.astype(np.float32))

    ds.attrs.update({"mode": "zoom", "model_name": "corrdiff-taiwan", "mock": "true"})
    return ds


def _save_dataset(ds: xr.Dataset, path: Path) -> xr.Dataset:
    if path.exists():
        shutil.rmtree(path)
    ds.to_zarr(path, mode="w")
    return xr.open_zarr(path)


def _model_output_variables(model) -> list[str]:
    """Read the model's output variable coordinate names."""
    coords = model.output_coords(model.input_coords())
    return [str(v) for v in coords.get("variable", [])]


def _select_output_variables(model, requested: list[str]) -> np.ndarray:
    available = _model_output_variables(model)
    selected = [v for v in requested if v in available]
    if not selected:
        # Fallback: keep a manageable subset while preserving model order.
        selected = available[: min(len(available), 6)]
    return np.array(selected)


def _ensure_lead_time_timedelta(ds: xr.Dataset) -> xr.Dataset:
    if "lead_time" not in ds.coords:
        return ds
    lead_coord = ds.coords["lead_time"]
    values = np.asarray(lead_coord.values)
    if np.issubdtype(values.dtype, np.timedelta64):
        return ds

    if np.issubdtype(values.dtype, np.number):
        units_hint = str(lead_coord.attrs.get("units", "")).lower()
        unit = "h"
        if "sec" in units_hint or units_hint in {"s"}:
            unit = "s"
        elif "min" in units_hint or units_hint in {"m"}:
            unit = "m"
        elif "hour" in units_hint or units_hint in {"h"}:
            unit = "h"
        else:
            nonzero = np.abs(values.astype(float))
            nonzero = nonzero[nonzero > 0]
            if nonzero.size and np.all(np.isclose(np.mod(nonzero, 3600.0), 0.0)):
                unit = "s"
            elif nonzero.size and np.nanmax(nonzero) > 5000:
                unit = "s"
            elif nonzero.size and np.all(np.isclose(np.mod(nonzero, 60.0), 0.0)) and np.nanmax(nonzero) > 500:
                unit = "m"
            else:
                unit = "h"

        converted = np.array([np.timedelta64(int(v), unit) for v in values])
        return ds.assign_coords(lead_time=converted)

    try:
        converted = values.astype("timedelta64[s]")
        return ds.assign_coords(lead_time=converted)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported lead_time coordinate dtype: {values.dtype}") from exc


def _subset_global_window(ds: xr.Dataset, lat: float, lon: float, half_width_deg: float = 3.0) -> xr.Dataset:
    if "lat" not in ds.coords or "lon" not in ds.coords:
        return ds
    if ds.coords["lat"].ndim != 1 or ds.coords["lon"].ndim != 1:
        return ds

    lat_vals = np.asarray(ds.coords["lat"].values, dtype=float)
    lon_vals = np.asarray(ds.coords["lon"].values, dtype=float)

    lat_mask = (lat_vals >= (lat - half_width_deg)) & (lat_vals <= (lat + half_width_deg))
    if float(np.nanmax(lon_vals)) > 180.0:
        lon_target = normalize_lon_360(lon)
        lon_diff = ((lon_vals - lon_target + 180.0) % 360.0) - 180.0
    else:
        lon_target = normalize_lon_180(lon)
        lon_diff = ((lon_vals - lon_target + 180.0) % 360.0) - 180.0
    lon_mask = np.abs(lon_diff) <= half_width_deg

    if lat_mask.sum() < 2 or lon_mask.sum() < 2:
        return ds

    return ds.isel(
        lat=np.where(lat_mask)[0],
        lon=np.where(lon_mask)[0],
    )


def _open_ecmwf_grib_as_dataset(path: Path, *, control_member: int | None = None) -> xr.Dataset:
    import cfgrib

    variable_map = {
        "t2m": "t2m",
        "2t": "t2m",
        "u10": "u10m",
        "10u": "u10m",
        "v10": "v10m",
        "10v": "v10m",
        "msl": "msl",
        "tp": "tp",
        "tcwv": "tcwv",
    }

    datasets = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
    out_vars: dict[str, xr.DataArray] = {}

    for ds in datasets:
        for short_name, target_name in variable_map.items():
            if short_name not in ds.data_vars:
                continue
            da = ds[short_name]

            rename_dims: dict[str, str] = {}
            for dim in da.dims:
                if dim == "number":
                    rename_dims[dim] = "ensemble"
                elif dim == "step":
                    rename_dims[dim] = "lead_time"
                elif dim == "latitude":
                    rename_dims[dim] = "lat"
                elif dim == "longitude":
                    rename_dims[dim] = "lon"
            da = da.rename(rename_dims)

            if "ensemble" not in da.dims:
                if control_member is None:
                    continue
                da = da.expand_dims({"ensemble": [control_member]})
            else:
                da = da.assign_coords(ensemble=np.asarray(da.coords["ensemble"].values, dtype=np.int32))

            keep_dims = [d for d in ["ensemble", "lead_time", "lat", "lon"] if d in da.dims]
            da = da.transpose(*keep_dims)
            extra_coords = [c for c in da.coords if c not in da.dims]
            if extra_coords:
                da = da.drop_vars(extra_coords, errors="ignore")
            da = da.astype(np.float32)
            out_vars[target_name] = da

    if not out_vars:
        raise RuntimeError(f"No recognized ECMWF variables found in {path}")

    ds_out = xr.Dataset(out_vars)
    ds_out = _ensure_lead_time_timedelta(ds_out)
    if "lat" in ds_out.coords:
        ds_out = ds_out.sortby("lat")
    return ds_out


def run_ecmwf_ensemble(
    *,
    lat: float,
    lon: float,
    output_dir: Path,
    steps: int,
    members: int,
    time: str | None,
    dry_run: bool,
    source: str = "aws",
    cache_dir: Path | None = None,
    use_cache: bool = True,
    now_utc: datetime | None = None,
    future_hours: int | None = None,
) -> ForecastRun:
    if steps < 1:
        raise ValueError("--steps must be >= 1.")
    if members < 1:
        raise ValueError("--members must be >= 1.")

    requested_init_time = _init_time_or_default(time, cycle_hours=6)
    warnings: list[str] = []
    run_tag = f"r{os.getpid()}_{pytime.time_ns()}"

    if dry_run:
        out_path = _output_path(output_dir, "forecast_ecmwf", requested_init_time, run_tag=run_tag)
        ds = _mock_global_dataset(
            lat=lat,
            lon=lon,
            model_name="ecmwf",
            steps=steps,
            members=members,
            init_time=requested_init_time,
        )
        ds.attrs["provider"] = "ecmwf"
        return ForecastRun(
            mode="forecast",
            model_name="ecmwf-ifs-ens",
            dataset=_save_dataset(ds, out_path),
            output_path=out_path,
            init_time=requested_init_time,
            member_dim="ensemble",
            warnings=warnings,
        )

    try:
        from ecmwf.opendata import Client
    except ImportError as exc:
        raise RuntimeError(
            "ECMWF provider requires ecmwf-opendata. Install with: pip install ecmwf-opendata cfgrib eccodes"
        ) from exc

    capped_members = min(max(int(members), 1), 50)
    if members > 50:
        warnings.append("ECMWF open ENS supports up to 50 perturbed members; capping --members to 50.")

    client = Client(source=source, model="ifs", preserve_request_order=True)

    if time is None:
        latest = client.latest(stream="enfo", type="pf", number=1, step=0, param="2t")
        init_time = latest.replace(tzinfo=UTC) if latest.tzinfo is None else latest.astimezone(UTC)
        warnings.append(f"Using latest available ECMWF ENS cycle: {init_time.isoformat()}.")
    else:
        init_time = requested_init_time

    out_path = _output_path(output_dir, "forecast_ecmwf", init_time, run_tag=run_tag)
    if cache_dir is None:
        cache_dir = Path(".cache/whether/ecmwf")
    cache_dir.mkdir(parents=True, exist_ok=True)

    start_idx, end_idx = _compute_step_window(
        init_time=init_time,
        base_step_hours=6,
        default_steps=steps,
        now_utc=now_utc,
        future_hours=future_hours,
    )
    step_hours = [int(v * 6) for v in range(start_idx, end_idx + 1)]
    if step_hours:
        warnings.append(
            f"Forecast lead-time window requested: +{step_hours[0]}h to +{step_hours[-1]}h from cycle init."
        )
    params_primary = ["2t", "10u", "10v", "tp"]
    params_fallback = ["2t", "10u", "10v"]
    numbers = list(range(1, capped_members + 1))

    cache_key_base = hashlib.sha1(  # noqa: S324
        "|".join(
            [
                source,
                init_time.strftime("%Y%m%d%H"),
                ",".join(str(v) for v in step_hours),
                ",".join(str(v) for v in numbers),
            ]
        ).encode("utf-8")
    ).hexdigest()[:12]

    def _params_sig(params: list[str]) -> str:
        return "-".join(params).replace(",", "-")

    if use_cache:
        transient_paths: set[Path] = set()
    else:
        transient_base = output_dir / ".tmp_ecmwf"
        transient_base.mkdir(parents=True, exist_ok=True)
        transient_paths = set()

    last_err: Exception | None = None
    params_used: list[str] = params_primary
    pf_path: Path | None = None
    for params in [params_primary, params_fallback]:
        param_sig = _params_sig(params)
        if use_cache:
            pf_candidate = cache_dir / f"ecmwf_{init_time.strftime('%Y%m%dT%H')}Z_pf_{cache_key_base}_{param_sig}.grib2"
        else:
            pf_candidate = transient_base / (
                f"ecmwf_{init_time.strftime('%Y%m%dT%H')}Z_pf_{cache_key_base}_{param_sig}_{run_tag}.grib2"
            )
            transient_paths.add(pf_candidate)
        try:
            _download_ecmwf_grib(
                client,
                target=pf_candidate,
                use_cache=use_cache,
                warnings=warnings,
                cache_label="perturbed members",
                request={
                    "date": init_time.strftime("%Y%m%d"),
                    "time": init_time.hour,
                    "stream": "enfo",
                    "type": "pf",
                    "number": numbers,
                    "step": step_hours,
                    "param": params,
                },
            )
            pf_path = pf_candidate
            params_used = params
            if params != params_primary:
                warnings.append(
                    f"ECMWF request fallback used due to parameter availability: requested {params_primary}, using {params}."
                )
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    else:
        raise RuntimeError(f"ECMWF perturbed-member retrieval failed. Last error: {last_err}")

    assert pf_path is not None

    params_sig_used = _params_sig(params_used)
    if use_cache:
        cf_path = cache_dir / f"ecmwf_{init_time.strftime('%Y%m%dT%H')}Z_cf_{cache_key_base}_{params_sig_used}.grib2"
    else:
        cf_path = transient_base / (
            f"ecmwf_{init_time.strftime('%Y%m%dT%H')}Z_cf_{cache_key_base}_{params_sig_used}_{run_tag}.grib2"
        )
        transient_paths.add(cf_path)

    control_available = True
    try:
        _download_ecmwf_grib(
            client,
            target=cf_path,
            use_cache=use_cache,
            warnings=warnings,
            cache_label="control member",
            request={
                "date": init_time.strftime("%Y%m%d"),
                "time": init_time.hour,
                "stream": "enfo",
                "type": "cf",
                "step": step_hours,
                "param": params_used,
            },
        )
    except Exception as exc:  # noqa: BLE001
        control_available = False
        warnings.append(f"ECMWF control member retrieval failed; continuing with perturbed members only. ({exc})")

    try:
        pf_ds = _open_ecmwf_grib_as_dataset(pf_path, control_member=1)
        cf_ds = _open_ecmwf_grib_as_dataset(cf_path, control_member=0) if control_available and cf_path.exists() else xr.Dataset()
    finally:
        for transient_path in transient_paths:
            transient_path.unlink(missing_ok=True)

    merged: dict[str, xr.DataArray] = {}
    all_vars = set(pf_ds.data_vars) | set(cf_ds.data_vars)
    for var in sorted(all_vars):
        parts: list[xr.DataArray] = []
        if var in cf_ds:
            parts.append(cf_ds[var])
        if var in pf_ds:
            parts.append(pf_ds[var])
        if not parts:
            continue
        if len(parts) == 1:
            merged[var] = parts[0]
        else:
            merged[var] = xr.concat(parts, dim="ensemble")
        merged[var] = merged[var].sortby("ensemble")

    if not merged:
        raise RuntimeError("ECMWF retrieval succeeded but no usable variables were parsed.")

    ds = xr.Dataset(merged)
    ds = ds.expand_dims({"time": [np.datetime64(init_time.replace(tzinfo=None))]})
    ds.attrs.update({"mode": "forecast", "model_name": "ecmwf-ifs-ens", "provider": "ecmwf"})

    return ForecastRun(
        mode="forecast",
        model_name="ecmwf-ifs-ens",
        dataset=_save_dataset(ds, out_path),
        output_path=out_path,
        init_time=init_time,
        member_dim="ensemble",
        warnings=warnings,
    )


def _patch_s3fs_set_session_bug() -> None:
    """Work around s3fs regression where http_session._sessions can be None.

    Seen with newer s3fs/fsspec combos while using earth2studio GFS AWS source.
    """
    try:
        import s3fs
    except ImportError:
        return

    if getattr(s3fs.S3FileSystem, "_whether_safe_set_session", False):
        return

    original = s3fs.S3FileSystem.set_session

    async def safe_set_session(self, refresh=False, kwargs={}):  # noqa: B006
        if self._s3 is not None and not refresh:
            hsess = getattr(getattr(self._s3, "_endpoint", None), "http_session", None)
            if hsess is not None:
                sessions = getattr(hsess, "_sessions", None)
                # Newer stacks can expose None here; force a clean refresh.
                if sessions is None or not isinstance(sessions, dict):
                    refresh = True
        return await original(self, refresh=refresh, kwargs=kwargs)

    s3fs.S3FileSystem.set_session = safe_set_session
    s3fs.S3FileSystem._whether_safe_set_session = True


def run_global_ensemble(
    *,
    lat: float,
    lon: float,
    output_dir: Path,
    model_name: str,
    steps: int,
    members: int,
    diagnostics: bool,
    time: str | None,
    dry_run: bool,
    now_utc: datetime | None = None,
    future_hours: int | None = None,
) -> ForecastRun:
    if steps < 1:
        raise ValueError("--steps must be >= 1.")
    if members < 1:
        raise ValueError("--members must be >= 1.")

    requested_init_time = _init_time_or_default(time, cycle_hours=6)
    run_tag = f"r{os.getpid()}_{pytime.time_ns()}"
    import http.client

    warnings: list[str] = []
    model_name = model_name.lower()
    if model_name not in {"dlwp", "fcn", "fcn3"}:
        raise ValueError("--model must be 'dlwp', 'fcn', or 'fcn3'.")
    if diagnostics and model_name not in {"fcn", "fcn3"}:
        raise ValueError("--diagnostics is currently supported only with --model fcn or --model fcn3.")

    if dry_run:
        _, dryrun_end_idx = _compute_step_window(
            init_time=requested_init_time,
            base_step_hours=6,
            default_steps=steps,
            now_utc=now_utc,
            future_hours=future_hours,
        )
        out_path = _output_path(output_dir, "forecast", requested_init_time, run_tag=run_tag)
        ds = _mock_global_dataset(
            lat=lat,
            lon=lon,
            model_name=model_name,
            steps=dryrun_end_idx,
            members=members,
            init_time=requested_init_time,
        )
        if diagnostics:
            ds["tp"] = np.maximum(0.0, ds["t2m"] * 0.03).astype(np.float32)
        return ForecastRun(
            mode="forecast",
            model_name=model_name,
            dataset=_save_dataset(ds, out_path),
            output_path=out_path,
            init_time=requested_init_time,
            member_dim="ensemble",
            warnings=warnings,
        )

    try:
        from earth2studio.data import GFS
        from earth2studio.io import ZarrBackend
        from earth2studio.models.px import DLWP, FCN
        from earth2studio.perturbation import SphericalGaussian
        from earth2studio.run import ensemble
    except ImportError as exc:
        raise RuntimeError(
            "Missing Earth-2 dependencies for weatherforecast. Install with: pip install 'earth2studio[dlwp]' "
            "or 'earth2studio[fcn]' (or --extra fcn3 for FCN3). Or run with --dry-run."
        ) from exc

    if model_name == "dlwp":
        model_cls = DLWP
    elif model_name == "fcn":
        model_cls = FCN
    else:
        try:
            from earth2studio.models.px import FCN3
        except ImportError as exc:
            raise RuntimeError(
                "FCN3 support requires extras. Install with: uv add earth2studio --extra fcn3"
            ) from exc
        model_cls = FCN3

    base_package = model_cls.load_default_package()
    try:
        base_model = model_cls.load_model(base_package)
    except Exception as exc:  # noqa: BLE001
        message = str(exc).lower()
        if model_name == "fcn" and (
            "requested public model package" in message
            or "not found" in message
            or isinstance(exc, http.client.HTTPException)
        ):
            raise RuntimeError(
                "Legacy FCN checkpoint could not be resolved from NGC. "
                "Use --model fcn3 and install extras with: uv add earth2studio --extra fcn3 --extra perturbation"
            ) from exc
        raise

    if diagnostics:
        try:
            from earth2studio.models.dx import PrecipitationAFNO
            from earth2studio.models.px import DiagnosticWrapper
        except ImportError as exc:
            raise RuntimeError(
                "Diagnostics require PrecipitationAFNO. Install with: uv add earth2studio --extra fcn (or --extra fcn3)."
            ) from exc

        dx_package = PrecipitationAFNO.load_default_package()
        dx_model = PrecipitationAFNO.load_model(dx_package)
        model = DiagnosticWrapper(base_model, dx_model)
        requested_vars = ["t2m", "u10m", "v10m", "sp", "tcwv", "tp"]
    else:
        model = base_model
        if model_name == "dlwp":
            requested_vars = ["t2m", "tcwv", "z500", "t850"]
        else:
            requested_vars = ["t2m", "u10m", "v10m", "sp", "tcwv"]

    output_vars = _select_output_variables(model, requested_vars)
    output_coords = {"variable": output_vars}

    _patch_s3fs_set_session_bug()
    perturb = SphericalGaussian(noise_amplitude=0.15 if model_name == "dlwp" else 0.1)

    max_retries = 3
    last_exc: Exception | None = None

    for retry in range(max_retries + 1):
        init_time = requested_init_time - timedelta(hours=6 * retry)
        _, model_nsteps = _compute_step_window(
            init_time=init_time,
            base_step_hours=6,
            default_steps=steps,
            now_utc=now_utc,
            future_hours=future_hours,
        )
        out_path = _output_path(output_dir, "forecast", init_time, run_tag=run_tag)
        io = ZarrBackend(
            file_name=str(out_path),
            chunks={"ensemble": 1, "time": 1, "lead_time": 1},
            backend_kwargs={"overwrite": True},
        )
        data = GFS(source="aws", cache=True, verbose=True)
        try:
            io = ensemble(
                [init_time],
                model_nsteps,
                members,
                model,
                data,
                io,
                perturb,
                batch_size=min(members, 4),
                output_coords=output_coords,
            )
            ds = xr.open_zarr(out_path)

            if diagnostics and "tp" not in ds.data_vars:
                warnings.append(
                    "Diagnostics requested, but 'tp' is not present in outputs. "
                    "Check installed earth2studio version and PrecipitationAFNO compatibility."
                )

            if retry > 0:
                warnings.append(
                    f"GFS cycle fallback used: requested {requested_init_time.isoformat()} "
                    f"but ran {init_time.isoformat()} after {retry} retry/retries."
                )

            return ForecastRun(
                mode="forecast",
                model_name=model_name,
                dataset=ds,
                output_path=out_path,
                init_time=init_time,
                member_dim="ensemble",
                warnings=warnings,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if retry < max_retries:
                warnings.append(
                    f"Forecast init cycle {init_time.isoformat()} failed. "
                    f"Retrying previous cycle (-6h)."
                )
                continue
            break

    raise RuntimeError(
        "weatherforecast failed after GFS fallback retries. "
        f"Last error: {last_exc}"
    )


def run_stormcast(
    *,
    lat: float,
    lon: float,
    output_dir: Path,
    steps: int,
    members: int,
    deterministic_mode: bool,
    time: str | None,
    dry_run: bool,
) -> ForecastRun:
    if steps < 1:
        raise ValueError("--steps must be >= 1.")
    if members < 1 and not deterministic_mode:
        raise ValueError("--members must be >= 1.")

    init_time = _init_time_or_default(time, cycle_hours=1)
    run_tag = f"r{os.getpid()}_{pytime.time_ns()}"
    out_path = _output_path(output_dir, "storm", init_time, run_tag=run_tag)

    if deterministic_mode:
        members = 1

    warnings: list[str] = []

    if dry_run:
        ds = _mock_storm_dataset(lat=lat, lon=lon, steps=steps, members=members, init_time=init_time)
        return ForecastRun(
            mode="storm",
            model_name="stormcast",
            dataset=_save_dataset(ds, out_path),
            output_path=out_path,
            init_time=init_time,
            member_dim="ensemble",
            warnings=warnings,
        )

    try:
        from earth2studio.data import HRRR
        from earth2studio.io import ZarrBackend
        from earth2studio.models.px import StormCast
        from earth2studio.perturbation import SphericalGaussian
        from earth2studio.run import deterministic, ensemble

        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Missing StormCast dependencies. Install with: pip install 'earth2studio[stormcast]' pyproj. "
            "Or run with --dry-run."
        ) from exc

    if not torch.cuda.is_available():
        warnings.append("StormCast on CPU can be very slow; use a CUDA GPU when possible.")

    package = StormCast.load_default_package()
    model = StormCast.load_model(package)
    data = HRRR()
    io = ZarrBackend(file_name=str(out_path), backend_kwargs={"overwrite": True})

    if deterministic_mode:
        io = deterministic([init_time], steps, model, data, io)
    else:
        perturb = SphericalGaussian(noise_amplitude=0.05)
        io = ensemble([init_time], steps, members, model, data, io, perturb, batch_size=1)

    ds = xr.open_zarr(out_path)
    return ForecastRun(
        mode="storm",
        model_name="stormcast",
        dataset=ds,
        output_path=out_path,
        init_time=init_time,
        member_dim="ensemble",
        warnings=warnings,
    )


def run_corrdiff(
    *,
    lat: float,
    lon: float,
    output_dir: Path,
    samples: int,
    time: str | None,
    dry_run: bool,
) -> ForecastRun:
    if samples < 1:
        raise ValueError("--samples must be >= 1.")

    init_time = _init_time_or_default(time, cycle_hours=6)
    run_tag = f"r{os.getpid()}_{pytime.time_ns()}"
    out_path = _output_path(output_dir, "zoom", init_time, run_tag=run_tag)

    warnings: list[str] = []

    if dry_run:
        ds = _mock_zoom_dataset(samples=samples, init_time=init_time)
        return ForecastRun(
            mode="zoom",
            model_name="corrdiff-taiwan",
            dataset=_save_dataset(ds, out_path),
            output_path=out_path,
            init_time=init_time,
            member_dim="sample",
            warnings=warnings,
        )

    raise RuntimeError(
        "CorrDiff workflow requires environment-specific Earth-2 setup and is scaffolded with --dry-run support in this initial version. "
        "Use --dry-run now, or install earth2studio[corrdiff] and wire the custom inference loop in forecast.run_corrdiff."
    )


__all__ = [
    "ForecastRun",
    "run_global_ensemble",
    "run_ecmwf_ensemble",
    "run_stormcast",
    "run_corrdiff",
]
