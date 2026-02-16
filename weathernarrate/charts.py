from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import MaxNLocator

from .geocode import normalize_lon_180
from .timefmt import fmt_time_label
from .units import precip_to_mm, to_celsius_if_needed, wind_to_mph

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except Exception:  # noqa: BLE001
    ccrs = None
    cfeature = None
    HAS_CARTOPY = False


try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:  # noqa: BLE001
    pass

plt.rcParams.update(
    {
        "axes.facecolor": "#FFFFFF",
        "figure.facecolor": "#FFFFFF",
        "grid.alpha": 0.2,
        "grid.linestyle": "-",
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
    }
)


def _temp_to_c(values: list[float] | list[list[float]], *, units: str, var_name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return to_celsius_if_needed(arr, units=units, var_name=var_name)


def _timeline_hours(extracted: dict[str, Any]) -> np.ndarray:
    timeline = extracted.get("timeline", [])
    if not timeline:
        return np.array([], dtype=float)
    try:
        times = np.array([np.datetime64(str(t)) for t in timeline])
        t0 = times[0]
        return ((times - t0) / np.timedelta64(1, "h")).astype(float)
    except Exception:  # noqa: BLE001
        return np.arange(len(timeline), dtype=float)


def _parse_utc_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        if text.endswith("Z"):
            return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except ValueError:
        try:
            ts = np.datetime64(text).astype("datetime64[s]")
            return datetime.fromtimestamp(int(ts.astype(np.int64)), tz=UTC)
        except Exception:  # noqa: BLE001
            return None


def _now_marker_x(extracted: dict[str, Any]) -> float | None:
    timeline = [str(t) for t in extracted.get("timeline", [])]
    if not timeline:
        return None
    t0 = _parse_utc_datetime(timeline[0])
    now_dt = _parse_utc_datetime(extracted.get("current_time_utc"))
    if t0 is None or now_dt is None:
        return None
    return (now_dt - t0).total_seconds() / 3600.0


def _add_now_line(ax, *, now_x: float | None, x_values: np.ndarray) -> None:
    if now_x is None or not np.isfinite(now_x) or x_values.size == 0:
        return
    xmin = float(np.nanmin(x_values))
    xmax = float(np.nanmax(x_values))
    if now_x < (xmin - 0.25) or now_x > (xmax + 0.25):
        return
    ax.axvline(
        now_x,
        color="#374151",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
        label="now",
        zorder=4,
    )


def _ensure_output(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _plot_member_fan(
    ax,
    *,
    x: np.ndarray,
    members: np.ndarray,
    mean: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    color: str,
    line_label: str,
) -> None:
    if members.ndim == 2:
        for i in range(members.shape[1]):
            ax.plot(x, members[:, i], color=color, linewidth=0.9, alpha=0.18, zorder=1)
    ax.fill_between(x, low, high, alpha=0.22, color=color, label="ensemble range", zorder=2)
    ax.plot(x, mean, color=color, linewidth=2.1, label=line_label, zorder=3)


def _plot_missing_panel(ax, *, title: str, message: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def _valid_times_for_dataarray(da: xr.DataArray) -> np.ndarray | None:
    if "lead_time" not in da.coords or "time" not in da.coords:
        return None
    try:
        base = np.asarray(da.coords["time"].values).reshape(-1)[0]
        base64 = np.datetime64(base)
        lead = np.asarray(da.coords["lead_time"].values)
        if not np.issubdtype(lead.dtype, np.timedelta64):
            return None
        return base64 + lead
    except Exception:  # noqa: BLE001
        return None


def _choose_lead_index(da: xr.DataArray, target_valid_time: str | None) -> int:
    if "lead_time" not in da.dims:
        return 0
    if not target_valid_time:
        return 0
    try:
        target = np.datetime64(str(target_valid_time))
    except Exception:  # noqa: BLE001
        return 0
    valid_times = _valid_times_for_dataarray(da)
    if valid_times is None or valid_times.size == 0:
        return 0
    diff_hours = np.abs((valid_times - target) / np.timedelta64(1, "s")).astype(float)
    return int(np.argmin(diff_hours))


def _select_t2m_field(
    ds: xr.Dataset | None,
    *,
    target_valid_time: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str] | None:
    if ds is None:
        return None

    da: xr.DataArray | None = None
    if "t2m" in ds.data_vars:
        da = ds["t2m"]
    elif len(ds.data_vars) == 1:
        _, only = next(iter(ds.data_vars.items()))
        if "variable" in only.dims and "variable" in only.coords and "t2m" in set(map(str, only.coords["variable"].values)):
            da = only.sel(variable="t2m").drop_vars("variable", errors="ignore")

    if da is None:
        return None
    if "lat" not in da.coords or "lon" not in da.coords:
        return None
    if "lat" not in da.dims or "lon" not in da.dims:
        return None

    units = str(da.attrs.get("units", ""))
    if "time" in da.dims:
        da = da.isel(time=0)
    if "lead_time" in da.dims:
        lead_idx = _choose_lead_index(da, target_valid_time)
        da = da.isel(lead_time=lead_idx)
    if "ensemble" in da.dims:
        da = da.mean("ensemble")
    if "sample" in da.dims:
        da = da.mean("sample")

    da = da.squeeze(drop=True)
    if "lat" not in da.dims or "lon" not in da.dims:
        return None

    da = da.transpose("lat", "lon")
    lat_vals = np.asarray(da.coords["lat"].values, dtype=float)
    lon_vals = np.asarray(da.coords["lon"].values, dtype=float)
    field = np.asarray(da.values, dtype=float)

    if np.nanmax(lon_vals) > 180.0:
        lon_vals = ((lon_vals + 180.0) % 360.0) - 180.0

    lat_order = np.argsort(lat_vals)
    lon_order = np.argsort(lon_vals)
    lat_vals = lat_vals[lat_order]
    lon_vals = lon_vals[lon_order]
    field = field[np.ix_(lat_order, lon_order)]
    field_c = to_celsius_if_needed(field, units=units, var_name="t2m")
    return lat_vals, lon_vals, field_c, units


def _robust_color_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(finite, 5))
    vmax = float(np.nanpercentile(finite, 95))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return vmin, vmax


def _half_width_degrees_for_radius(lat_center: float, *, radius_km: float = 1300.0) -> tuple[float, float]:
    lat_half = radius_km / 111.0
    lon_half = radius_km / max(111.0 * max(np.cos(np.deg2rad(abs(lat_center))), 0.2), 1e-6)
    return min(lat_half, 40.0), min(lon_half, 60.0)


def _subset_regional_map(
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    field_c: np.ndarray,
    *,
    lat_center: float,
    lon_center: float,
    radius_km: float = 1300.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_half, lon_half = _half_width_degrees_for_radius(lat_center, radius_km=radius_km)
    lat_mask = np.abs(lat_vals - lat_center) <= lat_half
    lon_diff = ((lon_vals - lon_center + 180.0) % 360.0) - 180.0
    lon_mask = np.abs(lon_diff) <= lon_half

    if int(np.sum(lat_mask)) < 3 or int(np.sum(lon_mask)) < 3:
        return lat_vals, lon_vals, field_c

    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    lat_sub = lat_vals[lat_idx]
    lon_sub_display = lon_center + lon_diff[lon_idx]
    field_sub = field_c[np.ix_(lat_idx, lon_idx)]

    lon_order = np.argsort(lon_sub_display)
    lon_sub = lon_sub_display[lon_order]
    field_sub = field_sub[:, lon_order]
    return lat_sub, lon_sub, field_sub


def _precip_series_for_plot(stats: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    tp_units = str(stats.get("units", ""))
    members_raw = np.asarray(stats.get("members", []), dtype=float)
    members_mm, precip_unit = precip_to_mm(members_raw, units=tp_units)
    if members_mm.ndim != 2 or members_mm.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty, precip_unit

    y = np.clip(members_mm, a_min=0.0, a_max=None)
    y_label = f"Precipitation ({precip_unit})"
    if y.shape[0] > 1:
        monotonic_fraction = float(np.mean(np.diff(y, axis=0) >= -1e-6))
        if monotonic_fraction >= 0.9:
            step = np.diff(y, axis=0, prepend=y[0:1, :])
            step[0, :] = 0.0
            y = np.clip(step, a_min=0.0, a_max=None)
            y_label = f"6h precipitation ({precip_unit})"

    mean = np.mean(y, axis=1)
    low = np.min(y, axis=1)
    high = np.max(y, axis=1)
    return y, mean, low, high, y_label


def _plot_map_panel(
    ax,
    *,
    ds: xr.Dataset | None,
    lat: float | None,
    lon: float | None,
    valid_time_utc: str | None = None,
    valid_time_label: str | None = None,
) -> None:
    title = "Temperature Map (t2m, ~1300 km context)"
    if valid_time_label:
        title = f"{title}\nValid: {fmt_time_label(valid_time_label)}"
    if lat is None or lon is None:
        _plot_missing_panel(ax, title=title, message="Location unavailable.")
        return

    selected = _select_t2m_field(ds, target_valid_time=valid_time_utc)
    if selected is None:
        _plot_missing_panel(ax, title=title, message="t2m field unavailable.")
        return

    lat_vals, lon_vals, field_c, _ = selected
    lat_center = float(lat)
    lon_center = float(normalize_lon_180(float(lon)))
    lat_sub, lon_sub, field_sub = _subset_regional_map(
        lat_vals,
        lon_vals,
        field_c,
        lat_center=lat_center,
        lon_center=lon_center,
        radius_km=1300.0,
    )

    vmin, vmax = _robust_color_limits(field_sub)

    if HAS_CARTOPY:
        mesh = ax.pcolormesh(
            lon_sub,
            lat_sub,
            field_sub,
            shading="auto",
            cmap="RdYlBu_r",
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            zorder=0,
        )
        ax.scatter(
            [lon_center],
            [lat_center],
            s=52,
            c="#111827",
            marker="x",
            linewidths=1.8,
            label="requested point",
            transform=ccrs.PlateCarree(),
            zorder=4,
        )
        try:
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)
            if hasattr(cfeature, "STATES"):
                ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3, alpha=0.5)
        except Exception:  # noqa: BLE001
            # Fall back to plain map rendering if Natural Earth assets are unavailable.
            pass
        lon_pad = max((float(np.nanmax(lon_sub)) - float(np.nanmin(lon_sub))) * 0.03, 0.6)
        lat_pad = max((float(np.nanmax(lat_sub)) - float(np.nanmin(lat_sub))) * 0.03, 0.6)
        ax.set_extent(
            [
                float(np.nanmin(lon_sub)) - lon_pad,
                float(np.nanmax(lon_sub)) + lon_pad,
                float(np.nanmin(lat_sub)) - lat_pad,
                float(np.nanmax(lat_sub)) + lat_pad,
            ],
            crs=ccrs.PlateCarree(),
        )
        try:
            ax.set_aspect("auto")
        except Exception:  # noqa: BLE001
            pass
    else:
        mesh = ax.pcolormesh(lon_sub, lat_sub, field_sub, shading="auto", cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
        ax.scatter([lon_center], [lat_center], s=52, c="#111827", marker="x", linewidths=1.8, label="requested point")
        ax.set_xlim(float(np.nanmin(lon_sub)), float(np.nanmax(lon_sub)))
        ax.set_ylim(float(np.nanmin(lat_sub)), float(np.nanmax(lat_sub)))

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    cbar = ax.figure.colorbar(mesh, ax=ax, orientation="vertical", fraction=0.045, pad=0.02)
    cbar.set_label("Temperature (C)")


def _plot_forecast(extracted: dict[str, Any], output_dir: Path, *, dataset: xr.Dataset | None, lat: float | None, lon: float | None) -> list[Path]:
    x = _timeline_hours(extracted)
    now_x = _now_marker_x(extracted)
    vars_ = extracted["variables"]
    timezone_name = str(extracted.get("timezone", "UTC"))
    current_label = fmt_time_label(extracted.get("current_time_local") or extracted.get("current_time_utc"))
    start_label = fmt_time_label(extracted.get("forecast_start_local") or extracted.get("forecast_start_utc"))
    end_label = fmt_time_label(extracted.get("forecast_end_local") or extracted.get("forecast_end_utc"))
    lead_xlabel = f"Lead Time (h from {start_label})"

    fig = plt.figure(figsize=(14.5, 9.2))
    gs = fig.add_gridspec(2, 2, left=0.05, right=0.98, bottom=0.08, top=0.90, wspace=0.24, hspace=0.34)
    if HAS_CARTOPY:
        ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    else:
        ax_map = fig.add_subplot(gs[0, 0])
    ax_t2m = fig.add_subplot(gs[0, 1])
    ax_wind = fig.add_subplot(gs[1, 0])
    ax_tp = fig.add_subplot(gs[1, 1])

    _plot_map_panel(
        ax_map,
        ds=dataset,
        lat=lat,
        lon=lon,
        valid_time_utc=extracted.get("forecast_start_utc"),
        valid_time_label=extracted.get("forecast_start_local") or extracted.get("forecast_start_utc"),
    )

    if "t2m" in vars_:
        stats = vars_["t2m"]
        units = str(stats.get("units", ""))
        mean = _temp_to_c(stats["mean"], units=units, var_name="t2m")
        low = _temp_to_c(stats["min"], units=units, var_name="t2m")
        high = _temp_to_c(stats["max"], units=units, var_name="t2m")
        members = _temp_to_c(stats["members"], units=units, var_name="t2m")
        _plot_member_fan(
            ax_t2m,
            x=x,
            members=members,
            mean=mean,
            low=low,
            high=high,
            color="#0B5CAB",
            line_label="mean t2m",
        )
        ax_t2m.set_title("Temperature Ensemble")
        ax_t2m.set_xlabel(lead_xlabel)
        ax_t2m.set_ylabel("Temperature (C)")
        ax_t2m.grid(alpha=0.2)
        ax_t2m.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        _add_now_line(ax_t2m, now_x=now_x, x_values=x)
        ax_t2m.legend()
    else:
        _plot_missing_panel(ax_t2m, title="Temperature Ensemble", message="t2m unavailable.")

    if "u10m" in vars_ and "v10m" in vars_:
        u_units = str(vars_["u10m"].get("units", ""))
        v_units = str(vars_["v10m"].get("units", ""))
        u_members_raw = np.asarray(vars_["u10m"]["members"], dtype=float)
        v_members_raw = np.asarray(vars_["v10m"]["members"], dtype=float)
        u_members, wind_unit = wind_to_mph(u_members_raw, units=u_units)
        v_members, _ = wind_to_mph(v_members_raw, units=v_units)
        if u_members.shape == v_members.shape and u_members.ndim == 2:
            wind_members = np.sqrt((u_members**2) + (v_members**2))
            wind_mean = np.mean(wind_members, axis=1)
            wind_min = np.min(wind_members, axis=1)
            wind_max = np.max(wind_members, axis=1)
            _plot_member_fan(
                ax_wind,
                x=x,
                members=wind_members,
                mean=wind_mean,
                low=wind_min,
                high=wind_max,
                color="#C66A00",
                line_label="mean wind speed",
            )
            ax_wind.set_title("Wind Speed Ensemble")
            ax_wind.set_xlabel(lead_xlabel)
            ax_wind.set_ylabel(f"Wind Speed ({wind_unit})")
            ax_wind.grid(alpha=0.2)
            ax_wind.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
            wind_upper = float(np.nanmax(wind_max)) if wind_max.size else 1.0
            ax_wind.set_ylim(bottom=0.0, top=max(wind_upper * 1.08, 1.0))
            _add_now_line(ax_wind, now_x=now_x, x_values=x)
            ax_wind.legend()
        else:
            _plot_missing_panel(ax_wind, title="Wind Speed Ensemble", message="u10m/v10m shape mismatch.")
    else:
        _plot_missing_panel(ax_wind, title="Wind Speed Ensemble", message="u10m/v10m unavailable.")

    if "tp" in vars_:
        stats = vars_["tp"]
        members, mean, low, high, y_label = _precip_series_for_plot(stats)
        _plot_member_fan(
            ax_tp,
            x=x,
            members=members,
            mean=mean,
            low=low,
            high=high,
            color="#2E8B57",
            line_label="mean precipitation",
        )
        ax_tp.set_title("Precipitation Ensemble")
        ax_tp.set_xlabel(lead_xlabel)
        ax_tp.set_ylabel(y_label)
        ax_tp.grid(alpha=0.2)
        ax_tp.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        precip_upper = float(np.nanmax(high)) if high.size else 0.0
        ax_tp.set_ylim(bottom=0.0, top=max(precip_upper * 1.12, 0.2))
        _add_now_line(ax_tp, now_x=now_x, x_values=x)
        ax_tp.legend()
    else:
        _plot_missing_panel(ax_tp, title="Precipitation Ensemble", message="tp unavailable.")

    fig.suptitle(
        f"Current Time ({timezone_name}): {current_label} | Forecast Window: {start_label} -> {end_label}",
        fontsize=11.5,
        y=0.965,
    )

    path = output_dir / "forecast_dashboard.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


def _plot_storm(extracted: dict[str, Any], output_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    x = _timeline_hours(extracted)
    vars_ = extracted["variables"]

    if "t2m" not in vars_ and "refc" not in vars_:
        return outputs

    fig, ax1 = plt.subplots(figsize=(10, 4.8))

    ax2 = None
    if "t2m" in vars_:
        t = vars_["t2m"]
        units = str(t.get("units", ""))
        t_mean = _temp_to_c(t["mean"], units=units, var_name="t2m")
        t_min = _temp_to_c(t["min"], units=units, var_name="t2m")
        t_max = _temp_to_c(t["max"], units=units, var_name="t2m")
        ax1.plot(x, t_mean, color="#C0392B", linewidth=2.0, label="t2m mean")
        ax1.fill_between(x, t_min, t_max, color="#C0392B", alpha=0.18, label="t2m range")
        ax1.set_ylabel("Temperature (C)", color="#C0392B")
        ax1.tick_params(axis="y", labelcolor="#C0392B")

    if "refc" in vars_:
        ax2 = ax1.twinx()
        r = vars_["refc"]
        ax2.bar(x, r["mean"], color="#1F77B4", alpha=0.35, label="refc mean")
        ax2.set_ylabel("Reflectivity (dBZ)", color="#1F77B4")
        ax2.tick_params(axis="y", labelcolor="#1F77B4")

    ax1.set_title("StormCast Hourly Signal")
    ax1.set_xlabel("Lead Time (hours)")
    ax1.grid(alpha=0.2)

    path = output_dir / "storm_t2m_refc.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    outputs.append(path)

    return outputs


def _plot_zoom(extracted: dict[str, Any], output_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    vars_ = extracted["variables"]

    sample_members = None
    if "tp" in vars_:
        sample_members = vars_["tp"]["members"]
        title = "tp"
    elif "t2m" in vars_:
        sample_members = vars_["t2m"]["members"]
        title = "t2m"
    if sample_members is None:
        return outputs

    arr = np.array(sample_members, dtype=float)
    if arr.ndim != 2:
        return outputs

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(np.arange(arr.shape[1]), arr[0], color="#6C5CE7")
    ax.set_title(f"CorrDiff Sample Spread at Point ({title})")
    ax.set_xlabel("Sample realization index")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.2, axis="y")

    path = output_dir / "zoom_sample_spread.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    outputs.append(path)

    return outputs


def generate_charts(
    extracted: dict[str, Any],
    output_dir: Path,
    mode: str,
    *,
    dataset: xr.Dataset | None = None,
    lat: float | None = None,
    lon: float | None = None,
) -> list[Path]:
    output_dir = _ensure_output(output_dir)
    if mode == "forecast":
        return _plot_forecast(extracted, output_dir, dataset=dataset, lat=lat, lon=lon)
    if mode == "storm":
        return _plot_storm(extracted, output_dir)
    if mode == "zoom":
        return _plot_zoom(extracted, output_dir)
    return []


__all__ = ["generate_charts"]
