from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .units import is_temperature_variable, precip_to_mm, wind_to_mph


@dataclass(frozen=True)
class CalibrationBand:
    variable: str
    spread_value: float
    spread_units: str
    sigma_value: float
    sigma_units: str
    ratio_sigma: float
    confidence_label: str
    confidence_guidance: str


_SIGMA_REF = {
    "t2m": (2.5, "C"),
    "t850": (3.0, "C"),
    "z500": (60.0, "gpm"),
    "tp": (2.0, "mm"),
    "sf": (2.0, "mm"),
    "sd": (8.0, "mm"),
    "u10m": (5.0, "mph"),
    "v10m": (5.0, "mph"),
    "refc": (8.0, "dBZ"),
    "sp": (1.5, "hPa"),
    "msl": (1.5, "hPa"),
}


def _band_from_ratio(ratio_sigma: float) -> tuple[str, str]:
    if ratio_sigma < 1.0:
        return ("high confidence", "Use confident language and emphasize agreement across members.")
    if ratio_sigma <= 2.0:
        return ("moderate confidence, notable uncertainty", "Use hedged language and describe a bounded range.")
    return ("low confidence, highly uncertain", "Explicitly state disagreement and discuss plausible scenarios.")


def _normalize_spread(var_name: str, spread: np.ndarray, units: str) -> tuple[np.ndarray, str]:
    if is_temperature_variable(var_name):
        normalized = units.strip().lower()
        if normalized in {"f", "degf", "fahrenheit", "degree_fahrenheit"}:
            return spread * (5.0 / 9.0), "C"
        # Kelvin and Celsius deltas have the same magnitude for spread.
        return spread, "C"
    if var_name in {"tp", "sf", "sd"}:
        values_mm, out_unit = precip_to_mm(spread, units=units)
        return values_mm, out_unit
    if var_name in {"u10m", "v10m"}:
        values_mph, out_unit = wind_to_mph(spread, units=units)
        return values_mph, out_unit
    if var_name in {"sp", "msl"}:
        normalized = units.strip().lower()
        if normalized in {"pa", "pascal", "pascals"}:
            return spread / 100.0, "hPa"
        return spread, units or "unknown"
    return spread, units or "unknown"


def calibrate_uncertainty_band(var_name: str, spread_values: list[float], units: str) -> CalibrationBand | None:
    if not spread_values:
        return None
    if var_name not in _SIGMA_REF:
        return None

    spread = np.asarray(spread_values, dtype=float)
    spread = spread[np.isfinite(spread)]
    if spread.size == 0:
        return None

    spread_norm, spread_unit = _normalize_spread(var_name, spread, units)
    spread_norm = spread_norm[np.isfinite(spread_norm)]
    if spread_norm.size == 0:
        return None

    spread_typical = float(np.nanmean(spread_norm))
    sigma_value, sigma_unit = _SIGMA_REF[var_name]

    if sigma_value <= 0.0:
        return None

    ratio = spread_typical / sigma_value
    label, guidance = _band_from_ratio(ratio)
    return CalibrationBand(
        variable=var_name,
        spread_value=spread_typical,
        spread_units=spread_unit,
        sigma_value=float(sigma_value),
        sigma_units=sigma_unit,
        ratio_sigma=float(ratio),
        confidence_label=label,
        confidence_guidance=guidance,
    )


__all__ = ["CalibrationBand", "calibrate_uncertainty_band"]
