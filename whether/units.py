from __future__ import annotations

import re

import numpy as np

MM_PER_METER = 1000.0
MM_PER_INCH = 25.4
MPH_PER_MPS = 2.2369362920544


def is_temperature_variable(name: str) -> bool:
    if name in {"tp", "tcwv"}:
        return False
    if name == "t2m":
        return True
    return re.match(r"^t\d+", name) is not None


def is_kelvin(units: str, values: np.ndarray, var_name: str) -> bool:
    normalized = units.strip().lower()
    if normalized in {"k", "kelvin", "degk", "degree_kelvin"}:
        return True
    if is_temperature_variable(var_name) and values.size:
        return float(np.nanmean(values)) > 150.0
    return False


def to_celsius_if_needed(values: np.ndarray, *, units: str, var_name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if is_kelvin(units, arr, var_name):
        return arr - 273.15
    return arr


def precip_to_mm(values: np.ndarray, *, units: str) -> tuple[np.ndarray, str]:
    arr = np.asarray(values, dtype=float)
    normalized = units.strip().lower()
    compact = normalized.replace(" ", "")

    if normalized in {"m", "meter", "metre"} or "water equivalent" in normalized:
        return arr * MM_PER_METER, "mm"
    if "kg" in compact and "m" in compact and "-2" in compact:
        # 1 kg m^-2 liquid water equivalent == 1 mm precipitation depth.
        return arr, "mm"
    if normalized in {"mm", "millimeter", "millimetre"}:
        return arr, "mm"
    if normalized in {"in", "inch", "inches"}:
        return arr * MM_PER_INCH, "mm"
    if not normalized:
        # Forecast dry-run and some model outputs omit units; treat tp as mm by convention.
        return arr, "mm"
    return arr, units or "unknown"


def mm_to_inches(values_mm: np.ndarray) -> np.ndarray:
    arr = np.asarray(values_mm, dtype=float)
    return arr / MM_PER_INCH


def wind_to_mph(values: np.ndarray, *, units: str) -> tuple[np.ndarray, str]:
    arr = np.asarray(values, dtype=float)
    normalized = units.strip().lower()
    compact = normalized.replace(" ", "")

    if normalized in {"mph", "mi/h", "mile per hour", "miles per hour"}:
        return arr, "mph"
    if "m/s" in compact or "ms-1" in compact or "m s**-1" in normalized or "m s^-1" in normalized:
        return arr * MPH_PER_MPS, "mph"
    if "km/h" in compact or "kph" in compact:
        return arr * 0.6213711922, "mph"
    if "knot" in normalized or normalized in {"kt", "kts"}:
        return arr * 1.150779448, "mph"
    if not normalized:
        # Most model wind components are m/s even if attrs are absent.
        return arr * MPH_PER_MPS, "mph"
    return arr, units or "unknown"


__all__ = [
    "is_temperature_variable",
    "is_kelvin",
    "to_celsius_if_needed",
    "precip_to_mm",
    "mm_to_inches",
    "wind_to_mph",
]
