from __future__ import annotations

from dataclasses import dataclass

from geopy.exc import GeocoderServiceError
from geopy.geocoders import Nominatim
import numpy as np


@dataclass(frozen=True)
class ResolvedLocation:
    query: str
    latitude: float
    longitude: float
    label: str


def normalize_lon_180(lon: float | np.ndarray) -> float | np.ndarray:
    """Normalize longitude into [-180, 180). Supports scalars and ndarrays."""
    wrapped = ((lon + 180.0) % 360.0) - 180.0
    return wrapped


def normalize_lon_360(lon: float) -> float:
    """Normalize longitude into [0, 360)."""
    return lon % 360.0


def snap_to_grid(lat: float, lon: float, resolution: float = 0.25) -> tuple[float, float]:
    """Snap to nearest regular lat/lon grid point."""
    lat_snapped = round(lat / resolution) * resolution
    lon_snapped = round(lon / resolution) * resolution
    return lat_snapped, float(normalize_lon_180(lon_snapped))


def resolve_location(query: str, timeout: int = 8) -> ResolvedLocation:
    """Resolve a human-readable location to coordinates via Nominatim."""
    geolocator = Nominatim(user_agent="weathernarrate")
    try:
        loc = geolocator.geocode(query, exactly_one=True, timeout=timeout)
    except GeocoderServiceError as exc:
        raise ValueError(f"Geocoding service failed: {exc}") from exc

    if not loc:
        raise ValueError(
            f"Could not geocode '{query}'. Try a more specific place name or use --lat/--lon."
        )

    return ResolvedLocation(
        query=query,
        latitude=float(loc.latitude),
        longitude=normalize_lon_180(float(loc.longitude)),
        label=str(loc.address),
    )


def validate_stormcast_domain(lat: float, lon: float) -> None:
    """Validate approximate StormCast Central-US domain."""
    lon180 = normalize_lon_180(lon)
    if not (25.0 <= lat <= 50.0 and -110.0 <= lon180 <= -85.0):
        raise ValueError(
            f"Location ({lat:.2f}, {lon180:.2f}) is outside StormCast's Central US domain. "
            "Use 'weatherforecast' for global coverage."
        )


def validate_corrdiff_domain(lat: float, lon: float) -> None:
    """Validate approximate CorrDiff Taiwan domain."""
    lon180 = normalize_lon_180(lon)
    if not (21.0 <= lat <= 26.5 and 118.5 <= lon180 <= 123.5):
        raise ValueError(
            f"Location ({lat:.2f}, {lon180:.2f}) is outside CorrDiff Taiwan domain. "
            "Use 'weatherforecast' for global coverage."
        )


def resolve_lat_lon(
    query: str | None,
    lat: float | None,
    lon: float | None,
) -> tuple[float, float, str]:
    """Resolve coordinates from CLI args (explicit coords override geocoding)."""
    if lat is not None and lon is not None:
        return float(lat), normalize_lon_180(float(lon)), "user-specified coordinates"

    if query is None:
        raise ValueError("Provide a location string or --lat and --lon.")

    resolved = resolve_location(query)
    return resolved.latitude, resolved.longitude, resolved.label
