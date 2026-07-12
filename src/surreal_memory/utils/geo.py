"""Geospatial helpers (U8, PR7) — stdlib-only, no schema change.

Coordinates live in ``fiber.metadata["location"] = {"lat", "lon", "label"?}`` (an
OBJECT FLEXIBLE metadata field — zero migration, sync/Merkle for free). Recall and
browse use these to HARD-filter fibers by distance (haversine on the WGS-84 sphere).
No external dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Mean Earth radius (IUGG), metres. Half the equatorial circumference is the largest
# meaningful search radius (any point on Earth is within it of any other).
_EARTH_RADIUS_M = 6_371_008.8
MAX_RADIUS_M = 20_015_087.0


@dataclass(frozen=True)
class GeoPoint:
    """A validated WGS-84 coordinate (degrees)."""

    lat: float
    lon: float
    label: str | None = None

    def __post_init__(self) -> None:
        if not -90.0 <= self.lat <= 90.0:
            raise ValueError(f"lat out of range [-90, 90]: {self.lat}")
        if not -180.0 <= self.lon <= 180.0:
            raise ValueError(f"lon out of range [-180, 180]: {self.lon}")


@dataclass(frozen=True)
class GeoFilter:
    """A center + radius (metres) hard filter."""

    center: GeoPoint
    radius_m: float

    def __post_init__(self) -> None:
        if not 0.0 < self.radius_m <= MAX_RADIUS_M:
            raise ValueError(f"radius_m out of range (0, {MAX_RADIUS_M}]: {self.radius_m}")

    def contains(self, point: GeoPoint) -> bool:
        return haversine_m(self.center, point) <= self.radius_m


def haversine_m(a: GeoPoint, b: GeoPoint) -> float:
    """Great-circle distance between two points, in metres (haversine)."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a.lat, a.lon, b.lat, b.lon))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2.0 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(h)))


def parse_geo_point(data: Any) -> GeoPoint:
    """Parse a {lat, lon, label?} dict into a GeoPoint. Raises ValueError on bad input."""
    if not isinstance(data, dict):
        raise ValueError("location must be an object with numeric lat and lon")
    try:
        lat = float(data["lat"])
        lon = float(data["lon"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"location needs numeric lat + lon: {exc}") from exc
    label = data.get("label")
    return GeoPoint(lat=lat, lon=lon, label=str(label) if label is not None else None)


def parse_geo_filter(data: Any) -> GeoFilter:
    """Parse a {lat, lon, radius_m, label?} dict into a GeoFilter. Raises ValueError."""
    if not isinstance(data, dict):
        raise ValueError("near must be an object with lat, lon and radius_m")
    center = parse_geo_point(data)
    try:
        radius_m = float(data["radius_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"near needs numeric radius_m: {exc}") from exc
    return GeoFilter(center=center, radius_m=radius_m)


def fiber_location(fiber: Any) -> GeoPoint | None:
    """Best-effort GeoPoint from ``fiber.metadata['location']`` — never raises.

    Tolerates missing metadata, a non-dict location, missing/non-numeric coords, and
    out-of-range values (returns None for all of them).
    """
    meta = getattr(fiber, "metadata", None)
    if not isinstance(meta, dict):
        return None
    loc = meta.get("location")
    if not isinstance(loc, dict):
        return None
    try:
        return parse_geo_point(loc)
    except ValueError:
        return None


def fiber_within(fiber: Any, geo_filter: GeoFilter) -> bool:
    """Whether a fiber's location is inside the geo filter — the shared hard-filter.

    A fiber with no (or malformed) location is NOT near anything, matching the
    ``valid_at`` precedent. Used by the recall pipeline and by browse pushdown so the
    exact-distance semantics are identical everywhere.
    """
    loc = fiber_location(fiber)
    return loc is not None and geo_filter.contains(loc)


def location_to_metadata(point: GeoPoint) -> dict[str, Any]:
    """Serialise a GeoPoint to the metadata dict shape stored on a fiber."""
    out: dict[str, Any] = {"lat": point.lat, "lon": point.lon}
    if point.label is not None:
        out["label"] = point.label
    return out
