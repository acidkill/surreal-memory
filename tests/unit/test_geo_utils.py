"""U8: utils/geo.py — haversine + GeoPoint/GeoFilter validation + parsing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from surreal_memory.utils.geo import (
    MAX_RADIUS_M,
    GeoFilter,
    GeoPoint,
    fiber_location,
    haversine_m,
    location_to_metadata,
    parse_geo_filter,
    parse_geo_point,
)

_OSLO = GeoPoint(59.9139, 10.7522, "Oslo")
_BERGEN = GeoPoint(60.3913, 5.3221, "Bergen")


class TestHaversine:
    def test_oslo_to_bergen_about_305km(self) -> None:
        d = haversine_m(_OSLO, _BERGEN)
        assert abs(d - 305_000) / 305_000 < 0.01  # within 1% of ~305 km

    def test_same_point_is_zero(self) -> None:
        assert haversine_m(_OSLO, _OSLO) == pytest.approx(0.0, abs=1e-6)

    def test_symmetric(self) -> None:
        assert haversine_m(_OSLO, _BERGEN) == pytest.approx(haversine_m(_BERGEN, _OSLO))

    def test_antimeridian_is_short(self) -> None:
        # 0.2° of longitude at the equator across the antimeridian ≈ 22 km, NOT half the globe.
        a = GeoPoint(0.0, 179.9)
        b = GeoPoint(0.0, -179.9)
        assert haversine_m(a, b) < 30_000

    def test_poles_coincide_regardless_of_lon(self) -> None:
        assert haversine_m(GeoPoint(90.0, 0.0), GeoPoint(90.0, 123.0)) == pytest.approx(
            0.0, abs=1.0
        )


class TestGeoPointValidation:
    @pytest.mark.parametrize("lat", [-90.0, 0.0, 90.0])
    def test_valid_lat(self, lat: float) -> None:
        assert GeoPoint(lat, 0.0).lat == lat

    @pytest.mark.parametrize("lat", [-90.001, 90.001, 100.0])
    def test_invalid_lat(self, lat: float) -> None:
        with pytest.raises(ValueError, match="lat out of range"):
            GeoPoint(lat, 0.0)

    @pytest.mark.parametrize("lon", [-180.001, 180.001, 360.0])
    def test_invalid_lon(self, lon: float) -> None:
        with pytest.raises(ValueError, match="lon out of range"):
            GeoPoint(0.0, lon)


class TestGeoFilter:
    def test_radius_validation(self) -> None:
        with pytest.raises(ValueError, match="radius_m out of range"):
            GeoFilter(_OSLO, 0.0)
        with pytest.raises(ValueError, match="radius_m out of range"):
            GeoFilter(_OSLO, MAX_RADIUS_M + 1)

    def test_contains(self) -> None:
        f = GeoFilter(_OSLO, 50_000)  # 50 km around Oslo
        assert f.contains(GeoPoint(59.95, 10.80)) is True  # a few km away
        assert f.contains(_BERGEN) is False  # ~305 km away


class TestParsing:
    def test_parse_geo_point(self) -> None:
        p = parse_geo_point({"lat": 59.9139, "lon": 10.7522, "label": "Oslo"})
        assert (p.lat, p.lon, p.label) == (59.9139, 10.7522, "Oslo")

    def test_parse_geo_point_string_coords(self) -> None:
        p = parse_geo_point({"lat": "59.9", "lon": "10.7"})
        assert p.lat == 59.9 and p.lon == 10.7 and p.label is None

    @pytest.mark.parametrize("bad", [None, [], "x", {"lat": 1}, {"lat": "x", "lon": 2}])
    def test_parse_geo_point_errors(self, bad: object) -> None:
        with pytest.raises(ValueError):
            parse_geo_point(bad)

    def test_parse_geo_filter(self) -> None:
        f = parse_geo_filter({"lat": 59.9, "lon": 10.7, "radius_m": 1000})
        assert f.radius_m == 1000 and f.center.lat == 59.9

    @pytest.mark.parametrize("bad", [{"lat": 1, "lon": 2}, {"lat": 1, "lon": 2, "radius_m": "x"}])
    def test_parse_geo_filter_errors(self, bad: object) -> None:
        with pytest.raises(ValueError):
            parse_geo_filter(bad)


class TestFiberLocation:
    def _fiber(self, metadata: object) -> object:
        return SimpleNamespace(metadata=metadata)

    def test_valid_location(self) -> None:
        p = fiber_location(self._fiber({"location": {"lat": 59.9, "lon": 10.7}}))
        assert p is not None and p.lat == 59.9

    @pytest.mark.parametrize(
        "meta",
        [
            None,
            "not-a-dict",
            {},
            {"location": None},
            {"location": "x"},
            {"location": {"lat": 59.9}},  # missing lon
            {"location": {"lat": 999, "lon": 10}},  # out of range
            {"location": {"lat": "x", "lon": "y"}},
        ],
    )
    def test_garbage_returns_none(self, meta: object) -> None:
        assert fiber_location(self._fiber(meta)) is None


def test_location_to_metadata_roundtrips() -> None:
    md = location_to_metadata(_OSLO)
    assert md == {"lat": 59.9139, "lon": 10.7522, "label": "Oslo"}
    assert location_to_metadata(GeoPoint(1.0, 2.0)) == {"lat": 1.0, "lon": 2.0}
    assert parse_geo_point(md).label == "Oslo"
