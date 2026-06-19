"""Phase B4 — GPS -> local equirectangular projection."""
import math

import pytest

from app.infrastructure.geo import gps_to_local


def test_reference_point_maps_to_origin():
    x, y = gps_to_local(40.0, -74.0, 40.0, -74.0)
    assert x == pytest.approx(0.0, abs=1e-6)
    assert y == pytest.approx(0.0, abs=1e-6)


def test_north_offset_is_positive_y():
    x, y = gps_to_local(40.001, -74.0, 40.0, -74.0)
    assert x == pytest.approx(0.0, abs=1e-6)
    assert y == pytest.approx(111.19, abs=1.0)  # ~111 m per 0.001 deg latitude
    assert y > 0


def test_east_offset_is_positive_x_scaled_by_cos_lat():
    x, y = gps_to_local(40.0, -73.999, 40.0, -74.0)
    expected = math.radians(0.001) * math.cos(math.radians(40.0)) * 6_371_000.0
    assert x == pytest.approx(expected, abs=1e-3)
    assert y == pytest.approx(0.0, abs=1e-6)
    assert x > 0
