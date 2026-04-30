"""Tests for CoordinateTransformer."""
from unittest.mock import Mock

import pytest

from app.domain.entities import CameraCalibration, Point3D
from app.infrastructure.world_model_adapter import CoordinateTransformer


def _make_transformer() -> CoordinateTransformer:
    config = Mock()
    return CoordinateTransformer(config)


def test_pixel_to_world_returns_none_for_uncalibrated_camera():
    t = _make_transformer()
    result = t.pixel_to_world(camera_id=99, pixel=(640, 360), depth=5.0)
    assert result is None


def test_pixel_to_world_at_image_center_yields_forward_ray(calibration_origin):
    """At image center with identity rotation and origin camera,
    the world point lies at (0, 0, depth) along +Z."""
    t = _make_transformer()
    t.set_calibration(calibration_origin)
    p = t.pixel_to_world(0, calibration_origin.image_center, depth=5.0)
    assert p is not None
    assert p.x == pytest.approx(0.0, abs=1e-9)
    assert p.y == pytest.approx(0.0, abs=1e-9)
    assert p.z == pytest.approx(5.0)


def test_set_calibration_stores_rotation_matrix(calibration_origin):
    t = _make_transformer()
    t.set_calibration(calibration_origin)
    assert calibration_origin.camera_id in t._calibrations
    assert calibration_origin.camera_id in t._rotation_cache


def test_pixel_to_world_offset_pixel_yields_offset_world(calibration_origin):
    """Offsetting the pixel in +x produces a world point with positive x."""
    t = _make_transformer()
    t.set_calibration(calibration_origin)
    cx, cy = calibration_origin.image_center
    p = t.pixel_to_world(0, (cx + 100, cy), depth=5.0)
    assert p is not None
    assert p.x > 0.0
    assert p.y == pytest.approx(0.0, abs=1e-9)
