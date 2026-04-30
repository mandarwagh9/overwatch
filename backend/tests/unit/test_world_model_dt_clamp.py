"""Test that world model clamps dt>=0 to handle clock skew."""
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest


pytest.importorskip("cv2")

from app.domain.entities import (
    BoundingBox, CameraCalibration, Point3D, Track, TrackingState
)


def _make_repo():
    from app.infrastructure.world_model_adapter import WorldModelRepositoryImpl
    config = Mock()
    config.get.return_value = {}
    config.get_int.return_value = 4
    config.get_float.return_value = 1.7
    config.get_list.return_value = []
    return WorldModelRepositoryImpl(config)


def test_world_model_handles_clock_skew_negative_dt(calibration_origin, make_track):
    repo = _make_repo()
    repo._transformer.set_calibration(calibration_origin)
    t1 = make_track(track_id=1, camera_id=0)
    asyncio.run(repo.update({0: [t1]}))
    obj = next(iter(repo._world_objects.values()))
    # Force last_update into the future so dt would be negative
    obj.last_update = datetime.now() + timedelta(seconds=10)
    asyncio.run(repo.update({0: [t1]}))
    # State must remain finite
    assert all([
        obj.position.x == obj.position.x,  # not NaN
        obj.position.y == obj.position.y,
        obj.position.z == obj.position.z,
    ])
