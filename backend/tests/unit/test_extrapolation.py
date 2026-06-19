"""Phase B3 — pixel-extrapolation ghosts (red EXTRAP, Path B)."""
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

pytest.importorskip("cv2")

from app.domain.entities import (
    PredictionMethod, Point3D, Velocity3D, WorldObject,
)


def _world_repo():
    from app.infrastructure.world_model_adapter import WorldModelRepositoryImpl
    config = Mock()
    config.get.return_value = {}
    config.get_int.return_value = 4
    config.get_float.return_value = 24.0  # fps and other floats
    config.get_list.return_value = []
    repo = WorldModelRepositoryImpl(config)
    repo._prediction_horizon = 5.0
    return repo


def _obj(**kw):
    base = dict(
        object_id=1, position=Point3D(0, 0, 3), velocity=Velocity3D(0, 0, 0),
        class_id=0, class_name="person", confidence=0.9,
        last_seen_camera=0, last_update=datetime.now(),
    )
    base.update(kw)
    return WorldObject(**base)


def _center(pred):
    b = pred.predicted_bbox
    return ((b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2)


def test_extrapolation_none_without_pixel_history():
    repo = _world_repo()
    obj = _obj()  # no camera_pixel_positions for cam 1
    assert repo._try_extrapolation_prediction(1, obj, 0.2) is None


def test_extrapolation_moves_in_velocity_direction():
    repo = _world_repo()
    obj = _obj(
        camera_pixel_positions={1: (100.0, 100.0)},
        camera_pixel_velocities={1: (5.0, 0.0)},  # moving +x
    )
    pred = repo._try_extrapolation_prediction(1, obj, time_since_seen=0.2)
    assert pred is not None
    assert pred.prediction_method == PredictionMethod.EXTRAPOLATION
    cx, cy = _center(pred)
    assert cx > 100.0
    assert cy == pytest.approx(100.0, abs=1e-6)


def test_extrapolation_displacement_capped_by_budget():
    repo = _world_repo()
    obj = _obj(
        camera_pixel_positions={1: (100.0, 100.0)},
        camera_pixel_velocities={1: (100000.0, 0.0)},  # absurd speed
    )
    pred = repo._try_extrapolation_prediction(1, obj, time_since_seen=0.2)
    cx, _ = _center(pred)
    budget = min(250.0, 80.0 + 40.0 * 0.2)
    assert cx - 100.0 <= budget + 1e-6


def test_zero_velocity_stays_at_last_pixel():
    repo = _world_repo()
    obj = _obj(
        camera_pixel_positions={1: (140.0, 160.0)},
        camera_pixel_velocities={1: (0.0, 0.0)},
    )
    cx, cy = _center(repo._try_extrapolation_prediction(1, obj, 1.0))
    assert (cx, cy) == pytest.approx((140.0, 160.0))


def test_generate_predictions_emits_extrapolation_for_lost_camera():
    repo = _world_repo()
    now = datetime.now()
    repo._world_objects[1] = _obj(
        last_update=now,                       # object still fresh (seen by someone now)
        source_tracks={1: 9},
        camera_pixel_positions={1: (100.0, 100.0)},
        camera_pixel_velocities={1: (5.0, 0.0)},
        camera_last_seen={1: now - timedelta(seconds=100)},  # cam 1 lost it long ago
    )
    methods = [p.prediction_method for p in repo.generate_predictions(camera_id=1)]
    assert PredictionMethod.EXTRAPOLATION in methods


def test_live_camera_is_skipped():
    repo = _world_repo()
    now = datetime.now()
    repo._world_objects[1] = _obj(
        last_update=now,
        camera_pixel_positions={1: (100.0, 100.0)},
        camera_last_seen={1: now},  # seen this instant -> live -> no ghost for cam 1
    )
    assert repo.generate_predictions(camera_id=1) == []
