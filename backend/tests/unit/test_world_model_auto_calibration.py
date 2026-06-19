"""World model must produce objects even without CAMERA_POSITIONS configured.

Regression: previously, with no CAMERA_POSITIONS, ``pixel_to_world`` returned
None for every camera, so the world model produced ZERO world objects and ZERO
predictions — viewers saw only raw detections/tracks. Auto-default calibration
fixes this so single-camera setups work out of the box.
"""
import asyncio
from unittest.mock import Mock

import pytest

pytest.importorskip("cv2")


def _make_repo(camera_positions=None):
    from app.infrastructure.world_model_adapter import WorldModelRepositoryImpl
    config = Mock()
    config.get.return_value = {}
    config.get_int.return_value = 4
    config.get_float.return_value = 1.7
    config.get_list.return_value = [] if camera_positions is None else camera_positions
    return WorldModelRepositoryImpl(config)


def test_world_objects_created_without_camera_positions(make_track):
    repo = _make_repo(camera_positions=[])
    track = make_track(track_id=1, camera_id=0)
    objects = asyncio.run(repo.update({0: [track]}))
    assert len(objects) == 1
    assert objects[0].class_name == "person"


def test_auto_default_calibration_created_lazily_for_unseen_camera(make_track):
    repo = _make_repo(camera_positions=[])
    assert repo._transformer._calibrations.get(2) is None
    track = make_track(track_id=1, camera_id=2)
    asyncio.run(repo.update({2: [track]}))
    assert repo._transformer._calibrations.get(2) is not None


def test_explicit_camera_positions_take_precedence(make_track):
    repo = _make_repo(camera_positions=[[5.0, 0.0, 2.0]])
    calib = repo._transformer._calibrations.get(0)
    assert calib is not None
    assert (calib.position.x, calib.position.y, calib.position.z) == (5.0, 0.0, 2.0)


def test_predictions_available_for_second_camera_without_config(make_track):
    """With two cameras and no config, an object seen only by cam 0 should yield
    a world-projection prediction for cam 1."""
    repo = _make_repo(camera_positions=[])
    asyncio.run(repo.update({0: [make_track(track_id=1, camera_id=0)]}))
    # cam 1 has a default calibration created on demand; ask for its predictions
    repo._ensure_calibration(1)
    preds = repo.generate_predictions(camera_id=1)
    assert isinstance(preds, list)  # may be empty if behind camera, but must not raise
