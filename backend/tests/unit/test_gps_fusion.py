"""Phase B4 — fuse mobile GPS/IMU sensor data into camera calibration."""
import math
from unittest.mock import Mock

import pytest

pytest.importorskip("cv2")


def _repo(reference=None):
    from app.infrastructure.world_model_adapter import WorldModelRepositoryImpl
    config = Mock()
    ref = {
        "gps_reference_lat": reference[0] if reference else None,
        "gps_reference_lng": reference[1] if reference else None,
    }
    config.get.side_effect = lambda k, d=None: ref.get(k, {} if d is None else d)
    config.get_int.return_value = 4
    config.get_float.return_value = 24.0
    config.get_list.return_value = []
    return WorldModelRepositoryImpl(config)


def test_first_gps_fix_becomes_local_origin():
    repo = _repo()
    repo.update_camera_sensor(0, gps={"latitude": 40.0, "longitude": -74.0, "altitude": 2.0})
    calib = repo.get_camera_calibration(0)
    assert calib is not None
    assert calib.position.x == pytest.approx(0.0, abs=1e-6)
    assert calib.position.y == pytest.approx(0.0, abs=1e-6)
    assert calib.position.z == pytest.approx(2.0)


def test_second_fix_offset_north_gives_positive_y():
    repo = _repo()
    repo.update_camera_sensor(0, gps={"latitude": 40.0, "longitude": -74.0})  # origin
    repo.update_camera_sensor(1, gps={"latitude": 40.001, "longitude": -74.0})
    c1 = repo.get_camera_calibration(1)
    assert c1.position.y > 100  # ~111 m north of the origin fix
    assert c1.position.x == pytest.approx(0.0, abs=1e-3)


def test_configured_reference_is_used():
    repo = _repo(reference=(40.0, -74.0))
    repo.update_camera_sensor(0, gps={"latitude": 40.001, "longitude": -74.0})
    assert repo.get_camera_calibration(0).position.y > 100


def test_orientation_sets_rotation_yaw():
    repo = _repo()
    repo.update_camera_sensor(0, orientation={"alpha": 90.0, "beta": 0.0, "gamma": 0.0})
    calib = repo.get_camera_calibration(0)
    assert calib is not None
    assert calib.rotation[2] == pytest.approx(math.radians(90.0))  # yaw from alpha


def test_no_sensor_data_is_noop():
    repo = _repo()
    repo.update_camera_sensor(0)
    assert repo.get_camera_calibration(0) is None
