"""Phase B5 — sensor trust scoring + adaptive Kalman by bbox area."""
from unittest.mock import Mock

import pytest

pytest.importorskip("cv2")

from app.domain.entities import Point3D
from app.infrastructure.world_model_adapter import KalmanFilter


def _repo():
    from app.infrastructure.world_model_adapter import WorldModelRepositoryImpl
    config = Mock()
    config.get.return_value = {}
    config.get_int.return_value = 4
    config.get_float.return_value = 1.0
    config.get_list.return_value = []
    repo = WorldModelRepositoryImpl(config)
    repo._trust_innovation_threshold = 1.0
    repo._bbox_reference_area = 40000.0
    return repo


# --------------------------------------------------------- adaptive Kalman

def test_update_returns_innovation_magnitude():
    kf = KalmanFilter()
    kf.state[0:3] = [0.0, 0.0, 0.0]
    inn = kf.update(Point3D(3.0, 4.0, 0.0))  # |(3,4,0)| = 5
    assert inn == pytest.approx(5.0, abs=1e-6)


def test_higher_area_factor_trusts_measurement_more():
    hi = KalmanFilter()
    hi.state[0:3] = [0, 0, 0]
    lo = KalmanFilter()
    lo.state[0:3] = [0, 0, 0]
    hi.update(Point3D(10, 0, 0), area_factor=1.0)
    lo.update(Point3D(10, 0, 0), area_factor=0.1)
    assert hi.position.x > lo.position.x


# ------------------------------------------------------------ sensor trust

def test_consistent_measurement_increases_trust():
    repo = _repo()
    repo._sensor_trust[0] = 0.5
    repo._update_sensor_trust(0, innovation=0.01)
    assert repo._sensor_trust[0] > 0.5


def test_outlier_decreases_trust():
    repo = _repo()
    repo._sensor_trust[0] = 0.5
    repo._update_sensor_trust(0, innovation=100.0)
    assert repo._sensor_trust[0] < 0.5


def test_trust_clamped_to_unit_interval():
    repo = _repo()
    repo._sensor_trust[0] = 1.0
    for _ in range(10):
        repo._update_sensor_trust(0, innovation=0.0)
    assert repo._sensor_trust[0] <= 1.0
    repo._sensor_trust[1] = 0.1
    for _ in range(10):
        repo._update_sensor_trust(1, innovation=1000.0)
    assert repo._sensor_trust[1] >= 0.1


def test_default_trust_is_one():
    assert _repo().get_sensor_trust(99) == 1.0


def test_bbox_area_factor_bounds():
    repo = _repo()
    assert repo._bbox_area_factor(0.0) == pytest.approx(0.1)
    assert repo._bbox_area_factor(1e9) == pytest.approx(1.0)
    assert 0.1 < repo._bbox_area_factor(20000.0) < 1.0


def test_trust_config_wired():
    from app.infrastructure.config_adapter import (
        PydanticConfigurationRepository, Settings,
    )
    repo = PydanticConfigurationRepository(
        Settings(sensor_trust_innovation_threshold=2.5, bbox_reference_area=10000.0)
    )
    assert repo.get_float("sensor_trust_innovation_threshold", 1.0) == 2.5
    assert repo.get_float("bbox_reference_area", 40000.0) == 10000.0
