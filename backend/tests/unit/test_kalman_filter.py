"""Tests for KalmanFilter. Locks current behavior; negative dt is buggy
(produces non-PSD covariance) — Task 2.1 fixes this and updates a test."""
import numpy as np
import pytest

from app.domain.entities import Point3D
from app.infrastructure.world_model_adapter import KalmanFilter


def test_predict_zero_dt_is_noop_for_state():
    kf = KalmanFilter()
    kf.state = np.array([1.0, 2.0, 3.0, 0.5, 0.5, 0.0])
    state_before = kf.state.copy()
    kf.predict(0.0)
    np.testing.assert_array_equal(kf.state, state_before)


def test_predict_advances_position_by_velocity_times_dt():
    kf = KalmanFilter()
    kf.state = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 0.5])
    kf.predict(2.0)
    assert kf.state[0] == pytest.approx(2.0)
    assert kf.state[1] == pytest.approx(4.0)
    assert kf.state[2] == pytest.approx(1.0)


def test_update_pulls_state_toward_measurement():
    kf = KalmanFilter()
    initial_pos = kf.position
    kf.update(Point3D(10.0, 20.0, 5.0), confidence=0.9)
    new_pos = kf.position
    assert new_pos.x > initial_pos.x
    assert new_pos.y > initial_pos.y


def test_predict_then_update_reduces_uncertainty():
    kf = KalmanFilter()
    kf.predict(0.1)
    cov_before = np.trace(kf.covariance[:3, :3])
    kf.update(Point3D(1.0, 1.0, 0.0), confidence=1.0)
    cov_after = np.trace(kf.covariance[:3, :3])
    assert cov_after < cov_before


def test_position_property_reads_first_three_state_elements():
    kf = KalmanFilter()
    kf.state = np.array([1.5, 2.5, 3.5, 0, 0, 0])
    assert kf.position == Point3D(1.5, 2.5, 3.5)


def test_velocity_property_reads_last_three_state_elements():
    kf = KalmanFilter()
    kf.state = np.array([0, 0, 0, 0.1, 0.2, 0.3])
    v = kf.velocity
    assert v.vx == pytest.approx(0.1)
    assert v.vy == pytest.approx(0.2)
    assert v.vz == pytest.approx(0.3)


def test_predict_future_does_not_mutate_state():
    kf = KalmanFilter()
    kf.state = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    state_before = kf.state.copy()
    future = kf.predict_future(1.0)
    np.testing.assert_array_equal(kf.state, state_before)
    assert future.x == pytest.approx(2.0)


def test_predict_zero_dt_keeps_covariance_psd():
    """After Task 2.1: callers clamp dt>=0 before predict, so KF only sees dt>=0."""
    kf = KalmanFilter()
    kf.predict(0.0)
    eigvals = np.linalg.eigvalsh(kf.covariance)
    assert np.all(eigvals >= -1e-9)


def test_predict_positive_dt_keeps_covariance_psd():
    kf = KalmanFilter()
    for _ in range(10):
        kf.predict(0.1)
    eigvals = np.linalg.eigvalsh(kf.covariance)
    assert np.all(eigvals >= -1e-9)
