"""Phase B2 — cross-camera ground-plane homography estimator."""
import numpy as np
import pytest

pytest.importorskip("cv2")

from app.infrastructure.homography import HomographyEstimator


def _apply_h(H, pt):
    v = H @ np.array([pt[0], pt[1], 1.0])
    return (v[0] / v[2], v[1] / v[2])


def test_no_homography_until_min_pairs():
    est = HomographyEstimator(min_pairs=4)
    for i in range(3):
        est.add_correspondence(0, 1, (i, i), (i, i))
    assert not est.has_homography(0, 1)
    assert est.project(0, 1, (1, 1)) is None


def test_recovers_known_homography():
    H_true = np.array([
        [1.2, 0.1, 30.0],
        [0.05, 1.1, -20.0],
        [0.0001, 0.0002, 1.0],
    ])
    est = HomographyEstimator(min_pairs=4)
    for s in [(10, 10), (200, 20), (30, 220), (250, 240), (120, 130)]:
        est.add_correspondence(0, 1, s, _apply_h(H_true, s))
    assert est.has_homography(0, 1)
    test_pt = (160, 90)
    expected = _apply_h(H_true, test_pt)
    got = est.project(0, 1, test_pt)
    assert got is not None
    assert got[0] == pytest.approx(expected[0], abs=1e-2)
    assert got[1] == pytest.approx(expected[1], abs=1e-2)


def test_identity_when_src_equals_dst():
    est = HomographyEstimator(min_pairs=4)
    for s in [(0, 0), (100, 0), (0, 100), (100, 100), (50, 50)]:
        est.add_correspondence(0, 1, s, s)
    got = est.project(0, 1, (42, 17))
    assert got is not None
    assert got[0] == pytest.approx(42, abs=1e-3)
    assert got[1] == pytest.approx(17, abs=1e-3)


def test_directional_homographies_are_independent():
    est = HomographyEstimator(min_pairs=4)
    for s in [(0, 0), (100, 0), (0, 100), (100, 100)]:
        est.add_correspondence(0, 1, s, (s[0] + 10, s[1]))
    assert est.has_homography(0, 1)
    assert not est.has_homography(1, 0)
    assert est.source_cameras_for(1) == [0]


def test_same_camera_correspondence_ignored():
    est = HomographyEstimator(min_pairs=4)
    for s in [(0, 0), (100, 0), (0, 100), (100, 100)]:
        est.add_correspondence(2, 2, s, s)
    assert not est.has_homography(2, 2)


def test_pair_buffer_capped_at_max_pairs():
    est = HomographyEstimator(min_pairs=4, max_pairs=10)
    for i in range(50):
        est.add_correspondence(0, 1, (i, i), (i, i))
    assert len(est._pairs[(0, 1)]) <= 10


# ---------------------------------------------- world-model integration

def _world_repo():
    from unittest.mock import Mock
    from app.infrastructure.world_model_adapter import WorldModelRepositoryImpl
    config = Mock()
    config.get.return_value = {}
    config.get_int.return_value = 4
    config.get_float.return_value = 0.5
    config.get_list.return_value = []
    return WorldModelRepositoryImpl(config)


def test_generate_predictions_uses_homography_when_available():
    from datetime import datetime
    from app.domain.entities import (
        PredictionMethod, Point3D, Velocity3D, WorldObject,
    )
    repo = _world_repo()
    for s in [(0, 0), (100, 0), (0, 100), (100, 100), (50, 50)]:
        repo._homography.add_correspondence(0, 1, s, s)  # identity cam0 -> cam1
    assert repo._homography.has_homography(0, 1)

    now = datetime.now()
    repo._world_objects[1] = WorldObject(
        object_id=1, position=Point3D(1, 0, 3), velocity=Velocity3D(0, 0, 0),
        class_id=0, class_name="person", confidence=0.9,
        last_seen_camera=0, last_update=now,
        source_tracks={0: 5}, camera_foot_points={0: (200, 300)},
    )
    preds = repo.generate_predictions(camera_id=1)
    assert len(preds) == 1
    assert preds[0].prediction_method == PredictionMethod.HOMOGRAPHY
    assert preds[0].source_camera == 0


def test_collect_correspondences_feeds_estimator_for_covisible_object():
    from datetime import datetime
    from app.domain.entities import Point3D, Velocity3D, WorldObject
    repo = _world_repo()
    now = datetime.now()
    repo._world_objects[1] = WorldObject(
        object_id=1, position=Point3D(1, 0, 3), velocity=Velocity3D(0, 0, 0),
        class_id=0, class_name="person", confidence=0.9, last_seen_camera=1,
        last_update=now,
        camera_last_seen={0: now, 1: now},
        camera_foot_points={0: (10, 20), 1: (30, 40)},
    )
    repo._collect_correspondences(now)
    assert len(repo._homography._pairs[(0, 1)]) == 1
    assert len(repo._homography._pairs[(1, 0)]) == 1
