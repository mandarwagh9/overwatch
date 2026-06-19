"""Phase B1 — appearance re-ID: HSV descriptors + cross-camera appearance gating."""
from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("cv2")

from app.domain.entities import (
    AppearanceDescriptor, BoundingBox, Point3D, Velocity3D, WorldObject,
)


def _solid(b: int, g: int, r: int, h: int = 40, w: int = 20) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (b, g, r)
    return img


# ---------------------------------------------------------------- descriptor

def test_descriptor_is_64d_and_l2_normalized():
    from app.infrastructure.detection_adapter import compute_hsv_appearance
    d = compute_hsv_appearance(_solid(0, 0, 255), BoundingBox(0, 0, 20, 40))
    assert d is not None
    assert d.vector.shape == (64,)
    assert float(np.linalg.norm(d.vector)) == pytest.approx(1.0, abs=1e-5)


def test_same_color_more_similar_than_different_color():
    from app.infrastructure.detection_adapter import compute_hsv_appearance
    red = compute_hsv_appearance(_solid(0, 0, 255), BoundingBox(0, 0, 20, 40))
    red2 = compute_hsv_appearance(_solid(0, 0, 255), BoundingBox(0, 0, 20, 40))
    blue = compute_hsv_appearance(_solid(255, 0, 0), BoundingBox(0, 0, 20, 40))
    assert red.cosine_similarity(red2) == pytest.approx(1.0, abs=1e-3)
    assert red.cosine_similarity(blue) < red.cosine_similarity(red2)


def test_degenerate_bbox_returns_none():
    from app.infrastructure.detection_adapter import compute_hsv_appearance
    assert compute_hsv_appearance(_solid(0, 0, 255), BoundingBox(0, 0, 0.4, 0.4)) is None


# ---------------------------------------------------- cross-camera matching

def _repo():
    from app.infrastructure.world_model_adapter import WorldModelRepositoryImpl
    config = Mock()
    config.get.return_value = {}
    config.get_int.return_value = 4
    config.get_float.return_value = 0.5
    config.get_list.return_value = []
    repo = WorldModelRepositoryImpl(config)
    repo._appearance_match_threshold = 0.5
    return repo


def _descriptor(vec):
    v = np.array(vec, dtype=np.float32)
    return AppearanceDescriptor(vector=v / (np.linalg.norm(v) + 1e-6))


def _insert(repo, obj_id, pos, appearance):
    repo._world_objects[obj_id] = WorldObject(
        object_id=obj_id, position=pos, velocity=Velocity3D(0, 0, 0),
        class_id=0, class_name="person", confidence=0.9,
        last_seen_camera=0, last_update=datetime.now(), appearance=appearance,
    )


def test_similar_appearance_within_distance_matches():
    repo = _repo()
    a = _descriptor([1, 0, 0, 0])
    _insert(repo, 1, Point3D(0, 0, 0), a)
    assert repo._find_matching_object(Point3D(0.5, 0, 0), 0, a) == 1


def test_dissimilar_appearance_does_not_match():
    """Two people at (nearly) the same spot but different clothing stay separate."""
    repo = _repo()
    _insert(repo, 1, Point3D(0, 0, 0), _descriptor([1, 0, 0, 0]))
    orthogonal = _descriptor([0, 1, 0, 0])  # cosine 0 < 0.5 threshold
    assert repo._find_matching_object(Point3D(0.3, 0, 0), 0, orthogonal) is None


def test_no_appearance_falls_back_to_distance():
    repo = _repo()
    _insert(repo, 1, Point3D(0, 0, 0), None)
    assert repo._find_matching_object(Point3D(0.5, 0, 0), 0, None) == 1
    assert repo._find_matching_object(Point3D(5, 0, 0), 0, None) is None


def test_reid_config_fields_are_wired():
    """Guard against doc drift: the new keys must actually flow through Settings."""
    from app.infrastructure.config_adapter import (
        PydanticConfigurationRepository, Settings,
    )
    repo = PydanticConfigurationRepository(
        Settings(cross_camera_appearance_threshold=0.7, appearance_reid_enabled=False)
    )
    assert repo.get_float("cross_camera_appearance_threshold", 0.5) == 0.7
    assert repo.get_bool("appearance_reid_enabled", True) is False
