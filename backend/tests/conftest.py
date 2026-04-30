"""Shared pytest fixtures for Overwatch tests."""
from __future__ import annotations
from datetime import datetime
from typing import Tuple

import numpy as np
import pytest

from app.domain.entities import (
    BoundingBox, CameraCalibration, CameraFrame, Detection,
    Point3D, Track, TrackingState
)


@pytest.fixture
def bbox_simple() -> BoundingBox:
    return BoundingBox(10.0, 20.0, 110.0, 220.0)


@pytest.fixture
def calibration_origin() -> CameraCalibration:
    """Camera at world origin, no rotation, focal=1000, center=(640,360)."""
    return CameraCalibration(
        camera_id=0,
        position=Point3D(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
        focal_length=1000.0,
        image_center=(640.0, 360.0),
    )


@pytest.fixture
def make_detection():
    """Factory for synthetic detections."""
    def _make(
        det_id: str = "det_1",
        camera_id: int = 0,
        bbox: Tuple[float, float, float, float] = (10.0, 20.0, 110.0, 220.0),
        confidence: float = 0.9,
        class_id: int = 0,
        class_name: str = "person",
    ) -> Detection:
        return Detection(
            detection_id=det_id,
            camera_id=camera_id,
            bbox=BoundingBox(*bbox),
            confidence=confidence,
            class_id=class_id,
            class_name=class_name,
            timestamp=datetime.now(),
        )
    return _make


@pytest.fixture
def make_track():
    """Factory for synthetic tracks."""
    def _make(
        track_id: int = 1,
        camera_id: int = 0,
        bbox: Tuple[float, float, float, float] = (10.0, 20.0, 110.0, 220.0),
        confidence: float = 0.9,
        state: TrackingState = TrackingState.CONFIRMED,
    ) -> Track:
        return Track(
            track_id=track_id,
            camera_id=camera_id,
            bbox=BoundingBox(*bbox),
            confidence=confidence,
            class_id=0,
            class_name="person",
            state=state,
            hits=5,
        )
    return _make


@pytest.fixture
def make_frame():
    """Factory for synthetic CameraFrame with all-zero image."""
    def _make(camera_id: int = 0, width: int = 1280, height: int = 720) -> CameraFrame:
        return CameraFrame(
            camera_id=camera_id,
            frame_data=np.zeros((height, width, 3), dtype=np.uint8),
            timestamp=datetime.now(),
            frame_number=0,
        )
    return _make
