"""Serialization tests for the WebSocket broadcast payload.

The pipeline serializes each PerceptionSnapshot to msgpack once and broadcasts the
bytes to every viewer; the frontend decodes this exact shape. These tests pin the
wire format via a msgpack round-trip.
"""
from datetime import datetime

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("msgpack")
pytest.importorskip("cv2")

import msgpack

from app.domain.entities import (
    BoundingBox, CameraFrame, Detection, PerceptionSnapshot, Point3D,
    PredictedTarget, PredictionMethod, Track, TrackingState, Velocity3D,
    WorldObject,
)
from app.infrastructure.websocket_adapter import WebSocketCommunicationRepository


def _snapshot() -> PerceptionSnapshot:
    ts = datetime.now()
    frame = CameraFrame(
        camera_id=0,
        frame_data=np.zeros((4, 4, 3), dtype=np.uint8),
        timestamp=ts,
        frame_number=1,
    )
    det = Detection(
        detection_id="det_1", camera_id=0,
        bbox=BoundingBox(10, 20, 110, 220), confidence=0.9,
        class_id=0, class_name="person", timestamp=ts,
    )
    trk = Track(
        track_id=7, camera_id=0, bbox=BoundingBox(10, 20, 110, 220),
        confidence=0.8, class_id=0, class_name="person",
        state=TrackingState.CONFIRMED, age=5, hits=4, velocity=(1.0, -2.0),
    )
    obj = WorldObject(
        object_id=3, position=Point3D(1.0, 2.0, 3.0),
        velocity=Velocity3D(0.1, 0.2, 0.3), class_id=0, class_name="person",
        confidence=0.7, last_seen_camera=0, last_update=ts,
    )
    pred = PredictedTarget(
        object_id=3, camera_id=1, predicted_bbox=BoundingBox(5, 5, 55, 105),
        confidence=0.6, time_since_seen=1.2, velocity_projection=(0.5, 0.5),
        source_camera=0, prediction_method=PredictionMethod.WORLD_PROJECTION,
    )
    return PerceptionSnapshot(
        timestamp=ts, generation=42,
        world_objects=[obj],
        camera_frames={0: frame},
        detections={0: [det]},
        tracks={0: [trk]},
        predictions={1: [pred]},
    )


def _roundtrip(snapshot: PerceptionSnapshot) -> dict:
    repo = WebSocketCommunicationRepository(max_clients=10)
    payload = repo._serialize_snapshot(snapshot)
    assert isinstance(payload, (bytes, bytearray))
    return msgpack.unpackb(payload, raw=False)


def test_snapshot_envelope_shape():
    msg = _roundtrip(_snapshot())
    assert msg["type"] == "snapshot"
    assert msg["generation"] == 42
    assert set(msg) >= {
        "type", "timestamp", "generation", "camera_frames",
        "world_objects", "detections", "tracks", "predictions", "metrics",
    }


def test_camera_frame_encoded_to_jpeg_bytes():
    msg = _roundtrip(_snapshot())
    frame_bytes = msg["camera_frames"]["0"]
    assert isinstance(frame_bytes, (bytes, bytearray))
    assert bytes(frame_bytes[:2]) == b"\xff\xd8"  # JPEG SOI marker


def test_detection_serialization_shape():
    det = _roundtrip(_snapshot())["detections"]["0"][0]
    assert det["detection_id"] == "det_1"
    assert det["bbox"] == [10, 20, 110, 220]
    assert det["class_name"] == "person"
    assert det["center"] == [60.0, 120.0]


def test_track_serialization_shape():
    trk = _roundtrip(_snapshot())["tracks"]["0"][0]
    assert trk["track_id"] == 7
    assert trk["state"] == "CONFIRMED"
    assert list(trk["velocity"]) == [1.0, -2.0]


def test_world_object_serialization_shape():
    obj = _roundtrip(_snapshot())["world_objects"][0]
    assert obj["object_id"] == 3
    assert obj["position"] == {"x": 1.0, "y": 2.0, "z": 3.0}
    assert obj["last_seen_camera"] == 0


def test_prediction_serialization_method_value():
    pred = _roundtrip(_snapshot())["predictions"]["1"][0]
    assert pred["object_id"] == 3
    assert pred["method"] == "world_projection"
    assert pred["source_camera"] == 0


def test_empty_snapshot_serializes_cleanly():
    empty = PerceptionSnapshot(timestamp=datetime.now(), generation=0)
    msg = _roundtrip(empty)
    assert msg["world_objects"] == []
    assert msg["camera_frames"] == {}
    assert msg["detections"] == {}
