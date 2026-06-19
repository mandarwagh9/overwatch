"""Characterization + robustness tests for CameraTracker.update lifecycle.

These pin the per-camera tracker behaviour (create → confirm → coast → remove)
and, importantly, that the matched/unmatched mapping resolves to the *correct
track ids* when multiple tracks exist — guarding the association refactor that
removes the positional-index → dict-ordering coupling.
"""
from datetime import datetime

from app.domain.entities import BoundingBox, Detection, TrackingState
from app.infrastructure.tracking_adapter import CameraTracker


def _tracker(min_hits: int = 3, max_age: int = 3, iou_threshold: float = 0.25) -> CameraTracker:
    return CameraTracker(
        camera_id=0,
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
        appearance_weight=0.4,
    )


def _det(x: float, det_id: str = "d") -> Detection:
    return Detection(
        detection_id=det_id,
        camera_id=0,
        bbox=BoundingBox(x, 0.0, x + 100.0, 200.0),
        confidence=0.9,
        class_id=0,
        class_name="person",
        timestamp=datetime.now(),
    )


def test_new_detection_creates_tentative_track():
    t = _tracker()
    tracks = t.update([_det(0)])
    assert len(tracks) == 1
    assert tracks[0].state == TrackingState.TENTATIVE
    assert tracks[0].track_id == 1


def test_repeated_match_confirms_and_keeps_same_id():
    t = _tracker(min_hits=3)
    for _ in range(3):
        t.update([_det(0)])
    tracks = list(t.tracks.values())
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].state == TrackingState.CONFIRMED


def test_unmatched_confirmed_track_coasts():
    t = _tracker(min_hits=2)
    t.update([_det(0)])
    t.update([_det(0)])  # confirmed now
    assert t.tracks[1].state == TrackingState.CONFIRMED
    conf_before = t.tracks[1].confidence
    # Next frame: a detection far away — does not match track 1
    t.update([_det(1000)])
    assert t.tracks[1].state == TrackingState.COASTING
    assert t.tracks[1].confidence < conf_before


def test_multi_track_mapping_resolves_correct_ids():
    """Two confirmed tracks; a single detection matches only the first.
    The matched/unmatched split must map to the right track ids."""
    t = _tracker(min_hits=2, max_age=10)
    # Confirm two well-separated tracks
    for _ in range(2):
        t.update([_det(0, "a"), _det(500, "b")])
    assert {tid: trk.state for tid, trk in t.tracks.items()} == {
        1: TrackingState.CONFIRMED,
        2: TrackingState.CONFIRMED,
    }
    # Frame with only the left detection: track 1 should update, track 2 coast
    t.update([_det(0, "a")])
    assert t.tracks[1].state == TrackingState.CONFIRMED
    assert t.tracks[1].time_since_update == 0
    assert t.tracks[2].state == TrackingState.COASTING
    assert t.tracks[2].time_since_update >= 1


def test_stale_track_is_removed_after_max_age():
    t = _tracker(min_hits=1, max_age=2)
    t.update([_det(0)])  # track 1 created + confirmed (min_hits=1)
    assert 1 in t.tracks
    # No matching detections for several frames -> time_since_update grows
    for _ in range(4):
        t.update([_det(1000, "far")])
    assert 1 not in t.tracks  # removed once time_since_update > max_age


def test_two_detections_create_two_distinct_tracks():
    t = _tracker()
    tracks = t.update([_det(0, "a"), _det(500, "b")])
    assert len(tracks) == 2
    assert {trk.track_id for trk in tracks} == {1, 2}
