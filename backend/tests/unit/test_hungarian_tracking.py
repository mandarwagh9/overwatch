"""Tests for Hungarian/greedy tracking matchers.

Locks the actual signatures in `tracking_adapter`:

- ``compute_cost_matrix(tracks, detections, iou_weight=0.6, appearance_weight=0.4)``
  takes ``Track`` and ``Detection`` objects and returns an ``np.ndarray``.
  Note: when either input is empty, it returns ``np.array([])`` (a 1-D empty
  array), NOT a (0, 0) matrix — locked here as current behavior.

- ``greedy_assignment(cost_matrix)`` returns ``(row_indices, col_indices)``
  as two lists of ints. There is NO threshold argument.
"""
from datetime import datetime

import numpy as np
import pytest

from app.domain.entities import (
    BoundingBox, Detection, Track, TrackingState,
)
from app.infrastructure.tracking_adapter import (
    compute_cost_matrix,
    greedy_assignment,
    hungarian_assignment,
)


def _bbox(x: float) -> BoundingBox:
    return BoundingBox(x, 0, x + 100, 100)


def _track(track_id: int, x: float) -> Track:
    return Track(
        track_id=track_id,
        camera_id=0,
        bbox=_bbox(x),
        confidence=0.9,
        class_id=0,
        class_name="person",
        state=TrackingState.CONFIRMED,
        age=1,
        hits=3,
        time_since_update=0,
    )


def _detection(det_id: str, x: float) -> Detection:
    return Detection(
        detection_id=det_id,
        camera_id=0,
        bbox=_bbox(x),
        confidence=0.9,
        class_id=0,
        class_name="person",
        timestamp=datetime.now(),
    )


def test_empty_inputs_produce_empty_array():
    """Current behavior: returns a 1-D empty array (not 2-D)."""
    cm = compute_cost_matrix([], [])
    assert cm.size == 0


def test_empty_tracks_with_detections_produces_empty_array():
    cm = compute_cost_matrix([], [_detection("d", 0)])
    assert cm.size == 0


def test_identical_boxes_have_zero_cost():
    t = _track(1, 0)
    d = _detection("d", 0)
    cm = compute_cost_matrix([t], [d])
    assert cm.shape == (1, 1)
    # iou=1, no appearance => cost = 0.6*0 + 0.4*0 = 0
    assert cm[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_disjoint_boxes_have_iou_weight_cost():
    """No appearance descriptors, so cost = iou_weight * (1 - 0) = 0.6."""
    t = _track(1, 0)
    d = _detection("d", 1000)
    cm = compute_cost_matrix([t], [d])
    assert cm[0, 0] == pytest.approx(0.6)


def test_greedy_assignment_pairs_minimum_cost():
    """Greedy returns (row_indices, col_indices) lists. Symmetric crossover
    matrix should pair (0,1) and (1,0)."""
    cm = np.array([[0.9, 0.1], [0.1, 0.9]])
    row_ind, col_ind = greedy_assignment(cm)
    assert sorted(zip(row_ind, col_ind)) == [(0, 1), (1, 0)]


def test_greedy_assignment_diagonal_min():
    cm = np.array([[0.1, 0.9], [0.9, 0.1]])
    row_ind, col_ind = greedy_assignment(cm)
    assert sorted(zip(row_ind, col_ind)) == [(0, 0), (1, 1)]


def test_greedy_assignment_empty_matrix():
    cm = np.array([])
    row_ind, col_ind = greedy_assignment(cm)
    assert row_ind == []
    assert col_ind == []


def test_hungarian_assignment_returns_lists():
    """hungarian_assignment uses scipy if available, else falls back to greedy.
    Either way, it returns two lists of indices."""
    cm = np.array([[0.1, 0.9], [0.9, 0.1]])
    row_ind, col_ind = hungarian_assignment(cm)
    assert isinstance(row_ind, list)
    assert isinstance(col_ind, list)
    assert sorted(zip(row_ind, col_ind)) == [(0, 0), (1, 1)]


def test_hungarian_assignment_empty_matrix():
    row_ind, col_ind = hungarian_assignment(np.array([]))
    assert row_ind == []
    assert col_ind == []
