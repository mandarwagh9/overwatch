"""
Tracking repository adapter.
Implements Hungarian algorithm tracking without external dependencies.
"""
from __future__ import annotations
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

from app.application.ports import TrackingRepository, ConfigurationRepository
from app.domain.entities import CameraFrame, Detection, Track, BoundingBox, TrackingState


logger = logging.getLogger(__name__)


def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Compute IoU between two bounding boxes."""
    return box1.iou(box2)


def compute_cost_matrix(
    tracks: List[Track],
    detections: List[Detection],
    iou_weight: float = 0.6,
    appearance_weight: float = 0.4
) -> np.ndarray:
    """
    Compute cost matrix for Hungarian assignment.
    Cost = iou_weight * (1 - IoU) + appearance_weight * (1 - cosine_similarity)
    """
    n_tracks = len(tracks)
    n_detections = len(detections)
    
    if n_tracks == 0 or n_detections == 0:
        return np.array([])
    
    cost = np.zeros((n_tracks, n_detections), dtype=np.float64)
    
    for i, track in enumerate(tracks):
        for j, detection in enumerate(detections):
            # IoU cost
            iou = compute_iou(track.bbox, detection.bbox)
            iou_cost = 1.0 - iou
            
            # Appearance cost
            app_cost = 0.0
            if track.appearance is not None and detection.appearance is not None:
                similarity = track.appearance.cosine_similarity(detection.appearance)
                app_cost = 1.0 - similarity
            
            cost[i, j] = iou_weight * iou_cost + appearance_weight * app_cost
    
    return cost


def hungarian_assignment(cost_matrix: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Solve assignment problem using the Hungarian algorithm.
    Returns (row_indices, col_indices) of optimal assignments.
    """
    if cost_matrix.size == 0:
        return [], []
    
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return row_ind.tolist(), col_ind.tolist()
    except ImportError:
        # Fallback to greedy assignment if scipy not available
        return greedy_assignment(cost_matrix)


def greedy_assignment(cost_matrix: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Greedy assignment as fallback when scipy is not available.
    Not optimal but works in O(n^2).
    """
    if cost_matrix.size == 0:
        return [], []
    
    rows, cols = cost_matrix.shape
    row_ind = []
    col_ind = []
    
    # Create a copy to modify
    cost_copy = cost_matrix.copy()
    
    for _ in range(min(rows, cols)):
        # Find minimum cost
        min_val = np.inf
        min_pos = (0, 0)
        
        for i in range(rows):
            for j in range(cols):
                if cost_copy[i, j] < min_val:
                    min_val = cost_copy[i, j]
                    min_pos = (i, j)
        
        if min_val == np.inf:
            break
        
        row_ind.append(min_pos[0])
        col_ind.append(min_pos[1])
        
        # Mark row and column as used
        cost_copy[min_pos[0], :] = np.inf
        cost_copy[:, min_pos[1]] = np.inf
    
    return row_ind, col_ind


@dataclass
class CameraTracker:
    """Tracker for a single camera."""
    camera_id: int
    max_age: int
    min_hits: int
    iou_threshold: float
    appearance_weight: float
    
    tracks: Dict[int, Track] = field(default_factory=dict)
    next_track_id: int = 1
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracks with new detections."""
        # Predict existing tracks
        self._predict_tracks()
        
        # Associate detections with tracks
        matched_tracks, matched_dets, unmatched_tracks, unmatched_dets = \
            self._associate_detections(detections)
        
        # Update matched tracks
        for track_idx, det_idx in zip(matched_tracks, matched_dets):
            track_id = list(self.tracks.keys())[track_idx]
            detection = detections[det_idx]
            self._update_track(track_id, detection)
        
        # Mark unmatched tracks as coasting
        for track_idx in unmatched_tracks:
            track_id = list(self.tracks.keys())[track_idx]
            self._coast_track(track_id)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx])
        
        # Remove old tracks
        self._remove_old_tracks()
        
        return list(self.tracks.values())
    
    def _predict_tracks(self) -> None:
        """Predict next positions for all tracks."""
        for track in self.tracks.values():
            if track.state == TrackingState.CONFIRMED:
                # Predict based on velocity
                cx, cy = track.bbox.center
                new_cx = cx + track.velocity[0]
                new_cy = cy + track.velocity[1]
                
                # Update bbox with predicted position
                w = track.bbox.width
                h = track.bbox.height
                track.bbox = BoundingBox(
                    new_cx - w/2,
                    new_cy - h/2,
                    new_cx + w/2,
                    new_cy + h/2
                )
                track.predicted_position = (new_cx, new_cy)
            
            track.age += 1
            track.time_since_update += 1
    
    def _associate_detections(
        self,
        detections: List[Detection]
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """Associate detections with existing tracks."""
        if not self.tracks or not detections:
            return [], [], list(range(len(self.tracks))), list(range(len(detections)))
        
        track_list = list(self.tracks.values())
        
        # Compute cost matrix
        cost_matrix = compute_cost_matrix(
            track_list, detections,
            appearance_weight=self.appearance_weight
        )
        
        # Run Hungarian algorithm
        row_ind, col_ind = hungarian_assignment(cost_matrix)
        
        # Filter by IoU threshold
        matched_tracks = []
        matched_dets = []
        
        for r, c in zip(row_ind, col_ind):
            track = track_list[r]
            det = detections[c]
            iou = compute_iou(track.bbox, det.bbox)
            
            if iou >= self.iou_threshold:
                matched_tracks.append(r)
                matched_dets.append(c)
        
        # Find unmatched
        unmatched_tracks = [i for i in range(len(track_list)) if i not in matched_tracks]
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        
        return matched_tracks, matched_dets, unmatched_tracks, unmatched_dets
    
    def _update_track(self, track_id: int, detection: Detection) -> None:
        """Update a track with a matched detection."""
        track = self.tracks[track_id]
        
        # Calculate velocity
        old_center = track.bbox.center
        new_center = detection.bbox.center
        velocity = (
            new_center[0] - old_center[0],
            new_center[1] - old_center[1]
        )
        
        track.bbox = detection.bbox
        track.confidence = detection.confidence
        track.velocity = velocity
        track.time_since_update = 0
        track.hits += 1
        track.keypoints = detection.keypoints
        track.appearance = detection.appearance
        
        # Confirm track after min_hits
        if track.hits >= self.min_hits:
            track.state = TrackingState.CONFIRMED
    
    def _coast_track(self, track_id: int) -> None:
        """Mark a track as coasting (no detection matched)."""
        track = self.tracks[track_id]
        track.state = TrackingState.COASTING
        track.confidence *= 0.9  # Decay confidence
    
    def _create_track(self, detection: Detection) -> None:
        """Create a new track from a detection."""
        track = Track(
            track_id=self.next_track_id,
            camera_id=self.camera_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            class_id=detection.class_id,
            class_name=detection.class_name,
            state=TrackingState.TENTATIVE,
            age=1,
            hits=1,
            time_since_update=0,
            keypoints=detection.keypoints,
            appearance=detection.appearance
        )
        
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1
    
    def _remove_old_tracks(self) -> None:
        """Remove tracks that haven't been updated for too long."""
        to_remove = [
            tid for tid, track in self.tracks.items()
            if track.time_since_update > self.max_age
        ]
        
        for tid in to_remove:
            del self.tracks[tid]


class TrackingRepositoryImpl(TrackingRepository):
    """Implementation of tracking repository."""
    
    def __init__(self, config_repo: ConfigurationRepository):
        self._config = config_repo
        self._camera_trackers: Dict[int, CameraTracker] = {}
        self._is_initialized = False
        
        # Configuration
        self._max_age = config_repo.get_int("tracking_max_age", 30)
        self._min_hits = config_repo.get_int("tracking_min_hits", 3)
        self._iou_threshold = config_repo.get_float("tracking_iou_threshold", 0.25)
        self._appearance_weight = config_repo.get_float("tracking_appearance_weight", 0.4)
        
        logger.info("TrackingRepositoryImpl initialized")
    
    async def initialize(self) -> None:
        """Initialize the tracking repository."""
        self._is_initialized = True
        logger.info("Tracking repository initialized")
    
    def is_ready(self) -> bool:
        """Check if tracking is ready."""
        return self._is_initialized
    
    async def update(
        self,
        camera_id: int,
        detections: List[Detection],
        frame: Optional[CameraFrame] = None
    ) -> List[Track]:
        """Update tracks with new detections."""
        # Create tracker for camera if needed
        if camera_id not in self._camera_trackers:
            self._camera_trackers[camera_id] = CameraTracker(
                camera_id=camera_id,
                max_age=self._max_age,
                min_hits=self._min_hits,
                iou_threshold=self._iou_threshold,
                appearance_weight=self._appearance_weight
            )
        
        tracker = self._camera_trackers[camera_id]
        return tracker.update(detections)
    
    def get_tracks(self, camera_id: int) -> List[Track]:
        """Get current tracks for a camera."""
        if camera_id not in self._camera_trackers:
            return []
        return list(self._camera_trackers[camera_id].tracks.values())
    
    def get_all_tracks(self) -> Dict[int, List[Track]]:
        """Get all tracks organized by camera."""
        return {
            cam_id: list(tracker.tracks.values())
            for cam_id, tracker in self._camera_trackers.items()
        }
