"""
Tracking Manager using DeepSORT for multi-camera object continuity
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    print("⚠️ DeepSORT not available, using simple tracking")
    DeepSort = None

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy not available, using greedy assignment instead of Hungarian")

from app.config import settings
from app.core.detection_engine import DetectionResult, Detection


@dataclass
class Track:
    """Container for tracking information"""
    track_id: int
    camera_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    center: Tuple[float, float]  # x, y center
    confidence: float
    class_id: int
    class_name: str
    age: int
    hits: int
    time_since_update: int
    velocity: Tuple[float, float] = (0.0, 0.0)  # dx, dy per frame
    predicted_position: Optional[Tuple[float, float]] = None
    feature_vector: object = None  # appearance descriptor (np.ndarray)
    keypoints: Optional[list] = None  # COCO 17-joint skeleton (x, y, conf)


@dataclass
class TrackingResult:
    """Container for tracking results from a frame"""
    camera_id: int
    frame_number: int
    timestamp: float
    tracks: List[Track]
    processing_time: float


class SimpleTracker:
    """Simple centroid-based tracker as fallback"""
    
    def __init__(self, max_age: int = 30):
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.max_age = max_age
        self.max_distance_threshold = 100  # pixels
        
    def update(self, detections: List[Detection], camera_id: int, frame_number: int) -> List[Track]:
        """Update tracks with new detections"""
        current_time = time.time()
        
        # Convert detections to centers for distance calculation
        detection_centers = [det.center for det in detections]
        track_centers = [track.center for track in self.tracks.values()]
        
        # Simple assignment based on distance
        used_detections = set()
        updated_tracks = []
        
        for track_id, track in list(self.tracks.items()):
            best_match_idx = None
            best_distance = float('inf')
            
            for i, det_center in enumerate(detection_centers):
                if i in used_detections:
                    continue
                
                distance = np.sqrt(
                    (track.center[0] - det_center[0])**2 + 
                    (track.center[1] - det_center[1])**2
                )
                
                if distance < best_distance and distance < self.max_distance_threshold:
                    best_distance = distance
                    best_match_idx = i
            
            if best_match_idx is not None:
                # Update existing track
                detection = detections[best_match_idx]
                
                # Calculate velocity
                old_center = track.center
                new_center = detection.center
                velocity = (
                    new_center[0] - old_center[0],
                    new_center[1] - old_center[1]
                )
                
                updated_track = Track(
                    track_id=track_id,
                    camera_id=camera_id,
                    bbox=detection.bbox,
                    center=detection.center,
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    age=track.age + 1,
                    hits=track.hits + 1,
                    time_since_update=0,
                    velocity=velocity,
                    feature_vector=detection.feature_vector,
                    keypoints=detection.keypoints,
                )
                
                self.tracks[track_id] = updated_track
                updated_tracks.append(updated_track)
                used_detections.add(best_match_idx)
            else:
                # No match found, age the track
                track.time_since_update += 1
                if track.time_since_update <= self.max_age:
                    # Predict position based on velocity
                    predicted_center = (
                        track.center[0] + track.velocity[0],
                        track.center[1] + track.velocity[1]
                    )
                    
                    predicted_track = Track(
                        track_id=track_id,
                        camera_id=camera_id,
                        bbox=track.bbox,  # Keep last known bbox
                        center=predicted_center,
                        confidence=track.confidence * 0.9,  # Decay confidence
                        class_id=track.class_id,
                        class_name=track.class_name,
                        age=track.age + 1,
                        hits=track.hits,
                        time_since_update=track.time_since_update,
                        velocity=track.velocity,
                        predicted_position=predicted_center,
                        feature_vector=track.feature_vector,  # propagate
                    )
                    
                    self.tracks[track_id] = predicted_track
                    updated_tracks.append(predicted_track)
                else:
                    # Remove old track
                    del self.tracks[track_id]
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in used_detections:
                new_track = Track(
                    track_id=self.next_track_id,
                    camera_id=camera_id,
                    bbox=detection.bbox,
                    center=detection.center,
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    age=1,
                    hits=1,
                    time_since_update=0,
                    feature_vector=detection.feature_vector,
                    keypoints=detection.keypoints,
                )
                
                self.tracks[self.next_track_id] = new_track
                updated_tracks.append(new_track)
                self.next_track_id += 1
        
        return updated_tracks


def _compute_iou(box1: Tuple, box2: Tuple) -> float:
    """Compute IoU between two (x1,y1,x2,y2) bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    return inter / (a1 + a2 - inter + 1e-6)


class HungarianTracker:
    """Multi-object tracker using Hungarian (Munkres) optimal assignment.

    Cost matrix = 0.6 * IoU_distance + 0.4 * cosine_distance(appearance).
    Falls back to IoU-only when appearance features are not available.
    """

    def __init__(self, max_age: int = 30, iou_threshold: float = 0.25):
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.max_age = max_age
        self.iou_threshold = iou_threshold

    def update(self, detections: List[Detection], camera_id: int, frame_number: int) -> List[Track]:
        """Update tracks using Hungarian assignment on IoU + appearance cost."""
        if not self.tracks and not detections:
            return []
        if not self.tracks:
            return self._create_tracks(detections, camera_id)
        if not detections:
            return self._coast_all(camera_id)

        track_ids = list(self.tracks.keys())
        track_list = [self.tracks[tid] for tid in track_ids]
        n_t, n_d = len(track_list), len(detections)

        # --- Build cost matrix ---
        cost = np.full((n_t, n_d), 1e6, dtype=np.float64)
        for i, trk in enumerate(track_list):
            for j, det in enumerate(detections):
                iou = _compute_iou(trk.bbox, det.bbox)
                iou_cost = 1.0 - iou
                if (trk.feature_vector is not None and
                        det.feature_vector is not None):
                    cos_sim = float(np.dot(trk.feature_vector, det.feature_vector))
                    app_cost = 1.0 - cos_sim
                    cost[i, j] = 0.6 * iou_cost + 0.4 * app_cost
                else:
                    cost[i, j] = iou_cost

        # --- Optimal assignment ---
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_t, matched_d = set(), set()
        updated = []

        for r, c in zip(row_ind, col_ind):
            if _compute_iou(track_list[r].bbox, detections[c].bbox) >= self.iou_threshold:
                upd = self._update_track(track_list[r], detections[c], camera_id)
                updated.append(upd)
                matched_t.add(track_ids[r])
                matched_d.add(c)

        # Coast unmatched tracks
        for tid in track_ids:
            if tid not in matched_t:
                c = self._coast_track(self.tracks[tid], camera_id)
                if c is not None:
                    updated.append(c)

        # Birth new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_d:
                updated.append(self._birth_track(det, camera_id))

        return updated

    # ── internal helpers ──

    def _update_track(self, track: Track, det: Detection, cam_id: int) -> Track:
        vel = (det.center[0] - track.center[0], det.center[1] - track.center[1])
        upd = Track(
            track_id=track.track_id, camera_id=cam_id,
            bbox=det.bbox, center=det.center,
            confidence=det.confidence, class_id=det.class_id,
            class_name=det.class_name, age=track.age + 1,
            hits=track.hits + 1, time_since_update=0,
            velocity=vel, feature_vector=det.feature_vector,
            keypoints=det.keypoints,
        )
        self.tracks[track.track_id] = upd
        return upd

    def _coast_track(self, track: Track, cam_id: int) -> Optional[Track]:
        track.time_since_update += 1
        if track.time_since_update > self.max_age:
            self.tracks.pop(track.track_id, None)
            return None
        pred_c = (track.center[0] + track.velocity[0],
                  track.center[1] + track.velocity[1])
        coasted = Track(
            track_id=track.track_id, camera_id=cam_id,
            bbox=track.bbox, center=pred_c,
            confidence=track.confidence * 0.9,
            class_id=track.class_id, class_name=track.class_name,
            age=track.age + 1, hits=track.hits,
            time_since_update=track.time_since_update,
            velocity=track.velocity, predicted_position=pred_c,
            feature_vector=track.feature_vector,
        )
        self.tracks[track.track_id] = coasted
        return coasted

    def _birth_track(self, det: Detection, cam_id: int) -> Track:
        tid = self.next_track_id
        self.next_track_id += 1
        nt = Track(
            track_id=tid, camera_id=cam_id,
            bbox=det.bbox, center=det.center,
            confidence=det.confidence, class_id=det.class_id,
            class_name=det.class_name, age=1, hits=1,
            time_since_update=0, feature_vector=det.feature_vector,
            keypoints=det.keypoints,
        )
        self.tracks[tid] = nt
        return nt

    def _create_tracks(self, dets: List[Detection], cam_id: int) -> List[Track]:
        return [self._birth_track(d, cam_id) for d in dets]

    def _coast_all(self, cam_id: int) -> List[Track]:
        result = []
        for tid in list(self.tracks.keys()):
            c = self._coast_track(self.tracks[tid], cam_id)
            if c is not None:
                result.append(c)
        return result


class DeepSORTTracker:
    """DeepSORT-based tracker"""
    
    def __init__(self):
        self.tracker: Optional[DeepSort] = None
        self.is_initialized = False
        
    def initialize(self):
        """Initialize DeepSORT tracker"""
        try:
            if DeepSort is None:
                return False
            
            self.tracker = DeepSort(
                max_age=settings.tracking_max_age,
                n_init=settings.tracking_n_init,
                max_iou_distance=settings.tracking_max_iou_distance,
                embedder="mobilenet",  # Lightweight embedding for CPU
                half=False,  # Use FP32 for CPU
                bgr=True,
                embedder_gpu=False  # Force CPU
            )
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize DeepSORT: {e}")
            return False
    
    def update(self, detections: List[Detection], frame: np.ndarray, camera_id: int) -> List[Track]:
        """Update tracker with detections"""
        if not self.is_initialized or self.tracker is None:
            return []
        
        try:
            # Convert detections to DeepSORT format
            bboxes = []
            confidences = []
            class_ids = []
            
            for det in detections:
                # Convert to (x, y, w, h) format
                x1, y1, x2, y2 = det.bbox
                w = x2 - x1
                h = y2 - y1
                bboxes.append([x1, y1, w, h])
                confidences.append(det.confidence)
                class_ids.append(det.class_id)
            
            if not bboxes:
                # No detections, just update tracker
                tracks = self.tracker.update_tracks([], frame=frame)
            else:
                # Update with detections
                tracks = self.tracker.update_tracks(
                    raw_detections=list(zip(bboxes, confidences, class_ids)),
                    frame=frame
                )
            
            # Convert back to our Track format
            result_tracks = []
            for track in tracks:
                if track.is_confirmed():
                    bbox = track.to_ltrb()  # Get (left, top, right, bottom)
                    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                    
                    # Get class info from original detections if available
                    class_id = 0
                    class_name = "unknown"
                    confidence = 0.5
                    
                    if hasattr(track, 'det_class') and track.det_class is not None:
                        class_id = track.det_class
                    if hasattr(track, 'det_conf') and track.det_conf is not None:
                        confidence = track.det_conf
                    
                    result_track = Track(
                        track_id=track.track_id,
                        camera_id=camera_id,
                        bbox=tuple(bbox),
                        center=center,
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name,
                        age=track.age,
                        hits=track.hits,
                        time_since_update=track.time_since_update
                    )
                    
                    result_tracks.append(result_track)
            
            return result_tracks
            
        except Exception as e:
            print(f"❌ DeepSORT tracking error: {e}")
            return []


class TrackingManager:
    """Manages tracking across multiple cameras"""
    
    def __init__(self):
        self.camera_trackers: Dict[int, SimpleTracker] = {}
        self.hungarian_trackers: Dict[int, HungarianTracker] = {}
        self.deepsort_tracker = DeepSORTTracker()
        self.use_deepsort = False
        self.global_tracks: Dict[int, Track] = {}  # Global track registry
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
        self.is_running = False
        
        self.stats = {
            'frames_processed': 0,
            'total_tracks': 0,
            'active_tracks': 0,
            'tracks_per_camera': defaultdict(int)
        }
    
    async def initialize(self):
        """Initialize the tracking manager"""
        # Try to initialize DeepSORT
        if self.deepsort_tracker.initialize():
            self.use_deepsort = True
            print("✅ DeepSORT tracker initialized")
        elif SCIPY_AVAILABLE:
            print("✅ Hungarian (Munkres) tracker active")
        else:
            print("⚠️ Using simple centroid tracker")
        
        self.is_running = True
        print("🎯 Tracking Manager initialized")
    
    async def process_detections(
        self, 
        detection_result: DetectionResult, 
        frame: Optional[np.ndarray] = None
    ) -> TrackingResult:
        """Process detections and update tracks for a camera"""
        start_time = time.time()
        camera_id = detection_result.camera_id
        
        # Use appropriate tracker: DeepSORT > Hungarian > Simple
        if self.use_deepsort and frame is not None:
            tracks = self.deepsort_tracker.update(
                detection_result.detections, 
                frame, 
                camera_id
            )
        elif SCIPY_AVAILABLE:
            if camera_id not in self.hungarian_trackers:
                self.hungarian_trackers[camera_id] = HungarianTracker()
            tracks = self.hungarian_trackers[camera_id].update(
                detection_result.detections,
                camera_id,
                detection_result.frame_number
            )
        else:
            if camera_id not in self.camera_trackers:
                self.camera_trackers[camera_id] = SimpleTracker()
            tracks = self.camera_trackers[camera_id].update(
                detection_result.detections,
                camera_id,
                detection_result.frame_number
            )
        
        processing_time = time.time() - start_time
        
        # Update global track registry
        self._update_global_tracks(camera_id, tracks)
        
        # Update statistics
        self._update_stats(camera_id, tracks)
        
        return TrackingResult(
            camera_id=camera_id,
            frame_number=detection_result.frame_number,
            timestamp=detection_result.timestamp,
            tracks=tracks,
            processing_time=processing_time
        )
    
    def _update_global_tracks(self, camera_id: int, tracks: List[Track]):
        """Update global track registry"""
        for track in tracks:
            # Store track history
            self.track_history[track.track_id].append({
                'camera_id': camera_id,
                'timestamp': time.time(),
                'position': track.center,
                'bbox': track.bbox,
                'confidence': track.confidence
            })
            
            # Update global tracks
            self.global_tracks[track.track_id] = track
    
    def _update_stats(self, camera_id: int, tracks: List[Track]):
        """Update tracking statistics"""
        self.stats['frames_processed'] += 1
        self.stats['tracks_per_camera'][camera_id] = len(tracks)
        self.stats['active_tracks'] = len(self.global_tracks)
        
        # Count total unique tracks seen
        all_track_ids = set()
        for tracker in self.camera_trackers.values():
            all_track_ids.update(tracker.tracks.keys())
        self.stats['total_tracks'] = len(all_track_ids)
    
    def get_tracks_for_camera(self, camera_id: int) -> List[Track]:
        """Get current tracks for a specific camera"""
        if camera_id in self.camera_trackers:
            return list(self.camera_trackers[camera_id].tracks.values())
        return []
    
    def get_all_active_tracks(self) -> Dict[int, List[Track]]:
        """Get all active tracks organized by camera"""
        result = {}
        for camera_id, tracker in self.camera_trackers.items():
            result[camera_id] = list(tracker.tracks.values())
        return result
    
    def get_track_history(self, track_id: int) -> List[dict]:
        """Get history for a specific track"""
        return list(self.track_history.get(track_id, []))
    
    def predict_track_position(self, track_id: int, frames_ahead: int = 1) -> Optional[Tuple[float, float]]:
        """Predict future position of a track"""
        if track_id not in self.global_tracks:
            return None
        
        track = self.global_tracks[track_id]
        
        # Simple linear prediction based on velocity
        predicted_x = track.center[0] + track.velocity[0] * frames_ahead
        predicted_y = track.center[1] + track.velocity[1] * frames_ahead
        
        return (predicted_x, predicted_y)
    
    def get_stats(self) -> dict:
        """Get tracking statistics"""
        return {
            'frames_processed': self.stats['frames_processed'],
            'total_tracks': self.stats['total_tracks'],
            'active_tracks': self.stats['active_tracks'],
            'tracks_per_camera': dict(self.stats['tracks_per_camera']),
            'tracker_type': 'DeepSORT' if self.use_deepsort else ('Hungarian' if SCIPY_AVAILABLE else 'Simple'),
            'cameras_active': len(self.camera_trackers)
        }
    
    def is_active(self) -> bool:
        """Check if tracking manager is active"""
        return self.is_running