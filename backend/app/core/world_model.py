"""
World Model for sensor fusion and coordinate transformation across multiple cameras
"""

import asyncio
import time
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque

from app.core.tracking_manager import TrackingResult, Track
from app.config import settings


@dataclass
class CameraCalibration:
    """Camera calibration and positioning information"""
    camera_id: int
    position: Tuple[float, float, float]  # x, y, z in world coordinates (meters)
    rotation: Tuple[float, float, float]  # roll, pitch, yaw in radians
    focal_length: float
    image_center: Tuple[float, float]  # cx, cy in pixels
    distortion: Optional[List[float]] = None
    gps_accuracy: float = 5.0  # GPS accuracy in meters (lower = better)
    last_update: float = 0.0  # timestamp of last GPS update
    position_at_h_learn: Optional[Tuple[float, float, float]] = None  # camera pos when homography was learned


@dataclass
class WorldObject:
    """Object in world coordinates"""
    object_id: int
    world_position: Tuple[float, float, float]  # x, y, z
    velocity: Tuple[float, float, float]  # dx, dy, dz per second
    class_id: int
    class_name: str
    confidence: float
    last_seen_camera: int
    last_update: float
    prediction_confidence: float = 1.0
    source_tracks: Dict[int, int] = field(default_factory=dict)  # camera_id -> track_id
    position_uncertainty: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # from KF P diagonal
    bbox_size: Tuple[float, float] = (0.0, 0.0)  # last known w, h in pixels
    feature_vector: object = None  # appearance descriptor for cross-camera re-ID (np.ndarray)
    keypoints: Optional[list] = None  # COCO 17-joint skeleton (x, y, conf)
    # Per-camera pixel-space tracking (essential for uncalibrated camera prediction)
    camera_pixel_positions: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    camera_pixel_velocities: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    camera_bbox_sizes: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    camera_keypoints: Dict[int, list] = field(default_factory=dict)
    camera_last_seen: Dict[int, float] = field(default_factory=dict)   # camera_id -> last update time


@dataclass
class PredictedTarget:
    """Predicted target position for visualization"""
    object_id: int
    camera_id: int
    predicted_bbox: Tuple[float, float, float, float]
    predicted_center: Tuple[float, float]
    confidence: float
    time_since_seen: float
    velocity_projection: Tuple[float, float]
    source_camera: int = -1  # camera that actually sees this person
    keypoints: Optional[list] = None  # projected COCO skeleton for ghost overlay
    homography_source: bool = False  # DEPRECATED — kept for compat
    prediction_method: str = 'EXTRAP'  # 'HOMOGRAPHY', 'EXTRAP', or 'WORLD'


# ── Cross-Camera Homography ──────────────────────────────────────────
# Learns the ground-plane homography H between camera pairs from shared
# person observations (foot-point correspondences).  Once H is known,
# a person detected in Camera A can be projected onto Camera B's pixel
# space instantly — no calibration, no extrapolation.
#
# Theory: Hartley & Zisserman, "Multiple View Geometry", Ch 13.
# Practice: AIFARMS multi-camera-pig-tracking (CVPR 2021 Workshop)
#           uses the same cv2.findHomography + RANSAC approach.

class CrossCameraHomography:
    """Learn and apply ground-plane homography between camera pairs
    from shared foot-point observations.  No calibration required."""

    def __init__(self, min_pairs: int = 4, max_pairs: int = 100,
                 ransac_thresh: float = 12.0, re_estimate_every: int = 5):
        # (cam_src, cam_dst) -> deque of (src_pt, dst_pt) foot-point pairs
        self._pairs: Dict[Tuple[int, int], deque] = defaultdict(
            lambda: deque(maxlen=max_pairs)
        )
        # (cam_src, cam_dst) -> 3x3 homography numpy array
        self._H: Dict[Tuple[int, int], np.ndarray] = {}
        # (cam_src, cam_dst) -> mean reprojection error (px) on inliers
        self._reproj_err: Dict[Tuple[int, int], float] = {}
        # (cam_src, cam_dst) -> number of RANSAC inliers
        self._inlier_count: Dict[Tuple[int, int], int] = {}

        self.min_pairs = min_pairs
        self.ransac_thresh = ransac_thresh
        self._re_est = re_estimate_every
        self._add_count: Dict[Tuple[int, int], int] = defaultdict(int)

    # ── Public API ─────────────────────────────────────────────────

    def add_correspondence(self, cam_src: int, cam_dst: int,
                           foot_src: Tuple[float, float],
                           foot_dst: Tuple[float, float]):
        """Record one shared-observation foot-point pair."""
        fwd = (cam_src, cam_dst)
        rev = (cam_dst, cam_src)
        self._pairs[fwd].append((
            np.array(foot_src, dtype=np.float64),
            np.array(foot_dst, dtype=np.float64),
        ))
        self._pairs[rev].append((
            np.array(foot_dst, dtype=np.float64),
            np.array(foot_src, dtype=np.float64),
        ))
        self._add_count[fwd] += 1
        self._add_count[rev] += 1
        # Re-estimate periodically (not every single add)
        if self._add_count[fwd] % self._re_est == 0:
            self._estimate(fwd)
            self._estimate(rev)

    def project_point(self, cam_src: int, cam_dst: int,
                      pt: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Project a 2D point from cam_src view → cam_dst view via H."""
        H = self._H.get((cam_src, cam_dst))
        if H is None:
            return None
        p = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
        proj = H @ p
        w = proj[2]
        if abs(w) < 1e-8:
            return None
        return (float(proj[0] / w), float(proj[1] / w))

    def project_bbox(self, cam_src: int, cam_dst: int,
                     foot_src: Tuple[float, float],
                     bbox_h_src: float,
                     bbox_w_src: float) -> Optional[Tuple[Tuple[float, float], float, float]]:
        """Project foot point + estimate bbox in target camera.

        Returns (projected_foot, estimated_width, estimated_height) or None.
        Uses the homography's local Jacobian to estimate scale change.
        """
        dst_foot = self.project_point(cam_src, cam_dst, foot_src)
        if dst_foot is None:
            return None
        # Estimate scale via Jacobian at the source point
        H = self._H[(cam_src, cam_dst)]
        head_src = (foot_src[0], foot_src[1] - bbox_h_src)
        side_src = (foot_src[0] + bbox_w_src / 2, foot_src[1])
        dst_head = self.project_point(cam_src, cam_dst, head_src)
        dst_side = self.project_point(cam_src, cam_dst, side_src)
        if dst_head is None or dst_side is None:
            # Fallback: preserve source bbox size
            return (dst_foot, bbox_w_src, bbox_h_src)
        est_h = ((dst_foot[0] - dst_head[0])**2 + (dst_foot[1] - dst_head[1])**2) ** 0.5
        est_w = 2.0 * ((dst_side[0] - dst_foot[0])**2 + (dst_side[1] - dst_foot[1])**2) ** 0.5
        # Clamp to reasonable range
        est_h = max(40.0, min(est_h, 600.0))
        est_w = max(20.0, min(est_w, 400.0))
        return (dst_foot, est_w, est_h)

    def has_homography(self, cam_src: int, cam_dst: int) -> bool:
        return (cam_src, cam_dst) in self._H

    def get_quality(self, cam_src: int, cam_dst: int) -> Optional[float]:
        """Mean reprojection error in px.  Lower = better.  None = no H."""
        return self._reproj_err.get((cam_src, cam_dst))

    def get_inlier_count(self, cam_src: int, cam_dst: int) -> int:
        return self._inlier_count.get((cam_src, cam_dst), 0)

    def get_pair_count(self, cam_src: int, cam_dst: int) -> int:
        return len(self._pairs.get((cam_src, cam_dst), []))

    def get_all_stats(self) -> dict:
        """Summary for debugging / UI."""
        stats = {}
        for key in self._H:
            stats[f"{key[0]}->{key[1]}"] = {
                'pairs': len(self._pairs.get(key, [])),
                'inliers': self._inlier_count.get(key, 0),
                'reproj_err': round(self._reproj_err.get(key, -1), 2),
            }
        return stats

    def invalidate(self, cam_src: int, cam_dst: int):
        """Flush homography when camera moves significantly."""
        for key in [(cam_src, cam_dst), (cam_dst, cam_src)]:
            self._pairs.pop(key, None)
            self._H.pop(key, None)
            self._reproj_err.pop(key, None)
            self._inlier_count.pop(key, None)
            self._add_count.pop(key, None)

    # ── Internal ──────────────────────────────────────────────────

    def _estimate(self, key: Tuple[int, int]):
        pairs = self._pairs.get(key)
        if pairs is None or len(pairs) < self.min_pairs:
            return
        src = np.array([p[0] for p in pairs], dtype=np.float64)
        dst = np.array([p[1] for p in pairs], dtype=np.float64)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_thresh)
        if H is None:
            return
        inlier_mask = mask.ravel().astype(bool)
        n_inliers = int(inlier_mask.sum())
        if n_inliers < self.min_pairs:
            return  # not enough inliers — unreliable
        # Compute reprojection error on inliers
        src_in = src[inlier_mask]
        dst_in = dst[inlier_mask]
        ones = np.ones((len(src_in), 1))
        pts_h = np.hstack([src_in, ones])  # Nx3
        proj = (H @ pts_h.T).T  # Nx3
        w = proj[:, 2:3]
        w[np.abs(w) < 1e-8] = 1e-8
        proj_2d = proj[:, :2] / w
        err = np.linalg.norm(proj_2d - dst_in, axis=1)
        mean_err = float(np.mean(err))
        # Quality gate — if error is too large, camera may have moved
        if mean_err > 50.0:
            return
        self._H[key] = H
        self._reproj_err[key] = mean_err
        self._inlier_count[key] = n_inliers


# Assumed average person height in meters (for depth estimation from bbox)
PERSON_HEIGHT_M = 1.7


def _build_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build a 3×3 rotation matrix that transforms WORLD vectors into CAMERA vectors.

    Coordinate systems:
      World: +X east, +Y north, +Z up  (GPS/equirectangular)
      Camera: +X right, +Y down, +Z forward (optical axis, OpenCV convention)

    At identity (roll=pitch=yaw=0) the camera points north (+Y world)
    held horizontally.  Yaw is compass heading (clockwise-positive),
    pitch tilts up/down, roll tilts side-to-side.
    """
    # Base rotation: world(XYZ) → camera(XYZ) when pointing north horizontally
    #   world +X (east)  → cam +X (right)
    #   world +Y (north) → cam +Z (forward)
    #   world +Z (up)    → cam -Y (up in image)
    R_base = np.array([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0],
    ], dtype=float)

    # Device orientation in world frame (compass: clockwise = positive yaw)
    # Negate yaw because Rz is CCW-positive but compass heading is CW-positive
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(-yaw), np.sin(-yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)

    R_device = Rz @ Ry @ Rx  # device orientation in world frame

    # Full world→camera: undo device rotation, then apply base mapping
    return R_base @ R_device.T


class CoordinateTransform:
    """Handles coordinate transformations between camera and world space.

    When a CameraCalibration includes live GPS position and IMU rotation
    (from a moving phone / body-cam), the transform uses a proper pinhole
    camera model with a full 3×3 rotation matrix.  For static uncalibrated
    cameras the simpler hardcoded-position fallback is used.
    """

    def __init__(self):
        self.camera_calibrations: Dict[int, CameraCalibration] = {}
        # Cache rotation matrices (rebuilt whenever calibration changes)
        self._R_cache: Dict[int, np.ndarray] = {}
        self._Rt_cache: Dict[int, np.ndarray] = {}

    def add_camera_calibration(self, calibration: CameraCalibration):
        """Add / update camera calibration data"""
        self.camera_calibrations[calibration.camera_id] = calibration
        # Pre-compute rotation matrices
        R = _build_rotation_matrix(*calibration.rotation)
        self._R_cache[calibration.camera_id] = R
        self._Rt_cache[calibration.camera_id] = R.T  # inverse rotation (world→cam)

    def pixel_to_world(self, camera_id: int, pixel_coords: Tuple[float, float],
                       depth: float = 1.0) -> Optional[Tuple[float, float, float]]:
        """Convert pixel coordinates to world coordinates using rotation-aware projection."""
        if camera_id not in self.camera_calibrations:
            return self._simple_pixel_to_world(camera_id, pixel_coords, depth)

        calib = self.camera_calibrations[camera_id]
        R_inv = self._Rt_cache.get(camera_id)  # R^T = R^{-1} for orthogonal R
        if R_inv is None:
            R_inv = _build_rotation_matrix(*calib.rotation).T

        # Pixel → normalised camera-frame ray direction
        x_norm = (pixel_coords[0] - calib.image_center[0]) / calib.focal_length
        y_norm = (pixel_coords[1] - calib.image_center[1]) / calib.focal_length
        ray_cam = np.array([x_norm, y_norm, 1.0], dtype=float)

        # Rotate ray into world frame
        ray_world = R_inv @ ray_cam
        ray_len = np.linalg.norm(ray_world)
        if ray_len < 1e-8:
            return None
        ray_world /= ray_len

        # Walk along the ray by `depth` metres
        pos = np.array(calib.position, dtype=float)
        world_pt = pos + ray_world * depth

        return (float(world_pt[0]), float(world_pt[1]), float(world_pt[2]))

    def world_to_pixel(self, camera_id: int,
                       world_coords: Tuple[float, float, float]) -> Optional[Tuple[float, float]]:
        """Convert world coordinates to pixel coordinates using rotation-aware projection."""
        if camera_id not in self.camera_calibrations:
            return self._simple_world_to_pixel(camera_id, world_coords)

        calib = self.camera_calibrations[camera_id]
        R = self._R_cache.get(camera_id)
        if R is None:
            R = _build_rotation_matrix(*calib.rotation)

        # Vector from camera position to world point
        rel_world = np.array(world_coords, dtype=float) - np.array(calib.position, dtype=float)

        # Rotate into camera frame
        rel_cam = R @ rel_world

        # rel_cam[2] is depth along camera's optical axis
        if rel_cam[2] <= 0.01:
            return None  # Point is behind or at camera

        # Perspective divide → normalised image coords
        x_norm = rel_cam[0] / rel_cam[2]
        y_norm = rel_cam[1] / rel_cam[2]

        pixel_x = x_norm * calib.focal_length + calib.image_center[0]
        pixel_y = y_norm * calib.focal_length + calib.image_center[1]

        return (float(pixel_x), float(pixel_y))
    
    def _simple_pixel_to_world(self, camera_id: int, pixel_coords: Tuple[float, float], depth: float) -> Tuple[float, float, float]:
        """Simple transformation for uncalibrated cameras using geometric assumptions"""
        # Assume cameras are positioned in a grid pattern
        # This is a simplified model for demonstration
        
        camera_positions = {
            0: (0, 0, 2),      # Front camera
            1: (5, 0, 2),      # Right camera  
            2: (0, 5, 2),      # Back camera
            3: (-5, 0, 2)      # Left camera
        }
        
        base_pos = camera_positions.get(camera_id, (0, 0, 2))
        
        # Convert pixel coordinates to world offset
        # Assume 1920x1080 frame with 60-degree FOV
        frame_width = settings.frame_width
        frame_height = settings.frame_height
        
        # Normalize pixel coordinates (-1 to 1)
        norm_x = (pixel_coords[0] - frame_width / 2) / (frame_width / 2)
        norm_y = (pixel_coords[1] - frame_height / 2) / (frame_height / 2)
        
        # Apply field of view scaling
        fov_scale = np.tan(np.radians(30))  # Half of 60-degree FOV
        world_offset_x = norm_x * fov_scale * depth
        world_offset_y = norm_y * fov_scale * depth
        
        world_x = base_pos[0] + world_offset_x
        world_y = base_pos[1] + world_offset_y
        world_z = base_pos[2] - depth
        
        return (world_x, world_y, world_z)
    
    def _simple_world_to_pixel(self, camera_id: int, world_coords: Tuple[float, float, float]) -> Tuple[float, float]:
        """Simple transformation from world to pixel coordinates"""
        camera_positions = {
            0: (0, 0, 2),      # Front camera
            1: (5, 0, 2),      # Right camera  
            2: (0, 5, 2),      # Back camera
            3: (-5, 0, 2)      # Left camera
        }
        
        base_pos = camera_positions.get(camera_id, (0, 0, 2))
        
        # Calculate relative position
        rel_x = world_coords[0] - base_pos[0]
        rel_y = world_coords[1] - base_pos[1]
        rel_z = base_pos[2] - world_coords[2]
        
        if rel_z <= 0:
            return (0, 0)  # Behind camera
        
        # Project to normalized coordinates
        fov_scale = np.tan(np.radians(30))
        norm_x = rel_x / (fov_scale * rel_z)
        norm_y = rel_y / (fov_scale * rel_z)
        
        # Convert to pixel coordinates
        frame_width = settings.frame_width
        frame_height = settings.frame_height
        
        pixel_x = (norm_x + 1) * frame_width / 2
        pixel_y = (norm_y + 1) * frame_height / 2
        
        return (pixel_x, pixel_y)


class WorldModel:
    """Central world model for sensor fusion"""
    
    def __init__(self):
        self.coordinate_transform = CoordinateTransform()
        # Kalman filters per object id
        self.kalman_filters: Dict[int, 'KalmanFilter'] = {}
        self.world_objects: Dict[int, WorldObject] = {}
        self.next_object_id = 1
        self.track_to_object_mapping: Dict[Tuple[int, int], int] = {}  # (camera_id, track_id) -> object_id
        self.object_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=20))
        self.prediction_horizon = 5.0  # seconds
        
        # Per-sensor trust scores (camera_id -> trust in [0.1, 1.0])
        self.sensor_trust: Dict[int, float] = defaultdict(lambda: 1.0)

        # Cross-camera ground-plane homography (self-calibrating)
        self.cross_camera_homography = CrossCameraHomography()
        
        self.stats = {
            'objects_tracked': 0,
            'fused_objects': 0,
            'predictions_generated': 0,
            'coordinate_transforms': 0
        }


class KalmanFilter:
    """Simple constant-velocity Kalman Filter for 3D position+velocity."""

    def __init__(self, init_pos: Tuple[float, float, float], init_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        # State: [x, y, z, vx, vy, vz]
        self.x = np.zeros((6, 1), dtype=float)
        self.x[0:3, 0] = np.array(init_pos, dtype=float)
        self.x[3:6, 0] = np.array(init_vel, dtype=float)

        # Covariance
        self.P = np.eye(6, dtype=float) * 1.0

        # Process noise (tunable)
        self.q_pos = 0.1
        self.q_vel = 1.0

        # Measurement matrix (we measure positions only)
        self.H = np.zeros((3, 6), dtype=float)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Measurement noise
        self.R = np.eye(3, dtype=float) * 0.5

    def _build_F(self, dt: float) -> np.ndarray:
        F = np.eye(6, dtype=float)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F

    def _build_Q(self, dt: float) -> np.ndarray:
        # Continuous white noise model discretized simplification
        q = np.zeros((6, 6), dtype=float)
        q[0:3, 0:3] = np.eye(3) * (self.q_pos * dt)
        q[3:6, 3:6] = np.eye(3) * (self.q_vel * dt)
        return q

    def predict(self, dt: float = 0.1):
        F = self._build_F(dt)
        Q = self._build_Q(dt)
        self.x = F.dot(self.x)
        self.P = F.dot(self.P).dot(F.T) + Q

    def update(self, meas_pos: Tuple[float, float, float],
               confidence: float = 1.0, bbox_area: float = 10000.0,
               sensor_trust: float = 1.0):
        """Update with measurement.  R is scaled inversely by detection quality
        AND sensor trust, so unreliable sensors influence the filter less."""
        # Adaptive measurement noise
        quality = max(confidence, 0.1) * min(1.0, bbox_area / 5000.0) * max(sensor_trust, 0.1)
        noise_scale = np.clip(0.5 / quality, 0.1, 10.0)
        R_adaptive = self.R * noise_scale

        z = np.array(meas_pos, dtype=float).reshape((3, 1))
        y = z - self.H.dot(self.x)
        S = self.H.dot(self.P).dot(self.H.T) + R_adaptive
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        self.x = self.x + K.dot(y)
        I = np.eye(6)
        self.P = (I - K.dot(self.H)).dot(self.P)

    def current_position(self) -> Tuple[float, float, float]:
        return (float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]))

    def current_velocity(self) -> Tuple[float, float, float]:
        return (float(self.x[3, 0]), float(self.x[4, 0]), float(self.x[5, 0]))

    def predict_future(self, dt: float) -> Tuple[float, float, float]:
        # Create local copy to predict without changing filter state
        F = self._build_F(dt)
        x_future = F.dot(self.x)
        return (float(x_future[0, 0]), float(x_future[1, 0]), float(x_future[2, 0]))


# --- WorldModel methods (were incorrectly nested inside KalmanFilter) ---

# Re-open WorldModel by monkey-patching is fragile; instead we fix indentation
# by making these proper WorldModel methods below.

# We add initialize and _setup_default_cameras to WorldModel via the class body extension trick:

async def _wm_initialize(self):
    """Initialize the world model"""
    self._setup_default_cameras()
    print("🌍 World Model initialized")

def _wm_setup_default_cameras(self):
    """Setup default camera calibrations for demo.

    Convention: yaw=0 → north, yaw=π/2 → east (clockwise-positive compass heading).
    Static cameras are arranged in a square, each pointing toward the center.
    """
    default_cameras = [
        CameraCalibration(0, (0, 0, 2), (0, 0, 0), 800, (640, 360)),                    # north-facing
        CameraCalibration(1, (5, 0, 2), (0, 0, 3 * np.pi / 2), 800, (640, 360)),        # west-facing (toward center)
        CameraCalibration(2, (0, 5, 2), (0, 0, np.pi), 800, (640, 360)),                 # south-facing (toward center)
        CameraCalibration(3, (-5, 0, 2), (0, 0, np.pi / 2), 800, (640, 360)),            # east-facing (toward center)
    ]
    for calib in default_cameras:
        self.coordinate_transform.add_camera_calibration(calib)

WorldModel.initialize = _wm_initialize
WorldModel._setup_default_cameras = _wm_setup_default_cameras


async def _wm_update_with_tracking_results(self, tracking_results: List[TrackingResult]):
    """Update world model with new tracking results"""
    current_time = time.time()
    
    for tracking_result in tracking_results:
        await self._process_camera_tracks(tracking_result, current_time)
    
    # ── Clean stale source_tracks entries ───────────────────────────
    # Build set of active (camera_id, track_id) pairs from this tick
    active_track_keys = set()
    cameras_reporting = set()
    for tr in tracking_results:
        cameras_reporting.add(tr.camera_id)
        for t in tr.tracks:
            # Only actively-measured tracks count; coasting tracks (predicted
            # positions from the tracker) should NOT block ghost predictions
            if t.time_since_update <= 3:
                active_track_keys.add((tr.camera_id, t.track_id))
    
    # For each world object, remove source_track entries where the camera
    # reported tracks this tick but the specific track_id is gone (lost tracking)
    for obj in self.world_objects.values():
        stale = [cam_id for cam_id in obj.source_tracks
                 if cam_id in cameras_reporting
                 and (cam_id, obj.source_tracks[cam_id]) not in active_track_keys]
        for cam_id in stale:
            del obj.source_tracks[cam_id]
    
    # Clean up old objects
    self._cleanup_old_objects(current_time)
    
    # Update statistics
    self.stats['fused_objects'] = len(self.world_objects)

async def _wm_process_camera_tracks(self, tracking_result: TrackingResult, current_time: float):
    """Process tracks from a single camera"""
    camera_id = tracking_result.camera_id
    
    for track in tracking_result.tracks:
        # Skip heavily coasting tracks — stale predicted positions corrupt
        # the world model and delay ghost prediction generation
        if track.time_since_update > 3:
            continue

        track_key = (camera_id, track.track_id)

        # ── Depth estimation from bounding box height ──────────────
        # depth ≈ (person_height_m × focal_length) / bbox_height_px
        # Falls back to 3.0m if bbox is too small or calibration missing.
        bbox_h_px = max(1.0, float(track.bbox[3] - track.bbox[1]))
        calib = self.coordinate_transform.camera_calibrations.get(camera_id)
        if calib is not None and bbox_h_px > 20:
            estimated_depth = (PERSON_HEIGHT_M * calib.focal_length) / bbox_h_px
            estimated_depth = max(0.5, min(estimated_depth, 100.0))  # clamp
        else:
            estimated_depth = 3.0  # safe default

        # Convert track position to world coordinates
        world_pos = self.coordinate_transform.pixel_to_world(
            camera_id,
            track.center,
            depth=estimated_depth
        )
        
        if world_pos is None:
            continue
        
        self.stats['coordinate_transforms'] += 1
        
        # Check if this track is already associated with a world object
        if track_key in self.track_to_object_mapping:
            object_id = self.track_to_object_mapping[track_key]
            await self._update_existing_object(object_id, track, world_pos, current_time)
        else:
            # Check for nearby objects that might be the same (cross-camera re-ID)
            fv = getattr(track, 'feature_vector', None)
            object_id = self._find_matching_object(world_pos, track.class_id, current_time, feature_vector=fv, camera_id=camera_id)
            if object_id:
                self.track_to_object_mapping[track_key] = object_id
                await self._update_existing_object(object_id, track, world_pos, current_time)
            else:
                # Create new object
                await self._create_new_object(track_key, track, world_pos, current_time)

async def _wm_update_existing_object(self, object_id: int, track, world_pos: Tuple[float, float, float], current_time: float):
    """Update an existing world object with adaptive KF and appearance EMA."""
    if object_id not in self.world_objects:
        return
    
    obj = self.world_objects[object_id]
    bbox_area = max(1.0, (track.bbox[2] - track.bbox[0]) * (track.bbox[3] - track.bbox[1]))

    # Get sensor trust for this camera, scaled by GPS accuracy if available
    cam_trust = self.sensor_trust[track.camera_id]
    calib = self.coordinate_transform.camera_calibrations.get(track.camera_id)
    if calib is not None and calib.gps_accuracy > 0:
        # Tight GPS (≤5m) → full trust multiplier; sloppy (50m) → 0.1× multiplier
        gps_quality = min(1.0, 5.0 / max(calib.gps_accuracy, 0.1))
        cam_trust = cam_trust * max(0.1, gps_quality)

    # Use Kalman filter when available to smooth and update state
    if object_id in self.kalman_filters:
        kf = self.kalman_filters[object_id]
        # Predict to current time delta
        dt = max(1e-3, current_time - obj.last_update)
        kf.predict(dt)
        # Adaptive update — confidence, bbox area, and sensor trust scale measurement noise
        kf.update(world_pos, confidence=track.confidence, bbox_area=bbox_area, sensor_trust=cam_trust)
        obj.world_position = kf.current_position()
        obj.velocity = kf.current_velocity()
        # Expose KF covariance as position uncertainty
        P_diag = np.diag(kf.P)
        obj.position_uncertainty = (float(P_diag[0]), float(P_diag[1]), float(P_diag[2]))
        
        # Update sensor trust: check if this measurement is consistent with the predicted state
        innovation = np.sqrt(sum((a - b)**2 for a, b in zip(world_pos, kf.current_position())))
        if innovation < 1.0:  # consistent measurement
            self.sensor_trust[track.camera_id] = min(1.0, cam_trust + 0.005)
        else:  # inconsistent — decay trust
            self.sensor_trust[track.camera_id] = max(0.1, cam_trust - 0.01)
    else:
        # Fallback velocity estimate
        time_delta = current_time - obj.last_update
        if time_delta > 0:
            velocity = (
                (world_pos[0] - obj.world_position[0]) / time_delta,
                (world_pos[1] - obj.world_position[1]) / time_delta,
                (world_pos[2] - obj.world_position[2]) / time_delta
            )
        else:
            velocity = obj.velocity

        obj.world_position = world_pos
        obj.velocity = velocity

    obj.confidence = track.confidence
    obj.last_seen_camera = track.camera_id
    obj.last_update = current_time
    obj.prediction_confidence = 1.0

    # Update source tracks
    obj.source_tracks[track.camera_id] = track.track_id

    # Update bbox size
    obj.bbox_size = (float(track.bbox[2] - track.bbox[0]), float(track.bbox[3] - track.bbox[1]))

    # ── Per-camera pixel-space tracking (critical for ghost projection) ──
    old_pixel = obj.camera_pixel_positions.get(track.camera_id)
    old_cam_time = obj.camera_last_seen.get(track.camera_id, current_time)
    dt = max(1e-3, current_time - old_cam_time)
    obj.camera_pixel_positions[track.camera_id] = track.center
    obj.camera_bbox_sizes[track.camera_id] = obj.bbox_size
    if old_pixel is not None:
        obj.camera_pixel_velocities[track.camera_id] = (
            (track.center[0] - old_pixel[0]) / dt,
            (track.center[1] - old_pixel[1]) / dt,
        )
    obj.camera_last_seen[track.camera_id] = current_time

    # Update keypoints (store latest skeleton + per-camera)
    kp = getattr(track, 'keypoints', None)
    if kp is not None:
        obj.keypoints = kp
        obj.camera_keypoints[track.camera_id] = kp

    # Update appearance feature (exponential moving average for re-ID stability)
    fv = getattr(track, 'feature_vector', None)
    if fv is not None:
        if obj.feature_vector is None:
            obj.feature_vector = fv.copy()
        else:
            alpha = 0.3  # blend new observation with running descriptor
            obj.feature_vector = alpha * fv + (1 - alpha) * obj.feature_vector
            norm = np.linalg.norm(obj.feature_vector) + 1e-6
            obj.feature_vector = obj.feature_vector / norm

    # ── Collect cross-camera foot-point correspondences ────────────
    # When this object is simultaneously visible from multiple cameras,
    # record foot-point pairs to build the ground-plane homography.
    this_cam = track.camera_id
    this_foot = (track.center[0], float(track.bbox[3]))  # bottom-center
    for other_cam, other_tid in list(obj.source_tracks.items()):
        if other_cam == this_cam:
            continue
        other_time = obj.camera_last_seen.get(other_cam, 0)
        if (current_time - other_time) > 0.5:
            continue  # other camera's observation is stale
        other_pos = obj.camera_pixel_positions.get(other_cam)
        other_bsz = obj.camera_bbox_sizes.get(other_cam)
        if other_pos is None or other_bsz is None:
            continue
        # Reconstruct other camera's foot point from its stored center + half-height
        other_foot = (other_pos[0], other_pos[1] + other_bsz[1] / 2.0)
        self.cross_camera_homography.add_correspondence(
            other_cam, this_cam, other_foot, this_foot
        )
        # Mark current camera positions as homography-learn anchors
        for cid in (this_cam, other_cam):
            cc = self.coordinate_transform.camera_calibrations.get(cid)
            if cc is not None and cc.position_at_h_learn is None:
                cc.position_at_h_learn = cc.position

    # Store in history
    self.object_history[object_id].append({
        'timestamp': current_time,
        'world_position': obj.world_position,
        'velocity': obj.velocity,
        'camera_id': track.camera_id,
        'confidence': track.confidence
    })

async def _wm_create_new_object(self, track_key: Tuple[int, int], track, world_pos: Tuple[float, float, float], current_time: float):
    """Create a new world object"""
    object_id = self.next_object_id
    self.next_object_id += 1
    
    _kp = getattr(track, 'keypoints', None)
    _bsz = (float(track.bbox[2] - track.bbox[0]), float(track.bbox[3] - track.bbox[1]))
    new_object = WorldObject(
        object_id=object_id,
        world_position=world_pos,
        velocity=(0.0, 0.0, 0.0),
        class_id=track.class_id,
        class_name=track.class_name,
        confidence=track.confidence,
        last_seen_camera=track.camera_id,
        last_update=current_time,
        source_tracks={track.camera_id: track.track_id},
        bbox_size=_bsz,
        feature_vector=getattr(track, 'feature_vector', None),
        keypoints=_kp,
        camera_pixel_positions={track.camera_id: track.center},
        camera_pixel_velocities={track.camera_id: (
            track.velocity[0] * settings.target_fps,
            track.velocity[1] * settings.target_fps,
        )},
        camera_bbox_sizes={track.camera_id: _bsz},
        camera_keypoints=({track.camera_id: _kp} if _kp else {}),
        camera_last_seen={track.camera_id: current_time},
    )
    
    self.world_objects[object_id] = new_object
    # Initialize Kalman filter for this object
    try:
        kf = KalmanFilter(world_pos, (0.0, 0.0, 0.0))
        self.kalman_filters[object_id] = kf
    except Exception:
        # If KF initialization fails, continue without it
        pass
    self.track_to_object_mapping[track_key] = object_id
    self.stats['objects_tracked'] += 1

def _wm_find_matching_object(self, world_pos: Tuple[float, float, float], class_id: int, current_time: float, feature_vector=None, camera_id: int = -1) -> Optional[int]:
    """Find existing object that might match this position.

    Two matching paths:
      1. Spatial proximity (< 2 m) — works for calibrated cameras.
      2. Appearance similarity — works cross-camera even when world
         coordinates are unreliable (uncalibrated cameras).
    """
    min_score = float('inf')
    best_match = None
    spatial_threshold = 2.0    # calibrated-camera gate
    appearance_gate   = 12.0   # allow appearance match up to 12 m world-dist

    for object_id, obj in self.world_objects.items():
        if obj.class_id != class_id:
            continue

        # Skip objects updated <100 ms ago by the SAME camera (avoid double-match)
        # but allow cross-camera matching on the same tick
        if (current_time - obj.last_update < 0.1
                and camera_id >= 0 and camera_id == obj.last_seen_camera):
            continue

        spatial_dist = np.sqrt(
            (obj.world_position[0] - world_pos[0])**2 +
            (obj.world_position[1] - world_pos[1])**2 +
            (obj.world_position[2] - world_pos[2])**2
        )

        has_appearance = (feature_vector is not None and obj.feature_vector is not None)
        cosine_sim = 0.0
        if has_appearance:
            cosine_sim = float(np.dot(feature_vector, obj.feature_vector))

        if spatial_dist < spatial_threshold:
            # Path 1 — close in world space (calibrated)
            score = spatial_dist / spatial_threshold
            if has_appearance:
                score = 0.4 * score + 0.6 * (1.0 - cosine_sim)
        elif has_appearance and cosine_sim > 0.45 and spatial_dist < appearance_gate:
            # Path 2 — far apart but looks like same person (uncalibrated / cross-cam)
            score = (1.0 - cosine_sim) + 0.05   # small penalty for no spatial confirmation
        else:
            continue

        if score < min_score:
            min_score = score
            best_match = object_id

    return best_match

def _wm_cleanup_old_objects(self, current_time: float, max_age: float = 5.0):
    """Remove objects that haven't been seen recently"""
    to_remove = []
    
    for object_id, obj in self.world_objects.items():
        if current_time - obj.last_update > max_age:
            to_remove.append(object_id)
    
    for object_id in to_remove:
        del self.world_objects[object_id]
        # Remove from track mapping
        to_remove_tracks = [
            track_key for track_key, obj_id in self.track_to_object_mapping.items() 
            if obj_id == object_id
        ]
        for track_key in to_remove_tracks:
            del self.track_to_object_mapping[track_key]
        # Remove associated Kalman filter
        if object_id in self.kalman_filters:
            del self.kalman_filters[object_id]

def _wm_generate_predictions_for_camera(self, camera_id: int, current_time: float) -> List[PredictedTarget]:
    """Generate predicted targets for a specific camera.

    Three projection paths (tried in order of accuracy):
      A) Homography — tries ALL cameras that currently see the person,
         looking for any valid H to the target camera.
      B) Pixel extrapolation — if this camera previously saw the person,
         slide last-known position by velocity × time (adaptive budget).
      C) World-coordinate projection — project the fused 3D world
         position through the camera model.  Rough but always works.
    """
    predictions = []
    fw, fh = float(settings.frame_width), float(settings.frame_height)

    for object_id, obj in self.world_objects.items():
        # Skip objects currently tracked by this camera
        if camera_id in obj.source_tracks:
            continue

        # Global time since any camera last saw this object
        time_since_global = current_time - obj.last_update
        if time_since_global > self.prediction_horizon:
            continue

        predicted_pixel = None
        bw, bh = 100.0, 200.0
        pv = (0.0, 0.0)
        method = 'EXTRAP'
        time_since_seen = current_time - obj.camera_last_seen.get(camera_id, obj.last_update)
        best_source_cam = obj.last_seen_camera

        # ── Path A: Cross-camera HOMOGRAPHY (try ALL source cams) ─
        # Iterate every camera that has recent pixel data for this
        # object and a valid homography to the target camera.
        # Pick the freshest successful projection.
        candidate_cams = set()
        # Primary: cameras currently tracking this object
        for c_id in obj.source_tracks:
            if c_id != camera_id:
                candidate_cams.add(c_id)
        # Secondary: any camera with recent pixel data
        for c_id, t in obj.camera_last_seen.items():
            if c_id != camera_id and (current_time - t) < 1.5:
                candidate_cams.add(c_id)

        best_h_result = None
        best_h_freshness = 999.0
        for src_cam in candidate_cams:
            src_age = current_time - obj.camera_last_seen.get(src_cam, 0)
            if src_age > 1.5:
                continue
            if src_cam not in obj.camera_pixel_positions:
                continue
            if not self.cross_camera_homography.has_homography(src_cam, camera_id):
                continue
            src_pos = obj.camera_pixel_positions[src_cam]
            src_bsz = obj.camera_bbox_sizes.get(src_cam, obj.bbox_size)
            src_foot = (src_pos[0], src_pos[1] + src_bsz[1] / 2.0)
            result = self.cross_camera_homography.project_bbox(
                src_cam, camera_id, src_foot, src_bsz[1], src_bsz[0]
            )
            if result is not None and src_age < best_h_freshness:
                best_h_result = result
                best_h_freshness = src_age
                best_source_cam = src_cam

        if best_h_result is not None:
            dst_foot, est_w, est_h = best_h_result
            cx, cy = dst_foot[0], dst_foot[1] - est_h / 2.0
            bw, bh = est_w, est_h
            predicted_pixel = (cx, cy)
            method = 'HOMOGRAPHY'
            time_since_seen = best_h_freshness

        # ── Path B: Pixel-space EXTRAPOLATION (adaptive budget) ───
        if predicted_pixel is None and camera_id in obj.camera_pixel_positions:
            cam_time = current_time - obj.camera_last_seen.get(camera_id, obj.last_update)
            if cam_time <= self.prediction_horizon:
                time_since_seen = cam_time
                last_pos = obj.camera_pixel_positions[camera_id]
                pv = obj.camera_pixel_velocities.get(camera_id, (0.0, 0.0))
                dx = pv[0] * time_since_seen
                dy = pv[1] * time_since_seen
                # Adaptive budget: further extrapolation with time, capped at 250px
                max_extrap = min(250.0, 80.0 + 40.0 * time_since_seen)
                extrap_dist = (dx**2 + dy**2) ** 0.5
                if extrap_dist > max_extrap:
                    s = max_extrap / max(extrap_dist, 1e-6)
                    dx *= s
                    dy *= s
                predicted_pixel = (last_pos[0] + dx, last_pos[1] + dy)
                method = 'EXTRAP'
                bw_c, bh_c = obj.camera_bbox_sizes.get(camera_id, obj.bbox_size)
                if bw_c >= 10 and bh_c >= 10:
                    bw, bh = bw_c, bh_c

        # ── Path C: World-coordinate PROJECTION (rough fallback) ──
        # Uses the fused 3D world position and a simple pinhole model
        # to place the ghost.  Less accurate but ALWAYS works even if
        # this camera has never seen the person and no homography exists.
        if predicted_pixel is None:
            wp = obj.world_position
            pixel = self.coordinate_transform.world_to_pixel(camera_id, wp)
            if pixel is not None:
                px_x, px_y = pixel
                # Sanity: only accept if the projection lands roughly on-screen
                if -fw * 0.5 <= px_x <= fw * 1.5 and -fh * 0.5 <= px_y <= fh * 1.5:
                    predicted_pixel = (px_x, px_y)
                    method = 'WORLD'
                    time_since_seen = time_since_global
                    # Estimate bbox size from world distance
                    depth_est = max(abs(wp[2] - 2.0), 0.5)  # rough depth
                    bh = min(fh * 0.8, max(80.0, 500.0 / depth_est))
                    bw = bh * 0.4

        # If all three paths failed, skip this object
        if predicted_pixel is None:
            continue

        # ── Common: clamp, bbox, confidence, keypoints ────────────
        px = max(bw / 2, min(fw - bw / 2, predicted_pixel[0]))
        py = max(bh / 2, min(fh - bh / 2, predicted_pixel[1]))
        predicted_pixel = (px, py)

        predicted_bbox = (px - bw / 2, py - bh / 2, px + bw / 2, py + bh / 2)

        # Confidence: homography is best, world is roughest
        if method == 'HOMOGRAPHY':
            confidence = obj.confidence * max(0.3, 1.0 - time_since_seen / 3.0)
        elif method == 'EXTRAP':
            confidence = obj.confidence * max(0.1, 1.0 - time_since_seen / self.prediction_horizon)
        else:  # WORLD
            confidence = obj.confidence * max(0.15, 0.6 - time_since_seen / self.prediction_horizon)

        # Project keypoints from best source
        projected_kp = None
        source_kp = obj.camera_keypoints.get(best_source_cam) or obj.keypoints
        src_bsz = obj.camera_bbox_sizes.get(best_source_cam, obj.bbox_size)
        if source_kp and len(source_kp) > 0:
            projected_kp = _project_keypoints_to_camera(
                source_kp, src_bsz, predicted_pixel, (bw, bh)
            )

        prediction = PredictedTarget(
            object_id=object_id,
            camera_id=camera_id,
            predicted_bbox=predicted_bbox,
            predicted_center=predicted_pixel,
            confidence=confidence,
            time_since_seen=time_since_seen,
            velocity_projection=(pv[0], pv[1]),
            source_camera=best_source_cam,
            keypoints=projected_kp,
            homography_source=(method == 'HOMOGRAPHY'),
            prediction_method=method,
        )

        predictions.append(prediction)
        self.stats['predictions_generated'] += 1

    return predictions

def _wm_get_world_objects(self) -> List[WorldObject]:
    """Get all current world objects"""
    return list(self.world_objects.values())

def _wm_get_object_history(self, object_id: int) -> List[dict]:
    """Get history for a specific object"""
    return list(self.object_history.get(object_id, []))

def _wm_get_stats(self) -> dict:
    """Get world model statistics"""
    return {
        'total_objects': len(self.world_objects),
        'objects_tracked': self.stats['objects_tracked'],
        'fused_objects': self.stats['fused_objects'],
        'predictions_generated': self.stats['predictions_generated'],
        'coordinate_transforms': self.stats['coordinate_transforms'],
        'active_cameras': len(set(obj.last_seen_camera for obj in self.world_objects.values()))
    }


def _project_keypoints_to_camera(src_kps: list, src_bbox_size: Tuple[float, float],
                                   target_center: Tuple[float, float],
                                   target_bbox_size: Tuple[float, float]) -> list:
    """Project keypoints from source camera view onto the predicted position.

    Uses relative-offset approach:
      1. Compute each joint's offset from the source bbox center in normalised coords
      2. Apply same offset around the predicted center in the target view, scaled by
         the target bbox size.
    This avoids needing per-joint 3D reconstruction.
    """
    sw, sh = max(src_bbox_size[0], 1.0), max(src_bbox_size[1], 1.0)
    tw, th = max(target_bbox_size[0], 1.0), max(target_bbox_size[1], 1.0)

    # Compute source bbox center from keypoints with confidence > 0.3
    vis = [(kp[0], kp[1]) for kp in src_kps if kp[2] > 0.3]
    if not vis:
        return None
    src_cx = sum(p[0] for p in vis) / len(vis)
    src_cy = sum(p[1] for p in vis) / len(vis)

    projected = []
    for kp in src_kps:
        x, y, conf = kp
        if conf < 0.05:
            projected.append((0.0, 0.0, 0.0))
            continue
        # Normalised offset relative to source bbox
        nx = (x - src_cx) / sw
        ny = (y - src_cy) / sh
        # Reconstruct in target view
        tx = target_center[0] + nx * tw
        ty = target_center[1] + ny * th
        projected.append((float(tx), float(ty), float(conf)))
    return projected


def _wm_get_homography_stats(self) -> dict:
    """Return cross-camera homography status for UI / debugging."""
    return self.cross_camera_homography.get_all_stats()

# Attach all methods to WorldModel
WorldModel.update_with_tracking_results = _wm_update_with_tracking_results
WorldModel._process_camera_tracks = _wm_process_camera_tracks
WorldModel._update_existing_object = _wm_update_existing_object
WorldModel._create_new_object = _wm_create_new_object
WorldModel._find_matching_object = _wm_find_matching_object
WorldModel._cleanup_old_objects = _wm_cleanup_old_objects
WorldModel.generate_predictions_for_camera = _wm_generate_predictions_for_camera
WorldModel.get_world_objects = _wm_get_world_objects
WorldModel.get_object_history = _wm_get_object_history
WorldModel.get_stats = _wm_get_stats
WorldModel.get_homography_stats = _wm_get_homography_stats