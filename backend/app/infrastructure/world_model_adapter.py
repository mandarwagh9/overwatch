"""
World model repository adapter.
Handles sensor fusion, coordinate transforms, and predictions.
"""
from __future__ import annotations
import asyncio
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from app.application.ports import WorldModelRepository, ConfigurationRepository
from app.domain.entities import (
    Track, WorldObject, PredictedTarget, CameraCalibration,
    Point3D, Velocity3D, BoundingBox, Keypoint, PredictionMethod
)


logger = logging.getLogger(__name__)


@dataclass
class KalmanFilter:
    """Constant velocity Kalman filter for 3D tracking."""
    
    # State: [x, y, z, vx, vy, vz]
    state: NDArray[np.float64] = field(default_factory=lambda: np.zeros(6))
    covariance: NDArray[np.float64] = field(default_factory=lambda: np.eye(6))
    
    # Process noise
    q_pos: float = 0.1
    q_vel: float = 1.0
    
    # Measurement noise (base)
    r_base: float = 0.5
    
    def predict(self, dt: float) -> None:
        """Predict state forward by dt seconds."""
        # State transition matrix
        F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Process noise
        Q = np.zeros((6, 6))
        Q[0:3, 0:3] = np.eye(3) * (self.q_pos * dt)
        Q[3:6, 3:6] = np.eye(3) * (self.q_vel * dt)
        
        # Predict
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q
    
    def update(
        self,
        measurement: Point3D,
        confidence: float = 1.0,
        sensor_trust: float = 1.0
    ) -> None:
        """Update with measurement."""
        # Measurement matrix (we measure position only)
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Adaptive measurement noise
        quality = max(confidence, 0.1) * max(sensor_trust, 0.1)
        R = np.eye(3) * (self.r_base / quality)
        
        # Kalman gain
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update
        z = np.array([measurement.x, measurement.y, measurement.z])
        y = z - H @ self.state
        self.state = self.state + K @ y
        self.covariance = (np.eye(6) - K @ H) @ self.covariance
    
    @property
    def position(self) -> Point3D:
        return Point3D(
            float(self.state[0]),
            float(self.state[1]),
            float(self.state[2])
        )
    
    @property
    def velocity(self) -> Velocity3D:
        return Velocity3D(
            float(self.state[3]),
            float(self.state[4]),
            float(self.state[5])
        )
    
    def predict_future(self, dt: float) -> Point3D:
        """Predict position at future time."""
        F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        future_state = F @ self.state
        return Point3D(
            float(future_state[0]),
            float(future_state[1]),
            float(future_state[2])
        )


class CoordinateTransformer:
    """Handles coordinate transformations between camera and world space."""
    
    def __init__(self, config_repo: ConfigurationRepository):
        self._config = config_repo
        self._calibrations: Dict[int, CameraCalibration] = {}
        self._rotation_cache: Dict[int, NDArray] = {}
    
    def set_calibration(self, calibration: CameraCalibration) -> None:
        """Set calibration for a camera."""
        self._calibrations[calibration.camera_id] = calibration
        
        # Pre-compute rotation matrix
        roll, pitch, yaw = calibration.rotation
        R = self._build_rotation_matrix(roll, pitch, yaw)
        self._rotation_cache[calibration.camera_id] = R
    
    def pixel_to_world(
        self,
        camera_id: int,
        pixel: Tuple[float, float],
        depth: float
    ) -> Optional[Point3D]:
        """Convert pixel coordinates to world coordinates."""
        if camera_id not in self._calibrations:
            return None
        
        calib = self._calibrations[camera_id]
        
        # Get rotation matrix inverse
        R = self._rotation_cache.get(camera_id)
        if R is None:
            return None
        R_inv = R.T
        
        # Pixel to normalized camera coordinates
        x_norm = (pixel[0] - calib.image_center[0]) / calib.focal_length
        y_norm = (pixel[1] - calib.image_center[1]) / calib.focal_length
        
        # Ray in camera frame
        ray_cam = np.array([x_norm, y_norm, 1.0])
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        
        # Transform to world frame
        ray_world = R_inv @ ray_cam
        
        # Position in world
        cam_pos = np.array([calib.position.x, calib.position.y, calib.position.z])
        world_pos = cam_pos + ray_world * depth
        
        return Point3D(
            float(world_pos[0]),
            float(world_pos[1]),
            float(world_pos[2])
        )
    
    def world_to_pixel(
        self,
        camera_id: int,
        world_pos: Point3D
    ) -> Optional[Tuple[float, float]]:
        """Convert world coordinates to pixel coordinates."""
        if camera_id not in self._calibrations:
            return None
        
        calib = self._calibrations[camera_id]
        R = self._rotation_cache.get(camera_id)
        if R is None:
            return None
        
        # Vector from camera to world point
        cam_pos = np.array([calib.position.x, calib.position.y, calib.position.z])
        world_arr = np.array([world_pos.x, world_pos.y, world_pos.z])
        rel_world = world_arr - cam_pos
        
        # Transform to camera frame
        rel_cam = R @ rel_world
        
        # Check if behind camera
        if rel_cam[2] <= 0.01:
            return None
        
        # Project to pixel
        x_norm = rel_cam[0] / rel_cam[2]
        y_norm = rel_cam[1] / rel_cam[2]
        
        pixel_x = x_norm * calib.focal_length + calib.image_center[0]
        pixel_y = y_norm * calib.focal_length + calib.image_center[1]
        
        return (float(pixel_x), float(pixel_y))
    
    def _build_rotation_matrix(
        self,
        roll: float,
        pitch: float,
        yaw: float
    ) -> NDArray:
        """Build rotation matrix from Euler angles."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        
        # World to camera rotation
        R = Rz @ Ry @ Rx
        return R


class WorldModelRepositoryImpl(WorldModelRepository):
    """Implementation of world model repository."""
    
    def __init__(self, config_repo: ConfigurationRepository):
        self._config = config_repo
        self._transformer = CoordinateTransformer(config_repo)
        self._kalman_filters: Dict[int, KalmanFilter] = {}
        self._world_objects: Dict[int, WorldObject] = {}
        self._track_to_object: Dict[Tuple[int, int], int] = {}  # (cam_id, track_id) -> obj_id
        self._next_object_id = 1
        
        # Configuration
        self._person_height = config_repo.get_float("person_height_meters", 1.7)
        self._max_age = config_repo.get_float("world_object_max_age_seconds", 5.0)
        self._prediction_horizon = config_repo.get_float("prediction_horizon_seconds", 5.0)
        
        # Initialize default calibrations if positions provided
        self._init_default_calibrations()
        
        logger.info("WorldModelRepositoryImpl initialized")
    
    def _init_default_calibrations(self) -> None:
        """Initialize default camera calibrations from config."""
        positions = self._config.get_list("camera_positions", [])
        
        if not positions:
            logger.warning("No camera positions configured")
            return
        
        for i, pos in enumerate(positions):
            if len(pos) != 3:
                logger.warning(f"Invalid camera position {i}: {pos}")
                continue
            
            # Default calibration with position
            calibration = CameraCalibration(
                camera_id=i,
                position=Point3D(pos[0], pos[1], pos[2]),
                rotation=(0.0, 0.0, 0.0),  # Default rotation
                focal_length=800.0,
                image_center=(640.0, 360.0)
            )
            self._transformer.set_calibration(calibration)
            logger.info(f"Camera {i} calibrated at position ({pos[0]}, {pos[1]}, {pos[2]})")
    
    async def initialize(self) -> None:
        """Initialize the world model."""
        logger.info("World model initialized")
    
    async def update(self, tracks: Dict[int, List[Track]]) -> List[WorldObject]:
        """Update world model with new tracking results."""
        now = datetime.now()
        
        for camera_id, camera_tracks in tracks.items():
            for track in camera_tracks:
                await self._process_track(camera_id, track, now)
        
        # Clean up old objects
        self._cleanup_old_objects(now)
        
        world_objs = list(self._world_objects.values())
        logger.info(f"🌍 World objects: {len(world_objs)}")
        
        return world_objs
    
    async def _process_track(
        self,
        camera_id: int,
        track: Track,
        timestamp: datetime
    ) -> None:
        """Process a single track update."""
        # Estimate depth from bbox height
        bbox_height = track.bbox.height
        if bbox_height > 20:
            calib = self._transformer._calibrations.get(camera_id)
            if calib:
                focal = calib.focal_length
                depth = (self._person_height * focal) / bbox_height
                depth = max(0.5, min(depth, 100.0))
            else:
                depth = 3.0
        else:
            depth = 3.0
        
        # Convert to world coordinates
        world_pos = self._transformer.pixel_to_world(
            camera_id,
            track.bbox.center,
            depth
        )
        
        if world_pos is None:
            return
        
        # Check if track is already associated
        track_key = (camera_id, track.track_id)
        
        if track_key in self._track_to_object:
            object_id = self._track_to_object[track_key]
            self._update_existing_object(object_id, track, world_pos, timestamp)
        else:
            # Check for nearby objects (cross-camera matching)
            existing_id = self._find_matching_object(world_pos, track.class_id)
            if existing_id:
                self._track_to_object[track_key] = existing_id
                self._update_existing_object(existing_id, track, world_pos, timestamp)
            else:
                self._create_new_object(track_key, track, world_pos, timestamp)
    
    def _update_existing_object(
        self,
        object_id: int,
        track: Track,
        world_pos: Point3D,
        timestamp: datetime
    ) -> None:
        """Update an existing world object."""
        if object_id not in self._world_objects:
            return
        
        obj = self._world_objects[object_id]
        
        # Update Kalman filter
        if object_id in self._kalman_filters:
            kf = self._kalman_filters[object_id]
            dt = (timestamp - obj.last_update).total_seconds()
            kf.predict(dt)
            kf.update(world_pos, confidence=track.confidence)
            
            obj.position = kf.position
            obj.velocity = kf.velocity
        else:
            # Fallback without KF
            dt = (timestamp - obj.last_update).total_seconds()
            if dt > 0:
                velocity = Velocity3D(
                    (world_pos.x - obj.position.x) / dt,
                    (world_pos.y - obj.position.y) / dt,
                    (world_pos.z - obj.position.z) / dt
                )
                obj.velocity = velocity
            obj.position = world_pos
        
        obj.confidence = track.confidence
        obj.last_seen_camera = track.camera_id
        obj.last_update = timestamp
        obj.source_tracks[track.camera_id] = track.track_id
        
        # Update pixel tracking
        obj.camera_pixel_positions[track.camera_id] = track.bbox.center
        obj.camera_pixel_velocities[track.camera_id] = track.velocity
        obj.camera_last_seen[track.camera_id] = timestamp
    
    def _create_new_object(
        self,
        track_key: Tuple[int, int],
        track: Track,
        world_pos: Point3D,
        timestamp: datetime
    ) -> None:
        """Create a new world object."""
        object_id = self._next_object_id
        self._next_object_id += 1
        
        obj = WorldObject(
            object_id=object_id,
            position=world_pos,
            velocity=Velocity3D(0.0, 0.0, 0.0),
            class_id=track.class_id,
            class_name=track.class_name,
            confidence=track.confidence,
            last_seen_camera=track.camera_id,
            last_update=timestamp,
            source_tracks={track.camera_id: track.track_id},
            camera_pixel_positions={track.camera_id: track.bbox.center},
            camera_pixel_velocities={track.camera_id: track.velocity},
            camera_last_seen={track.camera_id: timestamp}
        )
        
        self._world_objects[object_id] = obj
        self._track_to_object[track_key] = object_id
        
        # Create Kalman filter
        self._kalman_filters[object_id] = KalmanFilter()
        self._kalman_filters[object_id].state[0:3] = [world_pos.x, world_pos.y, world_pos.z]
        
        logger.debug(f"Created new world object {object_id}")
    
    def _find_matching_object(
        self,
        world_pos: Point3D,
        class_id: int
    ) -> Optional[int]:
        """Find existing object that matches position."""
        best_match = None
        min_distance = float('inf')
        threshold = 2.0  # meters
        
        for obj_id, obj in self._world_objects.items():
            if obj.class_id != class_id:
                continue
            
            distance = obj.position.distance_to(world_pos)
            if distance < min_distance and distance < threshold:
                min_distance = distance
                best_match = obj_id
        
        return best_match
    
    def _cleanup_old_objects(self, current_time: datetime) -> None:
        """Remove objects not seen recently."""
        to_remove = []
        
        for obj_id, obj in self._world_objects.items():
            age = (current_time - obj.last_update).total_seconds()
            if age > self._max_age:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self._world_objects[obj_id]
            if obj_id in self._kalman_filters:
                del self._kalman_filters[obj_id]
            
            # Remove from track mapping
            keys_to_remove = [
                k for k, v in self._track_to_object.items() if v == obj_id
            ]
            for k in keys_to_remove:
                del self._track_to_object[k]
    
    def get_world_objects(self) -> List[WorldObject]:
        """Get all current world objects."""
        return list(self._world_objects.values())
    
    def generate_predictions(self, camera_id: int) -> List[PredictedTarget]:
        """Generate predictions for a camera view."""
        predictions = []
        now = datetime.now()
        
        for obj in self._world_objects.values():
            # Skip if already seen by this camera
            if camera_id in obj.source_tracks:
                continue
            
            time_since_seen = (now - obj.last_update).total_seconds()
            if time_since_seen > self._prediction_horizon:
                continue
            
            # Try world-to-pixel projection
            pixel = self._transformer.world_to_pixel(camera_id, obj.position)
            
            if pixel is None:
                continue
            
            # Estimate bbox size from depth
            depth = abs(obj.position.z)
            bbox_height = min(500, max(50, 500 / max(depth, 0.5)))
            bbox_width = bbox_height * 0.4
            
            bbox = BoundingBox(
                pixel[0] - bbox_width / 2,
                pixel[1] - bbox_height / 2,
                pixel[0] + bbox_width / 2,
                pixel[1] + bbox_height / 2
            )
            
            # Confidence decays with time
            confidence = obj.confidence * max(0.1, 1.0 - time_since_seen / self._prediction_horizon)
            
            prediction = PredictedTarget(
                object_id=obj.object_id,
                camera_id=camera_id,
                predicted_bbox=bbox,
                confidence=confidence,
                time_since_seen=time_since_seen,
                velocity_projection=(obj.velocity.vx, obj.velocity.vy),
                source_camera=obj.last_seen_camera,
                prediction_method=PredictionMethod.WORLD_PROJECTION
            )
            
            predictions.append(prediction)
        
        return predictions
    
    def update_camera_calibration(self, calibration: CameraCalibration) -> None:
        """Update calibration for a camera."""
        self._transformer.set_calibration(calibration)
        logger.info(f"Updated calibration for camera {calibration.camera_id}")
    
    def get_camera_calibration(self, camera_id: int) -> Optional[CameraCalibration]:
        """Get calibration for a camera."""
        return self._transformer._calibrations.get(camera_id)
