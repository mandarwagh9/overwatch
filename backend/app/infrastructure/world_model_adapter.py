"""
World model repository adapter.
Handles sensor fusion, coordinate transforms, and predictions.
"""
from __future__ import annotations
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from app.application.ports import WorldModelRepository, ConfigurationRepository
from app.domain.entities import (
    Track, WorldObject, PredictedTarget, CameraCalibration,
    Point3D, Velocity3D, BoundingBox, PredictionMethod, AppearanceDescriptor
)
from app.infrastructure.homography import HomographyEstimator


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

        # Default calibration (used when CAMERA_POSITIONS is not configured)
        self._default_focal_length = 800.0
        self._default_camera_height = 2.5
        self._default_camera_spacing = 3.0
        self._warned_default_calibration = False

        # Cross-camera appearance re-ID
        self._appearance_match_threshold = config_repo.get_float(
            "cross_camera_appearance_threshold", 0.5
        )
        self._appearance_ema_alpha = 0.3

        # Cross-camera homography (Path A green H-PROJ ghost predictions)
        self._homography = HomographyEstimator(
            min_pairs=config_repo.get_int("homography_min_pairs", 4),
            max_pairs=config_repo.get_int("homography_max_pairs", 100),
            ransac_threshold=config_repo.get_float("homography_ransac_threshold", 12.0),
        )

        # Pixel-extrapolation ghosts (Path B red EXTRAP)
        fps = config_repo.get_float("target_fps", 24.0)
        self._extrap_fps = fps if fps > 0 else 24.0
        # A camera seen within this window is "live"; longer means it has lost the
        # object and becomes eligible for a ghost prediction.
        self._live_track_seconds = 2.0 / self._extrap_fps

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

    def _ensure_calibration(self, camera_id: int) -> None:
        """Ensure ``camera_id`` has a calibration, synthesizing a default if not.

        Without ``CAMERA_POSITIONS`` the world model would otherwise produce no
        world objects or predictions at all (``pixel_to_world`` returns ``None``
        for an uncalibrated camera). The default spreads cameras along the x-axis
        so multi-camera setups stay distinct; set ``CAMERA_POSITIONS`` for accuracy.
        """
        if camera_id in self._transformer._calibrations:
            return

        if not self._warned_default_calibration:
            logger.warning(
                "No CAMERA_POSITIONS configured; using auto-default camera "
                "calibration. World coordinates are approximate — set "
                "CAMERA_POSITIONS for accuracy."
            )
            self._warned_default_calibration = True

        position = Point3D(
            camera_id * self._default_camera_spacing, 0.0, self._default_camera_height
        )
        self._transformer.set_calibration(
            CameraCalibration(
                camera_id=camera_id,
                position=position,
                rotation=(0.0, 0.0, 0.0),
                focal_length=self._default_focal_length,
                image_center=(640.0, 360.0),
            )
        )
        logger.info(
            f"Camera {camera_id}: auto-default calibration at "
            f"({position.x}, {position.y}, {position.z})"
        )

    async def initialize(self) -> None:
        """Initialize the world model."""
        logger.info("World model initialized")
    
    async def update(self, tracks: Dict[int, List[Track]]) -> List[WorldObject]:
        """Update world model with new tracking results."""
        now = datetime.now()
        
        for camera_id, camera_tracks in tracks.items():
            for track in camera_tracks:
                await self._process_track(camera_id, track, now)

        # Feed cross-camera homography from objects co-visible this tick
        self._collect_correspondences(now)

        # Clean up old objects
        self._cleanup_old_objects(now)
        
        world_objs = list(self._world_objects.values())
        logger.debug(f"🌍 World objects: {len(world_objs)}")
        
        return world_objs
    
    async def _process_track(
        self,
        camera_id: int,
        track: Track,
        timestamp: datetime
    ) -> None:
        """Process a single track update."""
        # Ensure the camera has a calibration (auto-default if unconfigured),
        # otherwise pixel_to_world returns None and no world object is created.
        self._ensure_calibration(camera_id)

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
            # Check for nearby objects (cross-camera matching, appearance-gated)
            existing_id = self._find_matching_object(
                world_pos, track.class_id, track.appearance
            )
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
            dt = max(0.0, (timestamp - obj.last_update).total_seconds())
            kf.predict(dt)
            kf.update(world_pos, confidence=track.confidence)

            obj.position = kf.position
            obj.velocity = kf.velocity
        else:
            # Fallback without KF
            dt = max(0.0, (timestamp - obj.last_update).total_seconds())
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
        obj.camera_foot_points[track.camera_id] = (track.bbox.center[0], track.bbox.y2)

        # EMA-smooth the appearance descriptor for re-ID stability
        obj.appearance = self._blend_appearance(obj.appearance, track.appearance)
    
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
            appearance=track.appearance,
            source_tracks={track.camera_id: track.track_id},
            camera_pixel_positions={track.camera_id: track.bbox.center},
            camera_pixel_velocities={track.camera_id: track.velocity},
            camera_last_seen={track.camera_id: timestamp},
            camera_foot_points={track.camera_id: (track.bbox.center[0], track.bbox.y2)},
        )
        
        self._world_objects[object_id] = obj
        self._track_to_object[track_key] = object_id
        
        # Create Kalman filter
        self._kalman_filters[object_id] = KalmanFilter()
        self._kalman_filters[object_id].state[0:3] = [world_pos.x, world_pos.y, world_pos.z]
        
        logger.debug(f"Created new world object {object_id}")
    
    def _blend_appearance(
        self,
        old: Optional[AppearanceDescriptor],
        new: Optional[AppearanceDescriptor],
    ) -> Optional[AppearanceDescriptor]:
        """EMA-blend two appearance descriptors (alpha weights the new observation)."""
        if new is None:
            return old
        if old is None:
            return new
        a = self._appearance_ema_alpha
        blended = (1.0 - a) * old.vector + a * new.vector
        norm = float(np.linalg.norm(blended)) + 1e-6
        return AppearanceDescriptor(vector=(blended / norm).astype(np.float32))

    def _find_matching_object(
        self,
        world_pos: Point3D,
        class_id: int,
        appearance: Optional[AppearanceDescriptor] = None,
    ) -> Optional[int]:
        """Find an existing world object matching this observation.

        Candidates must share the class and lie within ``distance_threshold`` metres.
        When both the candidate and the observation carry an appearance descriptor,
        a cosine similarity below ``_appearance_match_threshold`` rejects the match —
        so two differently-dressed people at the same spot stay separate. With no
        appearance available it falls back to nearest-within-threshold.
        """
        distance_threshold = 2.0  # metres
        best_match = None
        best_score = -1.0

        for obj_id, obj in self._world_objects.items():
            if obj.class_id != class_id:
                continue

            distance = obj.position.distance_to(world_pos)
            if distance >= distance_threshold:
                continue

            if appearance is not None and obj.appearance is not None:
                similarity = obj.appearance.cosine_similarity(appearance)
                if similarity < self._appearance_match_threshold:
                    continue
                score = similarity
            else:
                # No appearance to compare — prefer the closest candidate.
                score = 1.0 - (distance / distance_threshold)

            if score > best_score:
                best_score = score
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

    def _collect_correspondences(self, now: datetime) -> None:
        """Feed foot-point correspondences for objects co-visible on this tick.

        An object seen by two cameras at the same instant gives one matched
        ground-plane point per camera pair, which the homography estimator uses
        to self-calibrate the camera-to-camera transform.
        """
        for obj in self._world_objects.values():
            cams = [
                c for c, seen in obj.camera_last_seen.items()
                if seen == now and c in obj.camera_foot_points
            ]
            if len(cams) < 2:
                continue
            for src in cams:
                for dst in cams:
                    if src == dst:
                        continue
                    self._homography.add_correspondence(
                        src, dst,
                        obj.camera_foot_points[src],
                        obj.camera_foot_points[dst],
                    )

    def _try_homography_prediction(
        self, camera_id: int, obj: WorldObject, time_since_seen: float
    ) -> Optional[PredictedTarget]:
        """Project ``obj``'s foot point from a source camera into ``camera_id`` via
        a learned homography. Returns a HOMOGRAPHY prediction, or None if no usable
        homography/foot-point exists (caller then falls back to world projection)."""
        for src_cam in self._homography.source_cameras_for(camera_id):
            foot = obj.camera_foot_points.get(src_cam)
            if foot is None:
                continue
            projected = self._homography.project(src_cam, camera_id, foot)
            if projected is None:
                continue

            depth = max(abs(obj.position.z), 0.5)
            bbox_height = min(500, max(50, 500 / depth))
            bbox_width = bbox_height * 0.4
            fx, fy = projected
            try:
                # Foot point is the bottom-centre; the body extends upward.
                bbox = BoundingBox(fx - bbox_width / 2, fy - bbox_height, fx + bbox_width / 2, fy)
            except ValueError:
                continue

            confidence = obj.confidence * max(
                0.1, 1.0 - time_since_seen / self._prediction_horizon
            )
            return PredictedTarget(
                object_id=obj.object_id,
                camera_id=camera_id,
                predicted_bbox=bbox,
                confidence=confidence,
                time_since_seen=time_since_seen,
                velocity_projection=(obj.velocity.vx, obj.velocity.vy),
                source_camera=src_cam,
                prediction_method=PredictionMethod.HOMOGRAPHY,
            )
        return None

    def _try_extrapolation_prediction(
        self, camera_id: int, obj: WorldObject, time_since_seen: float
    ) -> Optional[PredictedTarget]:
        """Dead-reckon a ghost from this camera's last-known pixel position, sliding it
        along the per-camera pixel velocity with an adaptive budget. Returns None when
        this camera has no pixel history for the object (so the caller falls through)."""
        last_pixel = obj.camera_pixel_positions.get(camera_id)
        if last_pixel is None:
            return None

        vx, vy = obj.camera_pixel_velocities.get(camera_id, (0.0, 0.0))
        speed = (vx * vx + vy * vy) ** 0.5
        budget = min(250.0, 80.0 + 40.0 * time_since_seen)
        if speed < 1e-6:
            cx, cy = last_pixel
        else:
            disp = min(budget, speed * time_since_seen * self._extrap_fps)
            cx = last_pixel[0] + (vx / speed) * disp
            cy = last_pixel[1] + (vy / speed) * disp

        depth = max(abs(obj.position.z), 0.5)
        bbox_height = min(500, max(50, 500 / depth))
        bbox_width = bbox_height * 0.4
        try:
            bbox = BoundingBox(
                cx - bbox_width / 2, cy - bbox_height / 2,
                cx + bbox_width / 2, cy + bbox_height / 2,
            )
        except ValueError:
            return None

        confidence = obj.confidence * max(
            0.1, 1.0 - time_since_seen / self._prediction_horizon
        )
        return PredictedTarget(
            object_id=obj.object_id,
            camera_id=camera_id,
            predicted_bbox=bbox,
            confidence=confidence,
            time_since_seen=time_since_seen,
            velocity_projection=(vx, vy),
            source_camera=camera_id,
            prediction_method=PredictionMethod.EXTRAPOLATION,
        )

    def generate_predictions(self, camera_id: int) -> List[PredictedTarget]:
        """Generate predictions for a camera view."""
        # A view-only camera still needs a calibration to project world objects.
        self._ensure_calibration(camera_id)

        predictions = []
        now = datetime.now()
        
        for obj in self._world_objects.values():
            # Skip objects this camera is tracking live (seen within the live window).
            last_seen_here = obj.camera_last_seen.get(camera_id)
            if last_seen_here is not None and \
                    (now - last_seen_here).total_seconds() < self._live_track_seconds:
                continue

            time_since_seen = (now - obj.last_update).total_seconds()
            if time_since_seen > self._prediction_horizon:
                continue

            # Path A: cross-camera homography (green H-PROJ) — most accurate.
            homography_pred = self._try_homography_prediction(
                camera_id, obj, time_since_seen
            )
            if homography_pred is not None:
                predictions.append(homography_pred)
                continue

            # Path B: pixel extrapolation (red EXTRAP) — dead-reckon from this camera's
            # own last-known pixel position. Only fires if it saw the object before.
            extrap_pred = self._try_extrapolation_prediction(
                camera_id, obj, time_since_seen
            )
            if extrap_pred is not None:
                predictions.append(extrap_pred)
                continue

            # Path C: world-to-pixel projection (orange WORLD) — always-available fallback.
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
