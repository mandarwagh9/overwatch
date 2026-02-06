"""
World Model for sensor fusion and coordinate transformation across multiple cameras
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque

from app.core.tracking_manager import TrackingResult, Track
from app.config import settings


@dataclass
class CameraCalibration:
    """Camera calibration and positioning information"""
    camera_id: int
    position: Tuple[float, float, float]  # x, y, z in world coordinates
    rotation: Tuple[float, float, float]  # roll, pitch, yaw in radians
    focal_length: float
    image_center: Tuple[float, float]  # cx, cy in pixels
    distortion: Optional[List[float]] = None


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


class CoordinateTransform:
    """Handles coordinate transformations between camera and world space"""
    
    def __init__(self):
        self.camera_calibrations: Dict[int, CameraCalibration] = {}
        
    def add_camera_calibration(self, calibration: CameraCalibration):
        """Add camera calibration data"""
        self.camera_calibrations[calibration.camera_id] = calibration
    
    def pixel_to_world(self, camera_id: int, pixel_coords: Tuple[float, float], depth: float = 1.0) -> Optional[Tuple[float, float, float]]:
        """Convert pixel coordinates to world coordinates"""
        if camera_id not in self.camera_calibrations:
            # Use simple geometric transformation for uncalibrated cameras
            return self._simple_pixel_to_world(camera_id, pixel_coords, depth)
        
        calib = self.camera_calibrations[camera_id]
        
        # Convert pixel to normalized camera coordinates
        x_norm = (pixel_coords[0] - calib.image_center[0]) / calib.focal_length
        y_norm = (pixel_coords[1] - calib.image_center[1]) / calib.focal_length
        
        # Apply rotation and translation to get world coordinates
        # Simplified transformation - in practice would use full camera matrix
        world_x = calib.position[0] + x_norm * depth
        world_y = calib.position[1] + y_norm * depth
        world_z = calib.position[2] + depth
        
        return (world_x, world_y, world_z)
    
    def world_to_pixel(self, camera_id: int, world_coords: Tuple[float, float, float]) -> Optional[Tuple[float, float]]:
        """Convert world coordinates to pixel coordinates"""
        if camera_id not in self.camera_calibrations:
            # Use simple geometric transformation for uncalibrated cameras
            return self._simple_world_to_pixel(camera_id, world_coords)
        
        calib = self.camera_calibrations[camera_id]
        
        # Transform world coordinates to camera coordinates
        # Simplified - in practice would use full projection matrix
        rel_x = world_coords[0] - calib.position[0]
        rel_y = world_coords[1] - calib.position[1]
        rel_z = world_coords[2] - calib.position[2]
        
        if rel_z <= 0:
            return None  # Point is behind camera
        
        # Project to image plane
        x_norm = rel_x / rel_z
        y_norm = rel_y / rel_z
        
        pixel_x = x_norm * calib.focal_length + calib.image_center[0]
        pixel_y = y_norm * calib.focal_length + calib.image_center[1]
        
        return (pixel_x, pixel_y)
    
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

    def update(self, meas_pos: Tuple[float, float, float]):
        z = np.array(meas_pos, dtype=float).reshape((3, 1))
        y = z - self.H.dot(self.x)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
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
    """Setup default camera calibrations for demo"""
    default_cameras = [
        CameraCalibration(0, (0, 0, 2), (0, 0, 0), 800, (640, 360)),
        CameraCalibration(1, (5, 0, 2), (0, 0, np.pi/2), 800, (640, 360)),
        CameraCalibration(2, (0, 5, 2), (0, 0, np.pi), 800, (640, 360)),
        CameraCalibration(3, (-5, 0, 2), (0, 0, -np.pi/2), 800, (640, 360))
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
    
    # Clean up old objects
    self._cleanup_old_objects(current_time)
    
    # Update statistics
    self.stats['fused_objects'] = len(self.world_objects)

async def _wm_process_camera_tracks(self, tracking_result: TrackingResult, current_time: float):
    """Process tracks from a single camera"""
    camera_id = tracking_result.camera_id
    
    for track in tracking_result.tracks:
        track_key = (camera_id, track.track_id)
        
        # Convert track position to world coordinates
        world_pos = self.coordinate_transform.pixel_to_world(
            camera_id, 
            track.center, 
            depth=1.0  # Assume 1 meter depth for 2D tracking
        )
        
        if world_pos is None:
            continue
        
        self.stats['coordinate_transforms'] += 1
        
        # Check if this track is already associated with a world object
        if track_key in self.track_to_object_mapping:
            object_id = self.track_to_object_mapping[track_key]
            await self._update_existing_object(object_id, track, world_pos, current_time)
        else:
            # Check for nearby objects that might be the same
            object_id = self._find_matching_object(world_pos, track.class_id, current_time)
            if object_id:
                self.track_to_object_mapping[track_key] = object_id
                await self._update_existing_object(object_id, track, world_pos, current_time)
            else:
                # Create new object
                await self._create_new_object(track_key, track, world_pos, current_time)

async def _wm_update_existing_object(self, object_id: int, track, world_pos: Tuple[float, float, float], current_time: float):
    """Update an existing world object"""
    if object_id not in self.world_objects:
        return
    
    obj = self.world_objects[object_id]
    # Use Kalman filter when available to smooth and update state
    if object_id in self.kalman_filters:
        kf = self.kalman_filters[object_id]
        # Predict to current time delta
        dt = max(1e-3, current_time - obj.last_update)
        kf.predict(dt)
        # Update with measurement
        kf.update(world_pos)
        obj.world_position = kf.current_position()
        obj.velocity = kf.current_velocity()
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

    obj.confidence = max(obj.confidence, track.confidence)
    obj.last_seen_camera = track.camera_id
    obj.last_update = current_time
    obj.prediction_confidence = 1.0

    # Update source tracks
    obj.source_tracks[track.camera_id] = track.track_id

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
    
    new_object = WorldObject(
        object_id=object_id,
        world_position=world_pos,
        velocity=(0.0, 0.0, 0.0),
        class_id=track.class_id,
        class_name=track.class_name,
        confidence=track.confidence,
        last_seen_camera=track.camera_id,
        last_update=current_time,
        source_tracks={track.camera_id: track.track_id}
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

def _wm_find_matching_object(self, world_pos: Tuple[float, float, float], class_id: int, current_time: float) -> Optional[int]:
    """Find existing object that might match this position"""
    min_distance = float('inf')
    best_match = None
    threshold = 2.0  # 2 meter threshold
    
    for object_id, obj in self.world_objects.items():
        if obj.class_id != class_id:
            continue
        
        # Skip objects that were updated very recently from other cameras
        if current_time - obj.last_update < 0.1:  # 100ms
            continue
        
        distance = np.sqrt(
            (obj.world_position[0] - world_pos[0])**2 +
            (obj.world_position[1] - world_pos[1])**2 +
            (obj.world_position[2] - world_pos[2])**2
        )
        
        if distance < min_distance and distance < threshold:
            min_distance = distance
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
    """Generate predicted targets for a specific camera"""
    predictions = []
    
    for object_id, obj in self.world_objects.items():
        # Skip objects currently visible in this camera
        if camera_id in obj.source_tracks:
            continue
        
        time_since_seen = current_time - obj.last_update
        
        # Only predict for recently seen objects
        if time_since_seen > self.prediction_horizon:
            continue
        
        # Predict future position
        # Use Kalman filter to predict future world position when available
        if object_id in self.kalman_filters:
            kf = self.kalman_filters[object_id]
            predicted_world_pos = kf.predict_future(time_since_seen)
        else:
            predicted_world_pos = (
                obj.world_position[0] + obj.velocity[0] * time_since_seen,
                obj.world_position[1] + obj.velocity[1] * time_since_seen,
                obj.world_position[2] + obj.velocity[2] * time_since_seen
            )
        
        # Convert to camera coordinates
        predicted_pixel = self.coordinate_transform.world_to_pixel(camera_id, predicted_world_pos)
        
        if predicted_pixel is None:
            continue
        
        # Check if prediction is within camera frame
        if (0 <= predicted_pixel[0] <= settings.frame_width and 
            0 <= predicted_pixel[1] <= settings.frame_height):
            
            # Create predicted bounding box (simplified)
            box_size = 100  # pixels
            predicted_bbox = (
                predicted_pixel[0] - box_size/2,
                predicted_pixel[1] - box_size/2,
                predicted_pixel[0] + box_size/2,
                predicted_pixel[1] + box_size/2
            )
            
            # Calculate confidence decay
            confidence = obj.confidence * max(0.1, 1.0 - time_since_seen / self.prediction_horizon)
            
            prediction = PredictedTarget(
                object_id=object_id,
                camera_id=camera_id,
                predicted_bbox=predicted_bbox,
                predicted_center=predicted_pixel,
                confidence=confidence,
                time_since_seen=time_since_seen,
                velocity_projection=(obj.velocity[0], obj.velocity[1])
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