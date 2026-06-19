"""
Domain entities for the Overwatch perception system.
These are pure Python objects with no external dependencies.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum, auto
from datetime import datetime
import numpy as np
from numpy.typing import NDArray


class TrackingState(Enum):
    """State of a track in its lifecycle."""
    TENTATIVE = auto()
    CONFIRMED = auto()
    COASTING = auto()
    DELETED = auto()


class PredictionMethod(Enum):
    """Method used for generating a prediction."""
    HOMOGRAPHY = "homography"
    EXTRAPOLATION = "extrapolation"
    WORLD_PROJECTION = "world_projection"
    NONE = "none"


@dataclass(frozen=True)
class BoundingBox:
    """Immutable bounding box in pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    
    def __post_init__(self):
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError(f"Invalid bbox: ({self.x1}, {self.y1}, {self.x2}, {self.y2})")
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def iou(self, other: BoundingBox) -> float:
        """Calculate Intersection over Union with another bbox."""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x1 >= x2 or y1 >= y2:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0
    
    def scale(self, scale_x: float, scale_y: float) -> BoundingBox:
        """Return a scaled version of this bbox."""
        return BoundingBox(
            self.x1 * scale_x,
            self.y1 * scale_y,
            self.x2 * scale_x,
            self.y2 * scale_y
        )


@dataclass(frozen=True)
class Point3D:
    """3D point in world coordinates."""
    x: float
    y: float
    z: float
    
    def distance_to(self, other: Point3D) -> float:
        """Euclidean distance to another point."""
        return np.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class Velocity3D:
    """3D velocity vector."""
    vx: float
    vy: float
    vz: float
    
    def magnitude(self) -> float:
        return np.sqrt(self.vx ** 2 + self.vy ** 2 + self.vz ** 2)
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.vx, self.vy, self.vz)


@dataclass
class AppearanceDescriptor:
    """Appearance feature vector for re-identification."""
    vector: NDArray[np.float32]
    
    def cosine_similarity(self, other: AppearanceDescriptor) -> float:
        """Compute cosine similarity with another descriptor."""
        dot = np.dot(self.vector, other.vector)
        norm1 = np.linalg.norm(self.vector)
        norm2 = np.linalg.norm(other.vector)
        return float(dot / (norm1 * norm2 + 1e-6))


@dataclass
class Keypoint:
    """Single keypoint (joint) with confidence."""
    x: float
    y: float
    confidence: float
    
    def is_visible(self, threshold: float = 0.3) -> bool:
        return self.confidence >= threshold


@dataclass
class Detection:
    """Single object detection from a camera frame."""
    detection_id: str
    camera_id: int
    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    timestamp: datetime
    appearance: Optional[AppearanceDescriptor] = None
    keypoints: List[Keypoint] = field(default_factory=list)
    
    @property
    def center(self) -> Tuple[float, float]:
        return self.bbox.center


@dataclass
class Track:
    """Tracked object across frames."""
    track_id: int
    camera_id: int
    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    state: TrackingState
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    velocity: Tuple[float, float] = (0.0, 0.0)
    predicted_position: Optional[Tuple[float, float]] = None
    appearance: Optional[AppearanceDescriptor] = None
    keypoints: List[Keypoint] = field(default_factory=list)
    
    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackingState.CONFIRMED
    
    @property
    def is_coasting(self) -> bool:
        return self.state == TrackingState.COASTING


@dataclass
class WorldObject:
    """Fused world-space object from multiple camera tracks."""
    object_id: int
    position: Point3D
    velocity: Velocity3D
    class_id: int
    class_name: str
    confidence: float
    last_seen_camera: int
    last_update: datetime
    
    # Uncertainty from Kalman filter
    position_uncertainty: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    
    # Source tracking
    source_tracks: Dict[int, int] = field(default_factory=dict)  # camera_id -> track_id
    
    # Appearance for re-ID
    appearance: Optional[AppearanceDescriptor] = None
    
    # Per-camera pixel-space data for predictions
    camera_pixel_positions: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    camera_pixel_velocities: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    camera_last_seen: Dict[int, datetime] = field(default_factory=dict)
    
    @property
    def time_since_update(self) -> float:
        return (datetime.now() - self.last_update).total_seconds()


@dataclass
class PredictedTarget:
    """Predicted target position for a camera view."""
    object_id: int
    camera_id: int
    predicted_bbox: BoundingBox
    confidence: float
    time_since_seen: float
    velocity_projection: Tuple[float, float]
    source_camera: int
    prediction_method: PredictionMethod
    keypoints: List[Keypoint] = field(default_factory=list)


@dataclass
class CameraCalibration:
    """Camera calibration data."""
    camera_id: int
    position: Point3D
    rotation: Tuple[float, float, float]  # roll, pitch, yaw in radians
    focal_length: float
    image_center: Tuple[float, float]
    distortion: Optional[List[float]] = None
    gps_accuracy: float = 5.0
    last_update: Optional[datetime] = None


@dataclass
class CameraFrame:
    """Raw camera frame data."""
    camera_id: int
    frame_data: NDArray[np.uint8]  # numpy image array
    timestamp: datetime
    frame_number: int
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.frame_data.shape
    
    @property
    def width(self) -> int:
        return self.frame_data.shape[1]
    
    @property
    def height(self) -> int:
        return self.frame_data.shape[0]


@dataclass
class PerceptionSnapshot:
    """Immutable snapshot of the world state at a point in time."""
    timestamp: datetime
    generation: int
    world_objects: List[WorldObject] = field(default_factory=list)
    camera_frames: Dict[int, CameraFrame] = field(default_factory=dict)
    detections: Dict[int, List[Detection]] = field(default_factory=dict)
    tracks: Dict[int, List[Track]] = field(default_factory=dict)
    predictions: Dict[int, List[PredictedTarget]] = field(default_factory=dict)
    
    # Performance metrics
    processing_time_ms: float = 0.0
    fps: float = 0.0


class DomainException(Exception):
    """Base exception for domain errors."""
    pass


class InvalidConfigurationError(DomainException):
    """Raised when configuration is invalid."""
    pass


class ProcessingError(DomainException):
    """Raised when frame processing fails."""
    pass
