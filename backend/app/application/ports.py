"""
Repository interfaces (ports) for the application layer.
These define the contracts that infrastructure adapters must implement.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict, Tuple
from datetime import datetime
import numpy as np
from numpy.typing import NDArray

from app.domain.entities import (
    CameraFrame, Detection, Track, WorldObject, PredictedTarget,
    CameraCalibration, BoundingBox, Point3D, AppearanceDescriptor
)


class CameraRepository(ABC):
    """Repository for camera frame acquisition."""
    
    @abstractmethod
    async def start(self) -> None:
        """Start the camera repository."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the camera repository and release resources."""
        pass
    
    @abstractmethod
    async def start_camera(self, camera_id: int) -> bool:
        """Start a specific camera."""
        pass
    
    @abstractmethod
    async def stop_camera(self, camera_id: int) -> bool:
        """Stop a specific camera."""
        pass
    
    @abstractmethod
    def get_latest_frames(self) -> List[CameraFrame]:
        """Get latest frames from all active cameras."""
        pass
    
    @abstractmethod
    def get_camera_count(self) -> int:
        """Get number of active cameras."""
        pass
    
    @abstractmethod
    def register_virtual_camera(self, camera_id: Optional[int] = None) -> Optional[int]:
        """Register a virtual camera for mobile streams."""
        pass
    
    @abstractmethod
    def unregister_virtual_camera(self, camera_id: int) -> bool:
        """Unregister a virtual camera."""
        pass
    
    @abstractmethod
    def inject_frame(self, camera_id: int, jpeg_bytes: bytes) -> bool:
        """Inject a JPEG frame into a virtual camera."""
        pass


class DetectionRepository(ABC):
    """Repository for object detection."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the detection model."""
        pass
    
    @abstractmethod
    async def detect(self, frame: CameraFrame) -> List[Detection]:
        """Run detection on a single frame."""
        pass
    
    @abstractmethod
    async def detect_batch(self, frames: List[CameraFrame]) -> Dict[int, List[Detection]]:
        """Run detection on multiple frames."""
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if detector is ready."""
        pass
    
    @abstractmethod
    async def compute_appearance(self, frame: NDArray[np.uint8], 
                                 bbox: BoundingBox) -> Optional[AppearanceDescriptor]:
        """Compute appearance descriptor for a detection."""
        pass


class TrackingRepository(ABC):
    """Repository for multi-object tracking."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the tracker."""
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if tracking is ready."""
        pass
    
    @abstractmethod
    async def update(self, camera_id: int, 
                     detections: List[Detection],
                     frame: Optional[CameraFrame] = None) -> List[Track]:
        """Update tracks with new detections."""
        pass
    
    @abstractmethod
    def get_tracks(self, camera_id: int) -> List[Track]:
        """Get current tracks for a camera."""
        pass
    
    @abstractmethod
    def get_all_tracks(self) -> Dict[int, List[Track]]:
        """Get all tracks organized by camera."""
        pass


class WorldModelRepository(ABC):
    """Repository for world model and sensor fusion."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the world model."""
        pass
    
    @abstractmethod
    async def update(self, tracks: Dict[int, List[Track]]) -> List[WorldObject]:
        """Update world model with new tracking results."""
        pass
    
    @abstractmethod
    def get_world_objects(self) -> List[WorldObject]:
        """Get all current world objects."""
        pass
    
    @abstractmethod
    def generate_predictions(self, camera_id: int) -> List[PredictedTarget]:
        """Generate predictions for a camera view."""
        pass
    
    @abstractmethod
    def update_camera_calibration(self, calibration: CameraCalibration) -> None:
        """Update calibration for a camera."""
        pass
    
    @abstractmethod
    def get_camera_calibration(self, camera_id: int) -> Optional[CameraCalibration]:
        """Get calibration for a camera."""
        pass


class FrameEncoderRepository(ABC):
    """Repository for frame encoding."""
    
    @abstractmethod
    def encode(self, frame: NDArray[np.uint8], 
               quality: int = 85) -> Optional[bytes]:
        """Encode frame to JPEG bytes."""
        pass
    
    @abstractmethod
    def decode(self, jpeg_bytes: bytes) -> Optional[NDArray[np.uint8]]:
        """Decode JPEG bytes to frame."""
        pass


class CommunicationRepository(ABC):
    """Repository for client communication."""
    
    @abstractmethod
    async def broadcast_snapshot(self, snapshot: 'PerceptionSnapshot') -> None:
        """Broadcast a perception snapshot to all connected clients."""
        pass
    
    @abstractmethod
    def get_client_count(self) -> int:
        """Get number of connected clients."""
        pass


class ConfigurationRepository(ABC):
    """Repository for configuration management."""
    
    @abstractmethod
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get configuration value."""
        pass
    
    @abstractmethod
    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer configuration value."""
        pass
    
    @abstractmethod
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float configuration value."""
        pass
    
    @abstractmethod
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean configuration value."""
        pass
    
    @abstractmethod
    def get_list(self, key: str, default: Optional[List] = None) -> List:
        """Get list configuration value."""
        pass
