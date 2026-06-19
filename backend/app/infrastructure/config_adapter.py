"""
Configuration adapter using Pydantic settings.
No hardcoded values - everything comes from environment or config files.
"""
from __future__ import annotations
import json
from typing import List, Optional, Any, Dict, Tuple
from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.application.ports import ConfigurationRepository


class Settings(BaseSettings):
    """Application settings - all values from environment or .env file."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Application
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Server
    host: str = Field(default="0.0.0.0", description="Server bind address")
    port: int = Field(default=8000, ge=1, le=65535, description="Server bind port")
    
    # Camera settings
    max_cameras: int = Field(default=4, ge=1, le=16, description="Maximum concurrent cameras")
    target_fps: float = Field(default=24.0, ge=1.0, le=60.0, description="Target processing framerate")
    frame_width: int = Field(default=1280, ge=320, le=4096, description="Frame width in pixels")
    frame_height: int = Field(default=720, ge=240, le=2160, description="Frame height in pixels")
    
    # Camera sources - must be provided via environment/config
    # Format: CAM_0_URL=http://..., CAM_1_URL=http://... or CAMERA_URLS=["url1", "url2"]
    camera_urls: Optional[List[str]] = Field(default=None, description="Camera URLs (RTSP/HTTP)")
    
    # Camera positions - must be provided via environment/config
    # Format: CAMERA_POSITIONS=[[x1,y1,z1],[x2,y2,z2]]
    camera_positions: Optional[List[Tuple[float, float, float]]] = Field(
        default=None, 
        description="Camera world positions as [x,y,z] tuples"
    )
    
    # Detection settings
    model_path: str = Field(default="yolov8n.pt", description="Model file path")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=100, ge=1)
    detection_classes: List[int] = Field(default=[0], description="COCO class IDs to detect")
    
    # Hardware settings
    device: str = Field(default="auto", description="Compute device: auto, cpu, cuda:0")
    half_precision: bool = Field(default=False, description="Enable FP16 inference")
    
    # Tracking settings
    tracking_max_age: int = Field(default=30, ge=1, description="Maximum track age in frames")
    tracking_min_hits: int = Field(default=3, ge=1, description="Frames needed to confirm track")
    tracking_iou_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    tracking_appearance_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    
    # WebSocket settings
    ws_max_size: int = Field(default=16777216, description="Max WebSocket message size")
    ws_ping_interval: int = Field(default=20, ge=5)
    ws_ping_timeout: int = Field(default=20, ge=5)

    # Security (additive, default-off)
    cors_origins: List[str] = Field(
        default=["*"],
        description="CORS allowlist; default ['*'] preserves dev behavior"
    )
    max_ws_clients: int = Field(
        default=100,
        description="Maximum concurrent WS viewers; new connections rejected past this"
    )
    
    # SSL settings
    ssl_enabled: bool = Field(default=True)
    ssl_certfile: str = Field(default="certs/cert.pem")
    ssl_keyfile: str = Field(default="certs/key.pem")
    
    # Mobile camera settings
    mobile_camera_fps: int = Field(default=15, ge=1, le=30)
    mobile_camera_quality: int = Field(default=50, ge=10, le=100)
    mobile_camera_max_width: int = Field(default=640, ge=320, le=1920)
    
    # JWT settings
    jwt_secret: str = Field(default="")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expire_minutes: int = Field(default=1440, ge=1)
    auth_enabled: bool = Field(default=False)
    
    # World model settings
    person_height_meters: float = Field(default=1.7, ge=0.5, le=3.0)
    prediction_horizon_seconds: float = Field(default=5.0, ge=1.0, le=30.0)
    world_object_max_age_seconds: float = Field(default=5.0, ge=1.0, le=60.0)
    
    # Homography settings
    homography_min_pairs: int = Field(default=4, ge=3)
    homography_max_pairs: int = Field(default=100, ge=10)
    homography_ransac_threshold: float = Field(default=12.0, ge=1.0)
    homography_movement_threshold: float = Field(default=1.0, ge=0.1)
    
    # GPS settings
    gps_reference_lat: Optional[float] = Field(default=None)
    gps_reference_lng: Optional[float] = Field(default=None)
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid:
            raise ValueError(f"Invalid log level: {v}. Must be one of: {valid}")
        return v_upper
    
    @field_validator("jwt_secret")
    @classmethod
    def validate_jwt_secret(cls, v: str, info) -> str:
        # In production, JWT secret must be provided
        is_debug = info.data.get("debug", False)
        auth_enabled = info.data.get("auth_enabled", False)
        
        if auth_enabled and not is_debug and not v:
            raise ValueError("JWT_SECRET must be set when auth is enabled")
        return v
    
    @field_validator("camera_positions")
    @classmethod
    def validate_camera_positions(cls, v: Optional[List[Tuple]]) -> Optional[List[Tuple]]:
        if v is None:
            return v
        for i, pos in enumerate(v):
            if len(pos) != 3:
                raise ValueError(f"Camera position {i} must have 3 coordinates [x,y,z]")
        return v


class PydanticConfigurationRepository(ConfigurationRepository):
    """Configuration repository using Pydantic settings."""
    
    def __init__(self, settings: Settings):
        self._settings = settings
        self._cache: Dict[str, Any] = {}
    
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get configuration value."""
        if key in self._cache:
            return self._cache[key]
        
        value = getattr(self._settings, key.lower(), default)
        self._cache[key] = value
        return value
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer configuration value."""
        value = self.get(key, default)
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float configuration value."""
        value = self.get(key, default)
        if isinstance(value, float):
            return value
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean configuration value."""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)
    
    def get_list(self, key: str, default: Optional[List] = None) -> List:
        """Get list configuration value."""
        value = self.get(key, default or [])
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return [item.strip() for item in value.split(",")]
        return default or []
    
    @property
    def settings(self) -> Settings:
        """Access raw settings object."""
        return self._settings


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def create_configuration_repository() -> PydanticConfigurationRepository:
    """Factory function to create configuration repository."""
    settings = get_settings()
    return PydanticConfigurationRepository(settings)
