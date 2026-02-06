"""
Configuration module for Overwatch system
"""

import os
from typing import Optional

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    model_config = {
        "protected_namespaces": (),
        "env_file": ".env",
        "case_sensitive": False,
    }
    
    # Server configuration
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Camera settings
    max_cameras: int = 4
    target_fps: int = 24
    frame_width: int = 1280
    frame_height: int = 720
    
    # CV Processing settings
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100
    detection_classes: list = [0]  # COCO class IDs to detect (0 = person). Empty list = all classes
    
    # Device / Jetson Orin Nano settings
    device: str = "auto"       # "auto", "cpu", "cuda", "cuda:0"
    half_precision: bool = False  # FP16 — set True on Jetson with TensorRT .engine model
    
    # Tracking settings
    tracking_max_age: int = 30
    tracking_n_init: int = 3
    tracking_max_iou_distance: float = 0.7
    
    # WebSocket settings
    ws_max_size: int = 16777216  # 16MB
    ws_ping_interval: int = 20
    ws_ping_timeout: int = 20
    
    # SSL settings (required for mobile camera getUserMedia over LAN)
    ssl_enabled: bool = True
    ssl_certfile: str = "certs/cert.pem"
    ssl_keyfile: str = "certs/key.pem"
    
    # Mobile camera settings
    mobile_camera_fps: int = 15  # Target FPS for mobile camera streams
    mobile_camera_quality: int = 50  # JPEG quality for mobile re-encoding
    mobile_camera_max_width: int = 640  # Max frame width from mobile


# Global settings instance
settings = Settings()