"""
Dependency injection container.
Manages application dependencies and their lifecycle.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import logging

from app.application.ports import (
    ConfigurationRepository, CameraRepository, DetectionRepository,
    TrackingRepository, WorldModelRepository, CommunicationRepository,
    FrameEncoderRepository
)
from app.application.services import (
    PerceptionPipelineService, CameraManagementService, HealthCheckService
)


logger = logging.getLogger(__name__)


@dataclass
class ApplicationContainer:
    """Container for all application dependencies."""
    
    # Configuration
    config: ConfigurationRepository
    
    # Infrastructure
    camera_repo: CameraRepository
    detection_repo: DetectionRepository
    tracking_repo: TrackingRepository
    world_model_repo: WorldModelRepository
    communication_repo: CommunicationRepository
    frame_encoder_repo: FrameEncoderRepository
    
    # Services
    pipeline_service: PerceptionPipelineService = field(init=False)
    camera_service: CameraManagementService = field(init=False)
    health_service: HealthCheckService = field(init=False)
    
    def __post_init__(self):
        """Initialize services after dependencies are set."""
        target_fps = self.config.get_float("target_fps", 24.0)
        
        self.pipeline_service = PerceptionPipelineService(
            camera_repo=self.camera_repo,
            detection_repo=self.detection_repo,
            tracking_repo=self.tracking_repo,
            world_model_repo=self.world_model_repo,
            communication_repo=self.communication_repo,
            frame_encoder_repo=self.frame_encoder_repo,
            target_fps=target_fps
        )
        
        self.camera_service = CameraManagementService(
            camera_repo=self.camera_repo
        )
        
        self.health_service = HealthCheckService(
            camera_repo=self.camera_repo,
            detection_repo=self.detection_repo,
            tracking_repo=self.tracking_repo,
            pipeline_service=self.pipeline_service
        )
        
        logger.info("ApplicationContainer initialized")
    
    async def start(self) -> None:
        """Start all services."""
        logger.info("Starting application services...")
        
        # Start infrastructure
        await self.camera_repo.start()
        await self.detection_repo.initialize()
        await self.tracking_repo.initialize()
        await self.world_model_repo.initialize()
        
        # Start pipeline
        await self.pipeline_service.start()
        
        logger.info("All services started")
    
    async def stop(self) -> None:
        """Stop all services."""
        logger.info("Stopping application services...")
        
        # Stop pipeline first
        await self.pipeline_service.stop()
        
        # Stop infrastructure
        await self.camera_repo.stop()
        
        logger.info("All services stopped")


def create_container() -> ApplicationContainer:
    """Factory function to create and configure the application container."""
    from app.infrastructure.config_adapter import create_configuration_repository
    from app.infrastructure.camera_adapter import OpenCVCameraRepository
    from app.infrastructure.detection_adapter import DetectionRepositoryImpl
    from app.infrastructure.tracking_adapter import TrackingRepositoryImpl
    from app.infrastructure.world_model_adapter import WorldModelRepositoryImpl
    from app.infrastructure.websocket_adapter import WebSocketCommunicationRepository
    from app.infrastructure.frame_encoder_adapter import OpenCVFrameEncoder
    
    # Create configuration repository
    config_repo = create_configuration_repository()
    
    # Get camera URLs from config
    camera_urls = config_repo.get_list("camera_urls", [])
    
    # Create infrastructure repositories
    camera_repo = OpenCVCameraRepository(
        config_repo=config_repo,
        camera_urls=camera_urls
    )
    
    detection_repo = DetectionRepositoryImpl(config_repo)
    tracking_repo = TrackingRepositoryImpl(config_repo)
    world_model_repo = WorldModelRepositoryImpl(config_repo)
    communication_repo = WebSocketCommunicationRepository()
    frame_encoder_repo = OpenCVFrameEncoder()
    
    # Create container
    container = ApplicationContainer(
        config=config_repo,
        camera_repo=camera_repo,
        detection_repo=detection_repo,
        tracking_repo=tracking_repo,
        world_model_repo=world_model_repo,
        communication_repo=communication_repo,
        frame_encoder_repo=frame_encoder_repo
    )
    
    return container
