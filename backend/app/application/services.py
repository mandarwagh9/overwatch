"""
Application services (use cases) that orchestrate domain logic.
"""
from __future__ import annotations
import asyncio
import time
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass
import logging

from app.domain.entities import (
    PerceptionSnapshot, CameraFrame, Detection, Track, 
    WorldObject, PredictedTarget
)
from app.application.ports import (
    CameraRepository, DetectionRepository, TrackingRepository,
    WorldModelRepository, CommunicationRepository, FrameEncoderRepository
)


logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Performance metrics for the perception pipeline."""
    frames_processed: int = 0
    total_processing_time_ms: float = 0.0
    average_processing_time_ms: float = 0.0
    current_fps: float = 0.0
    dropped_frames: int = 0
    _last_update_time: float = 0.0
    
    def update(self, processing_time_ms: float):
        """Update metrics with new timing data."""
        import time
        current_time = time.perf_counter()
        
        self.frames_processed += 1
        self.total_processing_time_ms += processing_time_ms
        self.average_processing_time_ms = (
            self.total_processing_time_ms / self.frames_processed
        )
        
        # Calculate FPS based on time since last frame
        if self._last_update_time > 0:
            elapsed = current_time - self._last_update_time
            if elapsed > 0:
                self.current_fps = 1.0 / elapsed
        
        self._last_update_time = current_time


class PerceptionPipelineService:
    """
    Singleton perception pipeline that runs detection → tracking → fusion
    ONCE per tick and broadcasts snapshots to all viewers.
    """
    
    def __init__(
        self,
        camera_repo: CameraRepository,
        detection_repo: DetectionRepository,
        tracking_repo: TrackingRepository,
        world_model_repo: WorldModelRepository,
        communication_repo: CommunicationRepository,
        frame_encoder_repo: FrameEncoderRepository,
        target_fps: float = 24.0
    ):
        self._camera_repo = camera_repo
        self._detection_repo = detection_repo
        self._tracking_repo = tracking_repo
        self._world_model_repo = world_model_repo
        self._communication_repo = communication_repo
        self._frame_encoder_repo = frame_encoder_repo
        self._target_fps = target_fps
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._latest_snapshot: Optional[PerceptionSnapshot] = None
        self._generation = 0
        self._metrics = PipelineMetrics()
        self._lock = asyncio.Lock()
        
        logger.info("PerceptionPipelineService initialized")
    
    async def start(self) -> None:
        """Start the perception pipeline."""
        if self._running:
            logger.warning("Pipeline already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Perception pipeline started")
    
    async def stop(self) -> None:
        """Stop the perception pipeline."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Perception pipeline stopped")
    
    @property
    def latest_snapshot(self) -> Optional[PerceptionSnapshot]:
        """Get the latest perception snapshot."""
        return self._latest_snapshot
    
    @property
    def metrics(self) -> PipelineMetrics:
        """Get current pipeline metrics."""
        return self._metrics
    
    async def _run_loop(self) -> None:
        """Main processing loop."""
        interval = 1.0 / self._target_fps
        
        while self._running:
            try:
                start_time = time.perf_counter()
                await self._process_tick()
                processing_time = (time.perf_counter() - start_time) * 1000
                self._metrics.update(processing_time)
                
                # Calculate sleep time to maintain target FPS
                elapsed = time.perf_counter() - start_time
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
            except Exception as e:
                logger.error(f"Pipeline tick error: {e}", exc_info=True)
                await asyncio.sleep(interval)
    
    async def _process_tick(self) -> None:
        """Process one tick of the pipeline."""
        tick_start = datetime.now()
        
        # 1. Capture frames from all cameras
        frames = self._camera_repo.get_latest_frames()
        
        if not frames:
            return
        
        # 2. Run detection on all frames
        detections_map = await self._detection_repo.detect_batch(frames)
        
        # Log detection summary
        total_dets = sum(len(dets) for dets in detections_map.values())
        logger.info(f"🔄 Pipeline tick: {len(frames)} frames, {total_dets} detections")
        
        # 3. Run tracking for each camera
        tracks_map: Dict[int, List[Track]] = {}
        for frame in frames:
            camera_id = frame.camera_id
            detections = detections_map.get(camera_id, [])
            tracks = await self._tracking_repo.update(camera_id, detections, frame)
            tracks_map[camera_id] = tracks
        
        total_tracks = sum(len(trks) for trks in tracks_map.values())
        logger.info(f"📍 Tracks: {total_tracks}")
        
        # 4. Update world model with all tracks
        world_objects = await self._world_model_repo.update(tracks_map)
        
        # 5. Generate predictions for all cameras
        predictions_map: Dict[int, List[PredictedTarget]] = {}
        all_camera_ids = set(frame.camera_id for frame in frames)
        for camera_id in all_camera_ids:
            predictions = self._world_model_repo.generate_predictions(camera_id)
            predictions_map[camera_id] = predictions
        
        # 6. Create snapshot
        self._generation += 1
        snapshot = PerceptionSnapshot(
            timestamp=tick_start,
            generation=self._generation,
            world_objects=world_objects,
            camera_frames={f.camera_id: f for f in frames},
            detections=detections_map,
            tracks=tracks_map,
            predictions=predictions_map
        )
        
        async with self._lock:
            self._latest_snapshot = snapshot
        
        # 7. Broadcast to clients
        await self._communication_repo.broadcast_snapshot(snapshot)


class CameraManagementService:
    """Service for managing camera operations."""
    
    def __init__(self, camera_repo: CameraRepository):
        self._camera_repo = camera_repo
    
    async def start_camera(self, camera_id: int) -> bool:
        """Start a specific camera."""
        return await self._camera_repo.start_camera(camera_id)
    
    async def stop_camera(self, camera_id: int) -> bool:
        """Stop a specific camera."""
        return await self._camera_repo.stop_camera(camera_id)
    
    def get_active_camera_count(self) -> int:
        """Get number of active cameras."""
        return self._camera_repo.get_camera_count()
    
    def register_virtual_camera(self, camera_id: Optional[int] = None) -> Optional[int]:
        """Register a virtual camera for mobile streams."""
        return self._camera_repo.register_virtual_camera(camera_id)
    
    def unregister_virtual_camera(self, camera_id: int) -> bool:
        """Unregister a virtual camera."""
        return self._camera_repo.unregister_virtual_camera(camera_id)
    
    def inject_mobile_frame(self, camera_id: int, jpeg_bytes: bytes) -> bool:
        """Inject a frame from a mobile camera."""
        return self._camera_repo.inject_frame(camera_id, jpeg_bytes)


class HealthCheckService:
    """Service for system health monitoring."""
    
    def __init__(
        self,
        camera_repo: CameraRepository,
        detection_repo: DetectionRepository,
        tracking_repo: TrackingRepository,
        pipeline_service: PerceptionPipelineService
    ):
        self._camera_repo = camera_repo
        self._detection_repo = detection_repo
        self._tracking_repo = tracking_repo
        self._pipeline_service = pipeline_service
    
    def get_status(self) -> Dict:
        """Get comprehensive system status."""
        snapshot = self._pipeline_service.latest_snapshot
        metrics = self._pipeline_service.metrics
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "cameras": {
                "active": self._camera_repo.get_camera_count(),
                "status": "active" if self._camera_repo.get_camera_count() > 0 else "idle"
            },
            "detection": {
                "ready": self._detection_repo.is_ready(),
                "status": "ready" if self._detection_repo.is_ready() else "initializing"
            },
            "pipeline": {
                "frames_processed": metrics.frames_processed,
                "average_processing_time_ms": round(metrics.average_processing_time_ms, 2),
                "current_fps": round(metrics.current_fps, 1),
                "world_objects": len(snapshot.world_objects) if snapshot else 0
            }
        }
