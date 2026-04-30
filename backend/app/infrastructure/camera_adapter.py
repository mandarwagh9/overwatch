"""
Camera repository adapter using OpenCV.
No hardcoded camera URLs - all from configuration.
"""
from __future__ import annotations
import asyncio
import os
import threading
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from dataclasses import dataclass, field

import numpy as np
import cv2

from app.application.ports import CameraRepository, ConfigurationRepository
from app.domain.entities import CameraFrame


logger = logging.getLogger(__name__)


@dataclass
class FrameBuffer:
    """Thread-safe frame buffer for a camera."""
    max_size: int = 2
    queue: Queue = field(default_factory=lambda: Queue(maxsize=2))
    last_frame_time: float = field(default_factory=time.time)
    frame_counter: int = 0
    
    def put(self, frame: CameraFrame) -> bool:
        """Try to put a frame in the buffer (drops old frames if full)."""
        try:
            # Drop oldest frame if buffer is full
            while self.queue.qsize() >= self.max_size:
                try:
                    self.queue.get_nowait()
                except Empty:
                    break
            
            self.queue.put_nowait(frame)
            self.last_frame_time = time.time()
            self.frame_counter += 1
            return True
        except Exception:
            return False
    
    def get(self) -> Optional[CameraFrame]:
        """Get the latest frame from the buffer."""
        try:
            return self.queue.get_nowait()
        except Empty:
            return None


class CameraCapture:
    """Individual camera capture handler."""
    
    def __init__(
        self,
        camera_id: int,
        camera_url: str,
        target_resolution: Tuple[int, int],
        target_fps: float
    ):
        self.camera_id = camera_id
        self.camera_url = camera_url
        self.target_width, self.target_height = target_resolution
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._buffer = FrameBuffer()
        self._is_running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Statistics
        self._frames_captured = 0
        self._frames_dropped = 0
        self._current_fps = 0.0
        self._last_fps_update = time.time()
        
        logger.debug(f"CameraCapture {camera_id} initialized with URL: {camera_url}")
    
    def start(self) -> bool:
        """Start camera capture."""
        try:
            # Try FFMPEG backend first for IP cameras
            self._cap = cv2.VideoCapture(self.camera_url, cv2.CAP_FFMPEG)
            
            if not self._cap.isOpened():
                logger.warning(f"FFMPEG failed for camera {self.camera_id}, trying default backend")
                self._cap = cv2.VideoCapture(self.camera_url)
            
            if not self._cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}: {self.camera_url}")
                return False
            
            # Test read
            ret, test_frame = self._cap.read()
            if not ret or test_frame is None:
                logger.error(f"Camera {self.camera_id} opened but cannot read frames")
                self._cap.release()
                return False
            
            # Configure camera
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            
            # Try MJPEG codec
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            
            self._is_running = True
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
            
            logger.info(f"Camera {self.camera_id} started: {self.camera_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start camera {self.camera_id}: {e}")
            if self._cap:
                self._cap.release()
            return False
    
    def stop(self) -> None:
        """Stop camera capture."""
        self._is_running = False
        
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        
        if self._cap:
            self._cap.release()
            self._cap = None
        
        logger.info(f"Camera {self.camera_id} stopped")
    
    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread.

        Reconnects ``cv2.VideoCapture`` with exponential backoff after
        persistent read failures (e.g. RTSP stream drops).
        """
        frame_times: List[float] = []
        consecutive_failures = 0
        reconnect_threshold = 30  # ~3s at 100ms sleep
        backoff = 1.0
        max_backoff = 30.0

        while self._is_running:
            try:
                if self._cap is None or not self._cap.isOpened():
                    self._reconnect(backoff)
                    backoff = min(backoff * 2, max_backoff)
                    consecutive_failures = 0
                    continue

                loop_start = time.time()

                ret, frame = self._cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= reconnect_threshold:
                        logger.warning(
                            f"Camera {self.camera_id}: {consecutive_failures} consecutive "
                            f"read failures, reconnecting"
                        )
                        try:
                            self._cap.release()
                        except Exception:
                            pass
                        self._cap = None
                        consecutive_failures = 0
                    else:
                        time.sleep(0.1)
                    continue

                consecutive_failures = 0
                backoff = 1.0

                # Resize if needed
                if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
                    frame = cv2.resize(frame, (self.target_width, self.target_height))

                # Create CameraFrame
                camera_frame = CameraFrame(
                    camera_id=self.camera_id,
                    frame_data=frame,
                    timestamp=datetime.now(),
                    frame_number=self._buffer.frame_counter
                )

                # Try to buffer the frame
                if self._buffer.put(camera_frame):
                    self._frames_captured += 1
                else:
                    self._frames_dropped += 1

                # Calculate FPS
                frame_times.append(time.time())
                if len(frame_times) > 30:
                    frame_times.pop(0)
                if len(frame_times) > 1:
                    self._current_fps = len(frame_times) / (frame_times[-1] - frame_times[0])

                # Maintain target FPS
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Camera {self.camera_id} capture error: {e}")
                time.sleep(0.1)

    def _reconnect(self, delay_s: float) -> None:
        """Open VideoCapture with backoff."""
        time.sleep(delay_s)
        if not self._is_running:
            return
        try:
            cap = cv2.VideoCapture(self.camera_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.camera_url)
            self._cap = cap
            if self._cap.isOpened():
                # Reapply capture configuration on reconnect
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                logger.info(
                    f"Camera {self.camera_id}: reconnected to {self.camera_url}"
                )
            else:
                logger.warning(
                    f"Camera {self.camera_id}: reconnect attempt failed"
                )
        except Exception as e:
            logger.error(f"Camera {self.camera_id}: reconnect error: {e}")
    
    def get_latest_frame(self) -> Optional[CameraFrame]:
        """Get the latest frame from the buffer."""
        return self._buffer.get()
    
    def get_stats(self) -> Dict:
        """Get camera statistics."""
        return {
            "camera_id": self.camera_id,
            "is_running": self._is_running,
            "fps": round(self._current_fps, 1),
            "frames_captured": self._frames_captured,
            "frames_dropped": self._frames_dropped,
            "drop_rate": (
                self._frames_dropped / max(1, self._frames_captured + self._frames_dropped) * 100
            )
        }


class VirtualCamera:
    """Push-based virtual camera for mobile streams."""
    
    def __init__(self, camera_id: int, max_width: int):
        self.camera_id = camera_id
        self.max_width = max_width
        self._buffer = FrameBuffer()
        self._is_running = True
        self._last_jpeg: Optional[bytes] = None
        
        # GPS/IMU data
        self.gps_position: Optional[Tuple[float, float, float]] = None
        self.heading: float = 0.0
        self.gps_accuracy: float = 999.0
        
        # Statistics
        self._frames_captured = 0
        self._frames_dropped = 0
        self._current_fps = 0.0
    
    def inject_frame(self, jpeg_bytes: bytes) -> bool:
        """Inject a JPEG frame from mobile client."""
        try:
            # Decode JPEG
            np_arr = np.frombuffer(jpeg_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                self._frames_dropped += 1
                return False
            
            # Resize if too large
            if frame.shape[1] > self.max_width:
                scale = self.max_width / frame.shape[1]
                new_w = int(frame.shape[1] * scale)
                new_h = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (new_w, new_h))
            
            # Store JPEG for passthrough
            self._last_jpeg = jpeg_bytes
            
            # Create and buffer CameraFrame
            camera_frame = CameraFrame(
                camera_id=self.camera_id,
                frame_data=frame,
                timestamp=datetime.now(),
                frame_number=self._buffer.frame_counter
            )
            
            if self._buffer.put(camera_frame):
                self._frames_captured += 1
                return True
            else:
                self._frames_dropped += 1
                return False
                
        except Exception as e:
            logger.error(f"VirtualCamera {self.camera_id} inject error: {e}")
            self._frames_dropped += 1
            return False
    
    def get_latest_frame(self) -> Optional[CameraFrame]:
        """Get latest decoded frame."""
        return self._buffer.get()
    
    def get_last_jpeg(self) -> Optional[bytes]:
        """Get last raw JPEG for passthrough."""
        return self._last_jpeg
    
    def stop(self) -> None:
        """Stop virtual camera."""
        self._is_running = False
        self._last_jpeg = None
    
    def get_stats(self) -> Dict:
        """Get virtual camera statistics."""
        return {
            "camera_id": self.camera_id,
            "is_running": self._is_running,
            "is_virtual": True,
            "fps": round(self._current_fps, 1),
            "frames_captured": self._frames_captured,
            "frames_dropped": self._frames_dropped
        }


class OpenCVCameraRepository(CameraRepository):
    """Camera repository using OpenCV."""
    
    def __init__(
        self,
        config_repo: ConfigurationRepository,
        camera_urls: Optional[List[str]] = None
    ):
        self._config = config_repo
        self._camera_urls = camera_urls or []
        self._cameras: Dict[int, CameraCapture] = {}
        self._virtual_cameras: Dict[int, VirtualCamera] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._lock = asyncio.Lock()
        self._virtual_camera_lock = threading.Lock()
        
        self._target_resolution = (
            self._config.get_int("frame_width", 1280),
            self._config.get_int("frame_height", 720)
        )
        self._target_fps = self._config.get_float("target_fps", 24.0)
        self._max_cameras = self._config.get_int("max_cameras", 4)
        
        logger.info(f"OpenCVCameraRepository initialized with {len(self._camera_urls)} URLs")
    
    async def start(self) -> None:
        """Start the camera repository."""
        logger.info("Camera repository started")
    
    async def stop(self) -> None:
        """Stop all cameras and cleanup."""
        async with self._lock:
            # Stop physical cameras
            for camera in list(self._cameras.values()):
                camera.stop()
            self._cameras.clear()
            
            # Stop virtual cameras
            for vcam in list(self._virtual_cameras.values()):
                vcam.stop()
            self._virtual_cameras.clear()
        
        self._executor.shutdown(wait=True)
        logger.info("Camera repository stopped")
    
    async def start_camera(self, camera_id: int) -> bool:
        """Start a specific camera."""
        async with self._lock:
            if camera_id in self._cameras or camera_id in self._virtual_cameras:
                logger.warning(f"Camera {camera_id} is already running")
                return False
            
            if len(self._cameras) + len(self._virtual_cameras) >= self._max_cameras:
                logger.error(f"Maximum camera limit reached ({self._max_cameras})")
                return False
            
            # Get camera URL from config
            url = self._get_camera_url(camera_id)
            if not url:
                logger.error(f"No URL configured for camera {camera_id}")
                return False
            
            # Create and start camera
            camera = CameraCapture(
                camera_id=camera_id,
                camera_url=url,
                target_resolution=self._target_resolution,
                target_fps=self._target_fps
            )
            
            # Run in thread pool
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(self._executor, camera.start)
            
            if success:
                self._cameras[camera_id] = camera
                return True
            
            return False
    
    async def stop_camera(self, camera_id: int) -> bool:
        """Stop a specific camera."""
        async with self._lock:
            if camera_id in self._virtual_cameras:
                return self.unregister_virtual_camera(camera_id)
            
            if camera_id not in self._cameras:
                return False
            
            camera = self._cameras.pop(camera_id)
            camera.stop()
            return True
    
    def get_latest_frames(self) -> List[CameraFrame]:
        """Get latest frames from all active cameras."""
        frames = []
        
        # Physical cameras
        for camera in self._cameras.values():
            frame = camera.get_latest_frame()
            if frame:
                frames.append(frame)
        
        # Virtual cameras
        for cam_id, vcam in self._virtual_cameras.items():
            frame = vcam.get_latest_frame()
            if frame:
                logger.debug(f"✅ Retrieved frame from virtual camera {cam_id}")
                frames.append(frame)
            else:
                logger.warning(f"⚠️ No frame available from virtual camera {cam_id}, buffer may be empty")
        
        if frames:
            logger.debug(f"📹 Total frames retrieved for processing: {len(frames)}")
        
        return frames
    
    def get_camera_count(self) -> int:
        """Get number of active cameras."""
        return len(self._cameras) + len(self._virtual_cameras)
    
    def register_virtual_camera(self, camera_id: Optional[int] = None) -> Optional[int]:
        """Register a virtual camera."""
        with self._virtual_camera_lock:
            if len(self._cameras) + len(self._virtual_cameras) >= self._max_cameras:
                logger.error(f"Maximum camera limit reached")
                return None

            # Find available slot
            if camera_id is None:
                for i in range(self._max_cameras):
                    if i not in self._cameras and i not in self._virtual_cameras:
                        camera_id = i
                        break

            if camera_id is None:
                return None

            if camera_id in self._cameras or camera_id in self._virtual_cameras:
                logger.warning(f"Camera slot {camera_id} already in use")
                return None

            max_width = self._config.get_int("mobile_camera_max_width", 640)
            vcam = VirtualCamera(camera_id, max_width)
            self._virtual_cameras[camera_id] = vcam

            logger.info(f"Virtual camera {camera_id} registered")
            return camera_id

    def unregister_virtual_camera(self, camera_id: int) -> bool:
        """Unregister a virtual camera."""
        with self._virtual_camera_lock:
            if camera_id not in self._virtual_cameras:
                return False

            vcam = self._virtual_cameras.pop(camera_id)
        vcam.stop()
        logger.info(f"Virtual camera {camera_id} unregistered")
        return True

    def inject_frame(self, camera_id: int, jpeg_bytes: bytes) -> bool:
        """Inject a JPEG frame into a virtual camera."""
        with self._virtual_camera_lock:
            vcam = self._virtual_cameras.get(camera_id)
        if vcam is None:
            return False
        return vcam.inject_frame(jpeg_bytes)
    
    def _get_camera_url(self, camera_id: int) -> Optional[str]:
        """Get camera URL from configuration."""
        # Try CAMERA_URLS list first
        urls = self._config.get_list("camera_urls", [])
        if urls and camera_id < len(urls):
            return urls[camera_id]
        
        # Try CAM_{ID}_URL environment variable
        env_url = os.environ.get(f"CAM_{camera_id}_URL")
        if env_url:
            return env_url
        
        # Fallback to device index for local cameras
        if camera_id < 4:  # Allow first 4 as local cameras
            return str(camera_id)
        
        return None

