"""
Camera Manager for handling multiple camera streams with 24 FPS optimization
"""

import asyncio
import threading
import time
import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from dataclasses import dataclass

from app.config import settings


@dataclass
class CameraFrame:
    """Container for camera frame data"""
    camera_id: int
    frame: np.ndarray
    timestamp: float
    frame_number: int


class FrameQueue:
    """Thread-safe frame queue with FPS limiting"""
    
    def __init__(self, max_size: int = 3):
        # Increase buffer to smooth bursts from IP cameras
        self.queue = Queue(maxsize=max_size)
        self.fps_target = settings.target_fps
        self.frame_interval = 1.0 / self.fps_target
        self.last_frame_time = 0
        self.frame_counter = 0
        
    def put_frame(self, frame: np.ndarray, camera_id: int) -> bool:
        """Add frame to queue with FPS limiting"""
        current_time = time.time()
        
        # FPS limiting - skip frames if we're going too fast
        if current_time - self.last_frame_time < self.frame_interval:
            return False
        
        try:
            # Try to put frame without blocking
            camera_frame = CameraFrame(
                camera_id=camera_id,
                frame=frame,
                timestamp=current_time,
                frame_number=self.frame_counter
            )
            
            # Remove old frame if queue is full
            try:
                self.queue.get_nowait()
            except Empty:
                pass
                
            self.queue.put_nowait(camera_frame)
            self.last_frame_time = current_time
            self.frame_counter += 1
            return True
            
        except Exception:
            return False
    
    def get_frame(self) -> Optional[CameraFrame]:
        """Get latest frame from queue"""
        try:
            return self.queue.get_nowait()
        except Empty:
            return None


class CameraCapture:
    """Individual camera capture handler"""
    
    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self.cap: Optional[cv2.VideoCapture] = None
        self.target_width = settings.frame_width
        self.target_height = settings.frame_height
        self.frame_queue = FrameQueue()
        self.is_running = False
        self.capture_thread: Optional[threading.Thread] = None
        self.stats = {
            'frames_captured': 0,
            'frames_dropped': 0,
            'fps': 0,
            'last_fps_update': time.time()
        }
        
    def start(self) -> bool:
        """Start camera capture"""
        try:
            # Define IP camera URLs for mobile devices
            # Replace these IPs with your actual mobile device IPs
            ip_cameras = {
                0: "http://192.168.1.4:8080/video",  # Mobile 1 - Video stream URL
                1: "http://192.168.1.101:8080/video",  # Mobile 2 - Replace with your IP
                2: "http://192.168.1.102:8080/video",  # Mobile 3 - Replace with your IP  
                3: "http://192.168.1.103:8080/video"   # Mobile 4 - Replace with your IP
            }
            
            # Try to open IP camera first, then fallback to local camera
            if self.camera_id in ip_cameras:
                ip_url = ip_cameras[self.camera_id]
                print(f"🔄 Attempting to connect to IP camera: {ip_url}")
                
                # Try different backends for IP camera
                self.cap = cv2.VideoCapture(ip_url, cv2.CAP_FFMPEG)
                
                if not self.cap.isOpened():
                    print(f"⚠️ FFMPEG backend failed, trying default backend...")
                    self.cap = cv2.VideoCapture(ip_url)
                
                if self.cap.isOpened():
                    # Test if we can actually read a frame
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        print(f"✅ IP camera connected successfully: {ip_url}")
                    else:
                        print(f"⚠️ IP camera opened but can't read frames, trying local camera {self.camera_id}")
                        self.cap.release()
                        self.cap = cv2.VideoCapture(self.camera_id)
                else:
                    print(f"⚠️ IP camera failed to open, trying local camera {self.camera_id}")
                    self.cap = cv2.VideoCapture(self.camera_id)
            else:
                print(f"🔄 Using local camera {self.camera_id}")
                self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                # Fallback for development - use default camera
                if self.camera_id != 0:
                    print(f"Camera {self.camera_id} not found, trying default camera...")
                    self.cap = cv2.VideoCapture(0)
                
                if not self.cap.isOpened():
                    print(f"Failed to open camera {self.camera_id}")
                    return False
            
            # Optimize camera settings
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            self.cap.set(cv2.CAP_PROP_FPS, settings.target_fps)

            # Lower capture resolution to reduce network/encode load if needed
            # Target a max width of 640px by default for IP cameras
            max_width = 640
            tw = min(settings.frame_width, max_width)
            th = int(tw * settings.frame_height / max(1, settings.frame_width))
            self.target_width = tw
            self.target_height = th
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            
            # Try to use MJPEG codec for better performance
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            print(f"✅ Camera {self.camera_id} started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start camera {self.camera_id}: {e}")
            return False
    
    def stop(self):
        """Stop camera capture"""
        self.is_running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        print(f"🛑 Camera {self.camera_id} stopped")
    
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        fps_counter = 0
        fps_start_time = time.time()
        
        while self.is_running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    print(f"⚠️ Camera {self.camera_id}: Failed to read frame")
                    continue
                
                # Resize frame if needed (optimization)
                if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
                    frame = cv2.resize(frame, (self.target_width, self.target_height))
                
                # Try to add frame to queue
                if self.frame_queue.put_frame(frame, self.camera_id):
                    self.stats['frames_captured'] += 1
                    fps_counter += 1
                else:
                    self.stats['frames_dropped'] += 1
                
                # Update FPS statistics every second
                current_time = time.time()
                if current_time - fps_start_time >= 1.0:
                    self.stats['fps'] = fps_counter / (current_time - fps_start_time)
                    self.stats['last_fps_update'] = current_time
                    fps_counter = 0
                    fps_start_time = current_time
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.001)  # 1ms
                
            except Exception as e:
                print(f"❌ Camera {self.camera_id} capture error: {e}")
                break
    
    def get_latest_frame(self) -> Optional[CameraFrame]:
        """Get the latest frame from this camera"""
        return self.frame_queue.get_frame()
    
    def get_stats(self) -> dict:
        """Get camera statistics"""
        return {
            'camera_id': self.camera_id,
            'is_running': self.is_running,
            'fps': round(self.stats['fps'], 1),
            'frames_captured': self.stats['frames_captured'],
            'frames_dropped': self.stats['frames_dropped'],
            'drop_rate': (
                self.stats['frames_dropped'] / max(1, self.stats['frames_captured'] + self.stats['frames_dropped'])
            ) * 100
        }


class VirtualCamera:
    """Push-based virtual camera for mobile browser streams.
    
    Instead of pulling frames from OpenCV, frames are pushed in
    via inject_frame() from a WebSocket connection.
    """
    
    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self.frame_queue = FrameQueue(max_size=3)
        self.is_running = True
        self.is_mobile = True  # Flag to identify mobile cameras
        self.stats = {
            'frames_captured': 0,
            'frames_dropped': 0,
            'fps': 0,
            'last_fps_update': time.time()
        }
        self._fps_counter = 0
        self._fps_start_time = time.time()
        # Keep raw JPEG bytes for pass-through to viewers (skip re-encoding)
        self._last_jpeg_bytes: Optional[bytes] = None
    
    def inject_frame(self, jpeg_bytes: bytes) -> bool:
        """Inject a JPEG frame received from the mobile client.
        
        Decodes JPEG to numpy array for the detection pipeline,
        and stores the original JPEG for pass-through to viewers.
        """
        try:
            # Decode JPEG to numpy array for detection/tracking pipeline
            np_arr = np.frombuffer(jpeg_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                self.stats['frames_dropped'] += 1
                return False
            
            # Resize if too large
            max_width = settings.mobile_camera_max_width
            if frame.shape[1] > max_width:
                scale = max_width / frame.shape[1]
                new_w = int(frame.shape[1] * scale)
                new_h = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (new_w, new_h))
            
            # Store original JPEG for pass-through
            self._last_jpeg_bytes = jpeg_bytes
            
            # Push decoded frame into the queue
            if self.frame_queue.put_frame(frame, self.camera_id):
                self.stats['frames_captured'] += 1
                self._fps_counter += 1
            else:
                self.stats['frames_dropped'] += 1
            
            # Update FPS stats
            current_time = time.time()
            elapsed = current_time - self._fps_start_time
            if elapsed >= 1.0:
                self.stats['fps'] = self._fps_counter / elapsed
                self.stats['last_fps_update'] = current_time
                self._fps_counter = 0
                self._fps_start_time = current_time
            
            return True
            
        except Exception as e:
            print(f"❌ VirtualCamera {self.camera_id} inject error: {e}")
            self.stats['frames_dropped'] += 1
            return False
    
    def get_latest_frame(self) -> Optional[CameraFrame]:
        """Get the latest decoded frame"""
        return self.frame_queue.get_frame()
    
    def get_last_jpeg(self) -> Optional[bytes]:
        """Get the last raw JPEG bytes for pass-through (skip re-encoding)"""
        return self._last_jpeg_bytes
    
    def stop(self):
        """Stop the virtual camera"""
        self.is_running = False
        self._last_jpeg_bytes = None
        print(f"🛑 Virtual camera {self.camera_id} stopped")
    
    def get_stats(self) -> dict:
        """Get camera statistics"""
        return {
            'camera_id': self.camera_id,
            'is_running': self.is_running,
            'is_mobile': True,
            'fps': round(self.stats['fps'], 1),
            'frames_captured': self.stats['frames_captured'],
            'frames_dropped': self.stats['frames_dropped'],
            'drop_rate': (
                self.stats['frames_dropped'] / max(1, self.stats['frames_captured'] + self.stats['frames_dropped'])
            ) * 100
        }


class CameraManager:
    """Manages multiple camera captures with CPU optimization"""
    
    def __init__(self):
        self.cameras: Dict[int, CameraCapture] = {}
        self.virtual_cameras: Dict[int, VirtualCamera] = {}
        self.is_running = False
        self.processing_executor = ThreadPoolExecutor(max_workers=settings.max_cameras)
        
    async def start(self):
        """Start the camera manager"""
        self.is_running = True
        print(f"📹 Camera Manager started (max cameras: {settings.max_cameras})")
    
    async def stop(self):
        """Stop all cameras and cleanup"""
        self.is_running = False
        
        # Stop all physical cameras
        stop_tasks = [self.stop_camera(camera_id) for camera_id in list(self.cameras.keys())]
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        # Stop all virtual cameras
        for cam_id in list(self.virtual_cameras.keys()):
            self.unregister_virtual_camera(cam_id)
        
        # Cleanup executor
        self.processing_executor.shutdown(wait=True)
        print("📹 Camera Manager stopped")
    
    async def start_camera(self, camera_id: int) -> bool:
        """Start a specific camera"""
        if camera_id in self.cameras or camera_id in self.virtual_cameras:
            print(f"⚠️ Camera {camera_id} is already running")
            return False
        
        total_cameras = len(self.cameras) + len(self.virtual_cameras)
        if total_cameras >= settings.max_cameras:
            print(f"❌ Maximum camera limit reached ({settings.max_cameras})")
            return False
        
        # Run camera start in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        camera = CameraCapture(camera_id)
        
        success = await loop.run_in_executor(
            self.processing_executor,
            camera.start
        )
        
        if success:
            self.cameras[camera_id] = camera
            return True
        
        return False
    
    async def stop_camera(self, camera_id: int) -> bool:
        """Stop a specific camera"""
        if camera_id in self.virtual_cameras:
            return self.unregister_virtual_camera(camera_id)
        
        if camera_id not in self.cameras:
            return False
        
        camera = self.cameras.pop(camera_id)
        
        # Run camera stop in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.processing_executor,
            camera.stop
        )
        
        return True
    
    def register_virtual_camera(self, camera_id: Optional[int] = None) -> Optional[int]:
        """Register a new virtual camera for a mobile stream.
        
        If camera_id is None, auto-assigns the next available slot.
        Returns the assigned camera_id, or None if no slots available.
        """
        total_cameras = len(self.cameras) + len(self.virtual_cameras)
        
        if camera_id is not None:
            # Specific slot requested
            if camera_id in self.cameras or camera_id in self.virtual_cameras:
                print(f"⚠️ Camera slot {camera_id} is already in use")
                return None
            if total_cameras >= settings.max_cameras:
                print(f"❌ Maximum camera limit reached ({settings.max_cameras})")
                return None
        else:
            # Auto-assign next available slot
            if total_cameras >= settings.max_cameras:
                print(f"❌ Maximum camera limit reached ({settings.max_cameras})")
                return None
            for i in range(settings.max_cameras):
                if i not in self.cameras and i not in self.virtual_cameras:
                    camera_id = i
                    break
            if camera_id is None:
                return None
        
        virtual_cam = VirtualCamera(camera_id)
        self.virtual_cameras[camera_id] = virtual_cam
        print(f"📱 Virtual camera {camera_id} registered (mobile stream)")
        return camera_id
    
    def unregister_virtual_camera(self, camera_id: int) -> bool:
        """Unregister a virtual camera"""
        if camera_id not in self.virtual_cameras:
            return False
        vcam = self.virtual_cameras.pop(camera_id)
        vcam.stop()
        print(f"📱 Virtual camera {camera_id} unregistered")
        return True
    
    def inject_frame(self, camera_id: int, jpeg_bytes: bytes) -> bool:
        """Inject a JPEG frame into a virtual camera"""
        if camera_id in self.virtual_cameras:
            return self.virtual_cameras[camera_id].inject_frame(jpeg_bytes)
        return False
    
    def get_all_frames(self) -> List[CameraFrame]:
        """Get latest frames from all active cameras (physical + virtual)"""
        frames = []
        for camera in self.cameras.values():
            frame = camera.get_latest_frame()
            if frame is not None:
                frames.append(frame)
        for vcam in self.virtual_cameras.values():
            frame = vcam.get_latest_frame()
            if frame is not None:
                frames.append(frame)
        return frames
    
    def get_camera_frame(self, camera_id: int) -> Optional[CameraFrame]:
        """Get latest frame from specific camera"""
        if camera_id in self.cameras:
            return self.cameras[camera_id].get_latest_frame()
        if camera_id in self.virtual_cameras:
            return self.virtual_cameras[camera_id].get_latest_frame()
        return None
    
    def get_active_camera_count(self) -> int:
        """Get number of active cameras"""
        return len(self.cameras) + len(self.virtual_cameras)
    
    def get_camera_info(self) -> List[dict]:
        """Get information about all cameras"""
        info = [camera.get_stats() for camera in self.cameras.values()]
        info += [vcam.get_stats() for vcam in self.virtual_cameras.values()]
        return info
    
    def get_camera_ids(self) -> List[int]:
        """Get list of active camera IDs"""
        return list(self.cameras.keys()) + list(self.virtual_cameras.keys())
    
    def is_virtual_camera(self, camera_id: int) -> bool:
        """Check if a camera_id is a virtual (mobile) camera"""
        return camera_id in self.virtual_cameras
    
    def get_virtual_camera(self, camera_id: int) -> Optional[VirtualCamera]:
        """Get a virtual camera by ID"""
        return self.virtual_cameras.get(camera_id)