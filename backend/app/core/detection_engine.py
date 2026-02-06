"""
Detection Engine using YOLOv8 with Jetson Orin Nano / GPU optimization.

Supports:
- YOLOv8 .pt (PyTorch), .engine (TensorRT), .onnx formats
- Person-only detection via COCO class filter (classes=[0])
- FP16 half-precision on CUDA / Jetson
- Graceful fallback to mock detector if ultralytics is missing
"""

import asyncio
import os
import time
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

try:
    from ultralytics import YOLO
except ImportError:
    print("⚠️ Ultralytics not available, using mock detection")
    YOLO = None

from app.config import settings
from app.core.camera_manager import CameraFrame


@dataclass
class Detection:
    """Container for detection results"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    center: Tuple[float, float]  # x, y center point
    feature_vector: object = None  # appearance descriptor (np.ndarray or None)
    keypoints: Optional[List[Tuple[float, float, float]]] = None  # COCO 17-joint (x, y, conf)


@dataclass
class DetectionResult:
    """Container for frame detection results"""
    camera_id: int
    frame_number: int
    timestamp: float
    detections: List[Detection]
    processing_time: float


class YOLODetector:
    """YOLOv8 detector with Jetson / GPU acceleration and person-only filtering"""
    
    def __init__(self):
        self.model: Optional[object] = None
        self.class_names: Dict[int, str] = {}
        self.is_initialized = False
        self.device = "cpu"
        
    async def initialize(self):
        """Initialize the YOLO model"""
        try:
            if YOLO is None:
                # Mock initialization for development
                print("⚠️ Using mock YOLO detector (person-only)")
                self.is_initialized = True
                self.class_names = {0: 'person'}
                return
            
            # Resolve model path from config
            model_path = settings.model_path
            if not os.path.exists(model_path):
                # Try common fallback locations
                for fallback in ['yolov8n.pt', 'models/yolov8n.pt', 'yolov8n.engine']:
                    if os.path.exists(fallback):
                        model_path = fallback
                        break
            
            print(f"🔄 Loading model: {model_path}")
            
            # Load model in thread pool to avoid blocking
            # Pass task='detect' to avoid warning for .engine files
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                lambda: YOLO(model_path, task='detect')
            )
            
            # Set up compute device (CPU / CUDA / Jetson)
            self._setup_device()
            
            # Get class names from model
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
            
            self.is_initialized = True
            
            # Log configuration summary
            if settings.detection_classes:
                class_desc = ', '.join(
                    self.class_names.get(c, f'class_{c}') for c in settings.detection_classes
                )
                print(f"📋 Detection filter: {class_desc} (classes={settings.detection_classes})")
            else:
                print(f"📋 Detection filter: ALL {len(self.class_names)} classes")
            
            print(f"✅ Model loaded on {self.device}")
            if settings.half_precision and self.device != "cpu":
                print(f"⚡ Half precision (FP16) enabled")
            
        except Exception as e:
            print(f"❌ Failed to initialize YOLO model: {e}")
            # Fall back to mock detector
            self.is_initialized = True
            self.class_names = {0: 'person'}
    
    def _setup_device(self):
        """Configure compute device (CPU / CUDA / Jetson Orin Nano)"""
        try:
            import torch
            
            if settings.device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda:0"
                    gpu_name = torch.cuda.get_device_name(0)
                    vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
                    print(f"🖥️  GPU detected: {gpu_name} ({vram:.1f} GB)")
                else:
                    self.device = "cpu"
                    print("⚠️ CUDA not available, using CPU")
            else:
                self.device = settings.device
            
            # Move model to target device (skip for TensorRT .engine — already GPU-bound)
            model_path = settings.model_path
            is_engine = model_path.endswith('.engine') or model_path.endswith('.trt')
            if self.device != "cpu" and not is_engine and hasattr(self.model, 'to'):
                self.model.to(self.device)
                print(f"✅ Model moved to {self.device}")
            elif is_engine:
                self.device = "cuda:0"
                print(f"✅ TensorRT engine loaded (GPU-bound)")
                
        except ImportError:
            self.device = "cpu"
            print("⚠️ PyTorch not available, using CPU")
        except Exception as e:
            self.device = "cpu"
            print(f"⚠️ GPU setup failed, falling back to CPU: {e}")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Perform person detection on a single frame"""
        if not self.is_initialized:
            return []
        
        if self.model is None:
            # Mock detection for development
            return self._mock_detection(frame)
        
        try:
            # Build inference kwargs
            infer_kwargs = dict(
                conf=settings.confidence_threshold,
                iou=settings.iou_threshold,
                max_det=settings.max_detections,
                verbose=False,
            )
            
            # Person-only filter: classes=[0] tells YOLO to only output person class
            # This filters during NMS — the most efficient approach
            if settings.detection_classes:
                infer_kwargs['classes'] = settings.detection_classes
            
            # FP16 half precision for Jetson TensorRT / CUDA acceleration
            if settings.half_precision and self.device != "cpu":
                infer_kwargs['half'] = True
            
            # Run inference
            results = self.model(frame, **infer_kwargs)
            
            detections = []
            
            # Check if model provides keypoints (YOLOv8-pose)
            _pose_available = settings.pose_enabled
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    
                    # Extract keypoints tensor if available: (N, 17, 3)
                    kp_data = None
                    if _pose_available and hasattr(result, 'keypoints') and result.keypoints is not None:
                        try:
                            kp_data = result.keypoints.data.cpu().numpy()
                        except Exception:
                            kp_data = None
                    
                    for idx, (box, conf, cls_id) in enumerate(zip(boxes, confidences, classes)):
                        x1, y1, x2, y2 = box
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # Compute lightweight appearance descriptor
                        feat = self._compute_appearance_feature(frame, (x1, y1, x2, y2))
                        
                        # Per-person keypoints: list of (x, y, conf) tuples
                        kpts = None
                        if kp_data is not None and idx < kp_data.shape[0]:
                            person_kp = kp_data[idx]  # (17, 3)
                            kpts = [(float(person_kp[j, 0]), float(person_kp[j, 1]), float(person_kp[j, 2])) for j in range(person_kp.shape[0])]
                        
                        detection = Detection(
                            bbox=(float(x1), float(y1), float(x2), float(y2)),
                            confidence=float(conf),
                            class_id=int(cls_id),
                            class_name=self.class_names.get(cls_id, 'person'),
                            center=(float(center_x), float(center_y)),
                            feature_vector=feat,
                            keypoints=kpts,
                        )
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return []
    
    def _compute_appearance_feature(self, frame: np.ndarray, bbox: Tuple) -> Optional[np.ndarray]:
        """Compute lightweight HSV histogram appearance descriptor (64-dim, L2-normalised).
        
        Fast to compute (~0.1ms), discriminative enough for cross-camera re-ID
        of tracked persons without requiring an additional neural network.
        """
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                return None

            crop = frame[y1:y2, x1:x2]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

            # 32 hue bins + 16 saturation bins + 16 value bins = 64-dim
            hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
            hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
            hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

            feature = np.concatenate([hist_h, hist_s, hist_v])
            norm = np.linalg.norm(feature) + 1e-6
            return (feature / norm).astype(np.float32)
        except Exception:
            return None

    def _mock_detection(self, frame: np.ndarray) -> List[Detection]:
        """Mock person-only detection for development/testing"""
        height, width = frame.shape[:2]
        detections = []
        
        # Fake person detection in center region
        if np.random.random() > 0.3:  # 70% chance
            x1 = width * 0.35
            y1 = height * 0.15
            x2 = width * 0.65
            y2 = height * 0.90
            
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                confidence=0.87,
                class_id=0,
                class_name='person',
                center=((x1 + x2) / 2, (y1 + y2) / 2)
            ))
        
        return detections


class DetectionEngine:
    """Main detection engine with CPU optimization"""
    
    def __init__(self):
        self.detector = YOLODetector()
        self.processing_executor = ThreadPoolExecutor(max_workers=2)  # Limit for CPU
        self.is_running = False
        self.stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'average_processing_time': 0,
            'fps': 0,
            'last_fps_update': time.time()
        }
        
    async def initialize(self):
        """Initialize the detection engine"""
        await self.detector.initialize()
        self.is_running = True
        print("🎯 Detection Engine initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        self.processing_executor.shutdown(wait=True)
        print("🎯 Detection Engine cleaned up")
    
    async def process_frame(self, camera_frame: CameraFrame) -> DetectionResult:
        """Process a single camera frame"""
        if not self.is_running:
            return DetectionResult(
                camera_id=camera_frame.camera_id,
                frame_number=camera_frame.frame_number,
                timestamp=camera_frame.timestamp,
                detections=[],
                processing_time=0
            )
        
        start_time = time.time()
        
        # Run detection in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        detections = await loop.run_in_executor(
            self.processing_executor,
            self.detector.detect,
            camera_frame.frame
        )
        
        processing_time = time.time() - start_time
        
        # Update statistics
        self._update_stats(detections, processing_time)
        
        return DetectionResult(
            camera_id=camera_frame.camera_id,
            frame_number=camera_frame.frame_number,
            timestamp=camera_frame.timestamp,
            detections=detections,
            processing_time=processing_time
        )
    
    async def process_frames(self, camera_frames: List[CameraFrame]) -> List[DetectionResult]:
        """Process multiple frames concurrently"""
        if not camera_frames:
            return []
        
        # Process frames concurrently
        tasks = [self.process_frame(frame) for frame in camera_frames]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, DetectionResult):
                valid_results.append(result)
            else:
                print(f"❌ Detection processing error: {result}")
        
        return valid_results
    
    def _update_stats(self, detections: List[Detection], processing_time: float):
        """Update detection statistics"""
        self.stats['frames_processed'] += 1
        self.stats['total_detections'] += len(detections)
        
        # Update average processing time
        frame_count = self.stats['frames_processed']
        current_avg = self.stats['average_processing_time']
        self.stats['average_processing_time'] = (
            (current_avg * (frame_count - 1) + processing_time) / frame_count
        )
        
        # Update FPS every second
        current_time = time.time()
        if current_time - self.stats['last_fps_update'] >= 1.0:
            fps = 1.0 / max(0.001, self.stats['average_processing_time'])
            self.stats['fps'] = round(fps, 1)
            self.stats['last_fps_update'] = current_time
    
    def get_stats(self) -> dict:
        """Get detection engine statistics"""
        return {
            'frames_processed': self.stats['frames_processed'],
            'total_detections': self.stats['total_detections'],
            'average_processing_time_ms': round(self.stats['average_processing_time'] * 1000, 2),
            'detection_fps': self.stats['fps'],
            'detections_per_frame': (
                self.stats['total_detections'] / max(1, self.stats['frames_processed'])
            )
        }
    
    def get_class_names(self) -> Dict[int, str]:
        """Get available detection classes"""
        return self.detector.class_names.copy()
    
    def is_ready(self) -> bool:
        """Check if detection engine is ready"""
        return self.is_running and self.detector.is_initialized