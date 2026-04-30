"""
Detection repository adapter.
Provides a permanent solution with proper model management.
"""
from __future__ import annotations
import asyncio
import logging
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
import cv2

from app.application.ports import DetectionRepository, ConfigurationRepository
from app.domain.entities import (
    CameraFrame, Detection, BoundingBox, Keypoint,
    AppearanceDescriptor
)


logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLOv8 detector with proper error handling."""
    
    def __init__(self, config_repo: ConfigurationRepository):
        self._config = config_repo
        self._model = None
        self._model_path = config_repo.get("model_path", "yolov8n.pt")
        self._device = config_repo.get("device", "auto")
        self._half_precision = config_repo.get_bool("half_precision", False)
        self._confidence_threshold = config_repo.get_float("confidence_threshold", 0.5)
        self._iou_threshold = config_repo.get_float("iou_threshold", 0.45)
        self._max_detections = config_repo.get_int("max_detections", 100)
        self._detection_classes = config_repo.get_list("detection_classes", [0])
        self._is_initialized = False
        
        # Try to import ultralytics
        try:
            from ultralytics import YOLO
            self._YOLO = YOLO
            logger.info("Ultralytics YOLO imported successfully")
        except ImportError:
            self._YOLO = None
            logger.error("Ultralytics not available. Install with: pip install ultralytics")
            raise RuntimeError("Detection engine requires ultralytics. Run: pip install ultralytics")
    
    async def initialize(self) -> None:
        """Initialize the YOLO model."""
        if self._is_initialized:
            return
        
        if self._YOLO is None:
            raise RuntimeError("Ultralytics YOLO not available")
        
        try:
            logger.info(f"Loading YOLO model: {self._model_path}")
            
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: self._YOLO(self._model_path, task='detect')
            )
            
            # Setup device
            self._setup_device()
            
            self._is_initialized = True
            logger.info(f"YOLO model loaded on {self._device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            raise RuntimeError(f"Failed to initialize detection model: {e}")
    
    def _setup_device(self) -> None:
        """Configure compute device."""
        try:
            import torch
            
            if self._device == "auto":
                if torch.cuda.is_available():
                    self._device = "cuda:0"
                    gpu_name = torch.cuda.get_device_name(0)
                    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    logger.info(f"GPU detected: {gpu_name} ({vram:.1f} GB)")
                else:
                    self._device = "cpu"
                    logger.info("CUDA not available, using CPU")
            
            # Move model to device (skip for TensorRT .engine files)
            if not self._model_path.endswith(('.engine', '.trt')):
                if hasattr(self._model, 'to'):
                    self._model.to(self._device)
                    logger.info(f"Model moved to {self._device}")
            
        except ImportError:
            self._device = "cpu"
            logger.warning("PyTorch not available, using CPU")
    
    def detect(self, frame: NDArray[np.uint8]) -> List[Detection]:
        """Run detection on a single frame."""
        if not self._is_initialized or self._model is None:
            raise RuntimeError("Detector not initialized")
        
        try:
            # Build inference arguments
            infer_kwargs = {
                'conf': self._confidence_threshold,
                'iou': self._iou_threshold,
                'max_det': self._max_detections,
                'verbose': False,
            }
            
            if self._detection_classes:
                infer_kwargs['classes'] = self._detection_classes
            
            if self._half_precision and self._device != "cpu":
                infer_kwargs['half'] = True
            
            # Run inference
            results = self._model(frame, **infer_kwargs)
            
            detections = []
            timestamp = datetime.now()
            
            for result in results:
                if result.boxes is None:
                    continue
                
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                logger.debug(f"Detection: camera={frame.camera_id if hasattr(frame, 'camera_id') else '?'}, boxes={len(boxes)}, classes={np.unique(classes).tolist()}")
                
                # Extract keypoints if available
                keypoints_data = None
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    try:
                        keypoints_data = result.keypoints.data.cpu().numpy()
                    except Exception:
                        pass
                
                for idx, (box, conf, cls_id) in enumerate(zip(boxes, confidences, classes)):
                    x1, y1, x2, y2 = box
                    
                    try:
                        bbox = BoundingBox(float(x1), float(y1), float(x2), float(y2))
                    except ValueError:
                        continue
                    
                    # Extract keypoints for this detection
                    keypoints = []
                    if keypoints_data is not None and idx < keypoints_data.shape[0]:
                        person_kp = keypoints_data[idx]
                        for j in range(person_kp.shape[0]):
                            keypoints.append(Keypoint(
                                x=float(person_kp[j, 0]),
                                y=float(person_kp[j, 1]),
                                confidence=float(person_kp[j, 2])
                            ))
                    
                    detection = Detection(
                        detection_id=f"det_{timestamp.timestamp()}_{idx}_{id(boxes)}",
                        camera_id=-1,  # Will be set by caller
                        bbox=bbox,
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name=self._model.names.get(cls_id, 'unknown'),
                        timestamp=timestamp,
                        keypoints=keypoints
                    )
                    
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            raise RuntimeError(f"Detection failed: {e}") from e


class DetectionRepositoryImpl(DetectionRepository):
    """Implementation of detection repository."""
    
    def __init__(self, config_repo: ConfigurationRepository):
        self._config = config_repo
        self._detector: Optional[YOLODetector] = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._is_ready = False
        
        logger.info("DetectionRepositoryImpl initialized")
    
    async def initialize(self) -> None:
        """Initialize the detection engine."""
        self._detector = YOLODetector(self._config)
        await self._detector.initialize()
        self._is_ready = True
        logger.info("Detection engine initialized")
    
    async def detect(self, frame: CameraFrame) -> List[Detection]:
        """Run detection on a single frame."""
        if not self._is_ready:
            raise RuntimeError("Detection engine not initialized")
        
        loop = asyncio.get_running_loop()
        detections = await loop.run_in_executor(
            self._executor,
            self._detector.detect,
            frame.frame_data
        )
        
        # Set camera ID for all detections
        for det in detections:
            det.camera_id = frame.camera_id
        
        return detections
    
    async def detect_batch(self, frames: List[CameraFrame]) -> Dict[int, List[Detection]]:
        """Run detection on multiple frames concurrently."""
        if not frames:
            return {}
        
        tasks = [self.detect(frame) for frame in frames]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        detections_map = {}
        for frame, result in zip(frames, results):
            if isinstance(result, Exception):
                logger.error(f"Detection failed for camera {frame.camera_id}: {result}")
                detections_map[frame.camera_id] = []
            else:
                detections_map[frame.camera_id] = result
        
        return detections_map
    
    def is_ready(self) -> bool:
        """Check if detector is ready."""
        return self._is_ready
    
    async def compute_appearance(
        self,
        frame: NDArray[np.uint8],
        bbox: BoundingBox
    ) -> Optional[AppearanceDescriptor]:
        """Compute HSV histogram appearance descriptor."""
        try:
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
            h, w = frame.shape[:2]
            
            # Clamp to frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            crop = frame[y1:y2, x1:x2]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            
            # 32 hue + 16 sat + 16 val = 64-dim histogram
            hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
            hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
            hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
            
            feature = np.concatenate([hist_h, hist_s, hist_v])
            norm = np.linalg.norm(feature) + 1e-6
            normalized = (feature / norm).astype(np.float32)
            
            return AppearanceDescriptor(vector=normalized)
            
        except Exception as e:
            logger.error(f"Appearance computation error: {e}")
            return None
