"""
Frame encoder adapter using OpenCV.
"""
from __future__ import annotations
import logging
from typing import Optional, cast

import numpy as np
from numpy.typing import NDArray
import cv2

from app.application.ports import FrameEncoderRepository


logger = logging.getLogger(__name__)


class OpenCVFrameEncoder(FrameEncoderRepository):
    """Frame encoder using OpenCV."""
    
    def __init__(self, default_quality: int = 85):
        self._default_quality = default_quality
        logger.debug("OpenCVFrameEncoder initialized")
    
    def encode(
        self,
        frame: NDArray[np.uint8],
        quality: int = 85
    ) -> Optional[bytes]:
        """Encode frame to JPEG bytes."""
        try:
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            success, buffer = cv2.imencode('.jpg', frame, params)
            
            if success:
                return buffer.tobytes()
            else:
                logger.warning("JPEG encoding failed")
                return None
                
        except Exception as e:
            logger.error(f"Frame encoding error: {e}")
            return None
    
    def decode(self, jpeg_bytes: bytes) -> Optional[NDArray[np.uint8]]:
        """Decode JPEG bytes to frame."""
        try:
            np_arr = np.frombuffer(jpeg_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # cv2 returns a generically-typed ndarray; IMREAD_COLOR yields uint8.
            return cast("Optional[NDArray[np.uint8]]", frame)
        except Exception as e:
            logger.error(f"Frame decoding error: {e}")
            return None
