"""Cross-camera ground-plane homography estimation (Phase B2).

Learns a projective transform H_{src->dst} between two cameras' image planes from
foot-point correspondences of the same person observed by both cameras, then projects
a foot point from one camera into another to render a cross-camera "ghost" prediction.

The estimator is self-calibrating: correspondences accumulate as the world model
re-identifies the same object across cameras; once enough pairs exist for a camera
pair, ``cv2.findHomography`` + RANSAC computes the transform, re-estimated periodically.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 always present in runtime/CI
    cv2 = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

Point = Tuple[float, float]
CamPair = Tuple[int, int]


class HomographyEstimator:
    """Accumulates foot-point correspondences and estimates per-camera-pair homographies."""

    def __init__(
        self,
        min_pairs: int = 4,
        max_pairs: int = 100,
        ransac_threshold: float = 12.0,
        reestimate_every: int = 5,
    ) -> None:
        # findHomography needs at least 4 points.
        self._min_pairs = max(4, int(min_pairs))
        self._max_pairs = max(self._min_pairs, int(max_pairs))
        self._ransac_threshold = float(ransac_threshold)
        self._reestimate_every = max(1, int(reestimate_every))

        self._pairs: Dict[CamPair, List[Tuple[Point, Point]]] = defaultdict(list)
        self._homographies: Dict[CamPair, NDArray[np.float64]] = {}
        self._since_estimate: Dict[CamPair, int] = defaultdict(int)

    def add_correspondence(
        self, src_cam: int, dst_cam: int, src_pt: Point, dst_pt: Point
    ) -> None:
        """Record a matched foot point seen as ``src_pt`` in ``src_cam`` and ``dst_pt``
        in ``dst_cam``. Triggers (re-)estimation once enough pairs are available."""
        if src_cam == dst_cam:
            return
        key = (src_cam, dst_cam)
        pairs = self._pairs[key]
        pairs.append((src_pt, dst_pt))
        if len(pairs) > self._max_pairs:
            pairs.pop(0)  # keep the most recent observations
        self._since_estimate[key] += 1

        ready = len(pairs) >= self._min_pairs
        first_time = key not in self._homographies
        if ready and (first_time or self._since_estimate[key] >= self._reestimate_every):
            self._estimate(key)

    def _estimate(self, key: CamPair) -> None:
        if cv2 is None:
            return
        pairs = self._pairs[key]
        src = np.array([p[0] for p in pairs], dtype=np.float64)
        dst = np.array([p[1] for p in pairs], dtype=np.float64)
        try:
            H, _mask = cv2.findHomography(src, dst, cv2.RANSAC, self._ransac_threshold)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"findHomography failed for cam{key[0]}->cam{key[1]}: {e}")
            return
        if H is not None and np.isfinite(H).all():
            self._homographies[key] = np.asarray(H, dtype=np.float64)
            self._since_estimate[key] = 0
            logger.debug(
                f"H learned: cam{key[0]}->cam{key[1]} from {len(pairs)} pairs"
            )

    def has_homography(self, src_cam: int, dst_cam: int) -> bool:
        return (src_cam, dst_cam) in self._homographies

    def source_cameras_for(self, dst_cam: int) -> List[int]:
        """Cameras that currently have a valid homography projecting into ``dst_cam``."""
        return [a for (a, b) in self._homographies if b == dst_cam]

    def project(self, src_cam: int, dst_cam: int, point: Point) -> Optional[Point]:
        """Project a ``src_cam`` ground-plane point into ``dst_cam`` pixel space."""
        H = self._homographies.get((src_cam, dst_cam))
        if H is None:
            return None
        v = H @ np.array([point[0], point[1], 1.0])
        if abs(v[2]) < 1e-9:
            return None
        return (float(v[0] / v[2]), float(v[1] / v[2]))
