"""
Singleton Perception Pipeline
==============================
Runs  detect → track → fuse  ONCE per tick on a shared asyncio task.
All connected viewers read from the latest *snapshot* instead of each
triggering its own detection cycle.  This eliminates redundant GPU
inference and guarantees a single, consistent world state.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import msgpack
import numpy as np

from app.config import settings


# ── Snapshot dataclass ─────────────────────────────────────────────────

@dataclass
class PerceptionSnapshot:
    """Immutable result of one pipeline tick."""
    timestamp: float = 0.0
    generation: int = 0

    # Pre-serialised msgpack bytes keyed by camera_id
    camera_packets: Dict[int, bytes] = field(default_factory=dict)
    prediction_packets: Dict[int, bytes] = field(default_factory=dict)

    # World-update packet (all fused WorldObjects)
    world_update_packet: Optional[bytes] = None

    # Raw dicts for internal consumers
    world_objects_raw: List[dict] = field(default_factory=list)

    # Performance counters
    stats: dict = field(default_factory=dict)


# ── Pipeline ───────────────────────────────────────────────────────────

class PerceptionPipeline:
    """Captures → detects → tracks → fuses → publishes snapshots."""

    def __init__(self, camera_manager, detection_engine, tracking_manager, world_model):
        self.camera_manager = camera_manager
        self.detection_engine = detection_engine
        self.tracking_manager = tracking_manager
        self.world_model = world_model

        self._latest: Optional[PerceptionSnapshot] = None
        self._generation: int = 0
        self._task: Optional[asyncio.Task] = None
        self._running: bool = False

        # Shared JPEG encoding settings
        self._jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 40]
        self._max_width = 640

        # Perf tracking
        self._tick_count = 0
        self._total_tick_ms = 0.0

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        print("🧠 Perception Pipeline started")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        print("🧠 Perception Pipeline stopped")

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def latest(self) -> Optional[PerceptionSnapshot]:
        return self._latest

    @property
    def generation(self) -> int:
        return self._generation

    # ── Main loop ──────────────────────────────────────────────────────

    async def _run_loop(self):
        interval = 1.0 / settings.target_fps
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                print(f"❌ Pipeline tick error: {e}")
            await asyncio.sleep(interval)

    async def _tick(self):
        t0 = time.time()

        # 1. Grab latest frames from all cameras (physical + virtual)
        camera_frames = self.camera_manager.get_all_frames()

        # 2. Detect (runs ONCE, not per viewer)
        detection_results = (
            await self.detection_engine.process_frames(camera_frames)
            if camera_frames else []
        )

        # 3. Track
        tracking_results = []
        for det_res in detection_results:
            frame = None
            for cf in camera_frames:
                if cf.camera_id == det_res.camera_id:
                    frame = cf.frame
                    break
            tr = await self.tracking_manager.process_detections(det_res, frame)
            tracking_results.append(tr)

        # 4. World-model fusion
        if tracking_results:
            await self.world_model.update_with_tracking_results(tracking_results)

        # 5. Build snapshot
        now = time.time()
        snap = PerceptionSnapshot(
            timestamp=now,
            generation=self._generation + 1,
        )

        # 5a. Per-camera frame packets (pre-serialised)
        for i, cf in enumerate(camera_frames):
            jpeg = self._encode_frame(cf.frame)
            if jpeg is None:
                continue

            det = detection_results[i] if i < len(detection_results) else None
            trk = tracking_results[i] if i < len(tracking_results) else None

            preds = self.world_model.generate_predictions_for_camera(cf.camera_id, now)

            msg = {
                'type': 'frame',
                'camera_id': cf.camera_id,
                'timestamp': now,
                'frame_data': jpeg,
                'detections': [_ser_det(d) for d in (det.detections if det else [])],
                'tracks': [_ser_track(t) for t in (trk.tracks if trk else [])],
                'predictions': [_ser_pred(p) for p in preds],
            }
            snap.camera_packets[cf.camera_id] = msgpack.packb(msg, use_bin_type=True)

        # 5b. Prediction packets for cameras with no current frame
        active_ids = {cf.camera_id for cf in camera_frames}
        for cam_id in range(settings.max_cameras):
            if cam_id not in active_ids:
                preds = self.world_model.generate_predictions_for_camera(cam_id, now)
                if preds:
                    msg = {
                        'type': 'predictions',
                        'camera_id': cam_id,
                        'timestamp': now,
                        'predictions': [_ser_pred(p) for p in preds],
                    }
                    snap.prediction_packets[cam_id] = msgpack.packb(msg, use_bin_type=True)

        # 5c. World-objects update
        world_objs = self._build_world_objects(now)
        snap.world_objects_raw = world_objs
        if world_objs:
            snap.world_update_packet = msgpack.packb({
                'type': 'world_update',
                'timestamp': now,
                'objects': world_objs,
            }, use_bin_type=True)

        # 5d. Pipeline stats
        tick_ms = (time.time() - t0) * 1000
        self._tick_count += 1
        self._total_tick_ms += tick_ms
        snap.stats = {
            'tick_ms': round(tick_ms, 1),
            'avg_tick_ms': round(self._total_tick_ms / self._tick_count, 1),
            'cameras_active': len(camera_frames),
            'world_objects': len(world_objs),
            'ticks': self._tick_count,
        }

        # 6. Publish
        self._generation += 1
        self._latest = snap

    # ── Helpers ─────────────────────────────────────────────────────────

    def _encode_frame(self, frame: np.ndarray) -> Optional[bytes]:
        if frame is None:
            return None
        if frame.shape[1] > self._max_width:
            s = self._max_width / frame.shape[1]
            frame = cv2.resize(frame, (int(frame.shape[1] * s), int(frame.shape[0] * s)))
        ok, buf = cv2.imencode('.jpg', frame, self._jpeg_params)
        return buf.tobytes() if ok else None

    def _build_world_objects(self, current_time: float) -> List[dict]:
        objs = []
        for wo in self.world_model.get_world_objects():
            # Extract position uncertainty from Kalman filter P diagonal
            unc = (1.0, 1.0, 1.0)
            if wo.object_id in self.world_model.kalman_filters:
                kf = self.world_model.kalman_filters[wo.object_id]
                d = np.diag(kf.P)
                unc = (float(d[0]), float(d[1]), float(d[2]))

            objs.append({
                'object_id': wo.object_id,
                'world_position': wo.world_position,
                'velocity': wo.velocity,
                'position_uncertainty': unc,
                'class_id': wo.class_id,
                'class_name': wo.class_name,
                'confidence': wo.confidence,
                'last_seen_camera': wo.last_seen_camera,
                'age': current_time - wo.last_update,
                'source_tracks': wo.source_tracks,
                'prediction_confidence': wo.prediction_confidence,
            })
        return objs


# ── Module-level serialization helpers ─────────────────────────────────

def _ser_det(det) -> dict:
    d = {
        'bbox': det.bbox,
        'confidence': det.confidence,
        'class_id': det.class_id,
        'class_name': det.class_name,
        'center': det.center,
    }
    if det.keypoints is not None:
        d['keypoints'] = det.keypoints
    return d


def _ser_track(track) -> dict:
    d = {
        'track_id': track.track_id,
        'bbox': track.bbox,
        'center': track.center,
        'confidence': track.confidence,
        'class_id': track.class_id,
        'class_name': track.class_name,
        'age': track.age,
        'hits': track.hits,
        'velocity': track.velocity,
        'predicted_position': track.predicted_position,
    }
    if track.keypoints is not None:
        d['keypoints'] = track.keypoints
    return d


def _ser_pred(pred) -> dict:
    d = {
        'object_id': pred.object_id,
        'bbox': pred.predicted_bbox,
        'center': pred.predicted_center,
        'confidence': pred.confidence,
        'time_since_seen': pred.time_since_seen,
        'velocity_projection': pred.velocity_projection,
        'type': 'prediction',
        'inferred': True,
        'source_camera': getattr(pred, 'source_camera', -1),
    }
    if pred.keypoints is not None:
        d['keypoints'] = pred.keypoints
    return d
