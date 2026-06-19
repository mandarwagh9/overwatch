"""Integration tests for the perception pipeline orchestration (services.py).

Drives PerceptionPipelineService with mock adapters to assert the detect -> track
-> fuse -> snapshot -> broadcast flow wires together, plus metrics and lifecycle.
No cv2/torch required — services depends only on domain + ports.
"""
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services import (
    HealthCheckService, PerceptionPipelineService, PipelineMetrics,
)
from app.domain.entities import Point3D, Velocity3D, WorldObject


def _world_object():
    return WorldObject(
        object_id=1, position=Point3D(0, 0, 3), velocity=Velocity3D(0, 0, 0),
        class_id=0, class_name="person", confidence=0.9,
        last_seen_camera=0, last_update=datetime.now(),
    )


def _service(frames, detections_map, tracks_for_cam0, world_objects, predictions=None):
    camera_repo = MagicMock()
    camera_repo.get_latest_frames.return_value = frames

    detection_repo = MagicMock()
    detection_repo.detect_batch = AsyncMock(return_value=detections_map)

    tracking_repo = MagicMock()
    tracking_repo.update = AsyncMock(return_value=tracks_for_cam0)

    world_model_repo = MagicMock()
    world_model_repo.update = AsyncMock(return_value=world_objects)
    world_model_repo.generate_predictions = MagicMock(return_value=(predictions or []))

    communication_repo = MagicMock()
    communication_repo.broadcast_snapshot = AsyncMock()

    svc = PerceptionPipelineService(
        camera_repo=camera_repo,
        detection_repo=detection_repo,
        tracking_repo=tracking_repo,
        world_model_repo=world_model_repo,
        communication_repo=communication_repo,
        frame_encoder_repo=MagicMock(),
        target_fps=30.0,
    )
    return svc, communication_repo


def test_tick_builds_and_broadcasts_snapshot(make_frame, make_detection, make_track):
    frame = make_frame(camera_id=0)
    det = make_detection(camera_id=0)
    trk = make_track(camera_id=0)
    obj = _world_object()
    svc, comm = _service([frame], {0: [det]}, [trk], [obj])

    asyncio.run(svc._process_tick())

    snap = svc.latest_snapshot
    assert snap is not None
    assert snap.generation == 1
    assert snap.detections[0] == [det]
    assert snap.tracks[0] == [trk]
    assert snap.world_objects == [obj]
    assert 0 in snap.camera_frames
    comm.broadcast_snapshot.assert_awaited_once()


def test_tick_with_no_frames_is_a_noop():
    svc, comm = _service([], {}, [], [])
    asyncio.run(svc._process_tick())
    assert svc.latest_snapshot is None
    comm.broadcast_snapshot.assert_not_awaited()


def test_generation_increments_each_tick(make_frame, make_detection, make_track):
    frame = make_frame(camera_id=0)
    svc, _ = _service([frame], {0: [make_detection(camera_id=0)]}, [make_track(camera_id=0)], [_world_object()])
    asyncio.run(svc._process_tick())
    asyncio.run(svc._process_tick())
    assert svc.latest_snapshot.generation == 2


def test_metrics_update_tracks_running_average():
    m = PipelineMetrics()
    m.update(10.0)
    m.update(20.0)
    assert m.frames_processed == 2
    assert m.total_processing_time_ms == pytest.approx(30.0)
    assert m.average_processing_time_ms == pytest.approx(15.0)


def test_start_then_stop_lifecycle():
    svc, _ = _service([], {}, [], [])

    async def run():
        await svc.start()
        assert svc._running is True
        await asyncio.sleep(0.03)
        await svc.stop()

    asyncio.run(run())
    assert svc._running is False


def test_health_status_reports_pipeline_state(make_frame, make_detection, make_track):
    frame = make_frame(camera_id=0)
    svc, _ = _service([frame], {0: [make_detection(camera_id=0)]}, [make_track(camera_id=0)], [_world_object()])
    asyncio.run(svc._process_tick())

    camera_repo = MagicMock()
    camera_repo.get_camera_count.return_value = 1
    detection_repo = MagicMock()
    detection_repo.is_ready.return_value = True
    tracking_repo = MagicMock()
    tracking_repo.is_ready.return_value = True
    health = HealthCheckService(camera_repo, detection_repo, tracking_repo, svc)

    status = health.get_status()
    assert status["status"] == "healthy"
    assert status["detection"]["ready"] is True
    assert status["pipeline"]["world_objects"] == 1
