"""Tests virtual camera registration is race-safe."""
import threading
from unittest.mock import Mock

import pytest

pytest.importorskip("cv2")


def _make_repo(max_cameras=4):
    """Construct a repo without invoking heavy __init__ so the test stays unit-fast."""
    from app.infrastructure.camera_adapter import OpenCVCameraRepository
    config = Mock()
    config.get.return_value = {}
    config.get_int.return_value = max_cameras
    config.get_str.return_value = ""
    config.get_list.return_value = []
    repo = OpenCVCameraRepository.__new__(OpenCVCameraRepository)
    repo._config = config
    repo._max_cameras = max_cameras
    repo._cameras = {}
    repo._virtual_cameras = {}
    repo._virtual_camera_lock = threading.Lock()
    repo._target_resolution = (1280, 720)
    return repo


def test_concurrent_registration_yields_unique_slots():
    repo = _make_repo(max_cameras=4)
    results = []

    def reg():
        cid = repo.register_virtual_camera()
        results.append(cid)

    threads = [threading.Thread(target=reg) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assigned = [r for r in results if r is not None]
    assert len(set(assigned)) == len(assigned), f"Duplicate slot allocations: {results}"
    assert len(assigned) <= 4
