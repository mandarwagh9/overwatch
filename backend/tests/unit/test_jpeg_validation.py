"""Tests JPEG magic-byte gate before cv2.imdecode."""
import pytest

pytest.importorskip("cv2")


def test_inject_rejects_empty():
    from app.infrastructure.camera_adapter import VirtualCamera
    cam = VirtualCamera(camera_id=0, max_width=640)
    assert cam.inject_frame(b"") is False


def test_inject_rejects_non_jpeg():
    from app.infrastructure.camera_adapter import VirtualCamera
    cam = VirtualCamera(camera_id=0, max_width=640)
    # GIF magic bytes
    assert cam.inject_frame(b"GIF89a" + b"\x00" * 20) is False


def test_inject_passes_magic_check_for_jpeg_start():
    """A payload with valid JPEG SOI but invalid body should not raise."""
    from app.infrastructure.camera_adapter import VirtualCamera
    cam = VirtualCamera(camera_id=0, max_width=640)
    # Has the SOI marker but isn't a real JPEG; cv2.imdecode will return None
    cam.inject_frame(b"\xff\xd8\xff" + b"\x00" * 100)  # must not raise
