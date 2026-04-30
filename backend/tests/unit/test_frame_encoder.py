"""Tests for the OpenCV JPEG frame encoder."""
import numpy as np
import pytest

# Skip whole module if cv2 isn't installed
pytest.importorskip("cv2")

from app.infrastructure.frame_encoder_adapter import OpenCVFrameEncoder


def test_encode_returns_jpeg_bytes():
    enc = OpenCVFrameEncoder()
    img = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    jpeg = enc.encode(img)
    assert isinstance(jpeg, (bytes, bytearray))
    assert len(jpeg) > 0
    # JPEG SOI marker
    assert jpeg[0:2] == b"\xff\xd8"


def test_encode_returns_nonempty_for_zeros():
    enc = OpenCVFrameEncoder()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    jpeg = enc.encode(img)
    assert jpeg is not None
    assert len(jpeg) > 100


def test_decode_roundtrip_yields_image_with_same_shape():
    enc = OpenCVFrameEncoder()
    img = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    jpeg = enc.encode(img)
    decoded = enc.decode(jpeg)
    assert decoded is not None
    assert decoded.shape == img.shape


def test_decode_invalid_bytes_returns_none():
    enc = OpenCVFrameEncoder()
    out = enc.decode(b"not-a-jpeg")
    # cv2.imdecode returns None on failure; our adapter forwards that.
    assert out is None
