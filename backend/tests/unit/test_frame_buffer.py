"""Tests locking FrameBuffer drop-old semantics."""
from datetime import datetime

import numpy as np
import pytest

# Skip whole module if cv2 isn't installed (camera_adapter imports cv2 at module load)
pytest.importorskip("cv2")

from app.domain.entities import CameraFrame
from app.infrastructure.camera_adapter import FrameBuffer


def _frame(n: int) -> CameraFrame:
    return CameraFrame(
        camera_id=0,
        frame_data=np.full((10, 10, 3), n % 256, dtype=np.uint8),
        timestamp=datetime.now(),
        frame_number=n,
    )


def test_put_get_roundtrip():
    buf = FrameBuffer()
    f = _frame(1)
    assert buf.put(f) is True
    out = buf.get()
    assert out is not None
    assert out.frame_number == 1


def test_get_returns_none_when_empty():
    buf = FrameBuffer()
    assert buf.get() is None


def test_buffer_drops_oldest_when_full():
    """LOAD-BEARING: pipeline relies on always-newest semantics.
    Default FrameBuffer has max_size=2 with a Queue(maxsize=2)."""
    buf = FrameBuffer()
    # Push three frames into a 2-slot buffer; oldest must be dropped.
    assert buf.put(_frame(1)) is True
    assert buf.put(_frame(2)) is True
    assert buf.put(_frame(3)) is True

    # The two retained frames should be the two newest (2 and 3),
    # in FIFO order.
    first = buf.get()
    second = buf.get()
    assert first is not None and second is not None
    retained = {first.frame_number, second.frame_number}
    assert retained == {2, 3}
    # After draining, buffer is empty.
    assert buf.get() is None


def test_put_increments_frame_counter():
    buf = FrameBuffer()
    assert buf.frame_counter == 0
    buf.put(_frame(0))
    assert buf.frame_counter == 1
    buf.put(_frame(1))
    assert buf.frame_counter == 2
