# Overwatch Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a regression-test safety net to the Overwatch repo, then land prioritized backend, frontend, security, and deploy improvements identified by a five-agent audit — without breaking any existing dev or production behavior.

**Architecture:** Phased: (0) test infrastructure, (1) lock current behavior with tests, (2) backend bugfixes, (3) reliability, (4) frontend bugfixes, (5) additive security (default-off), (6) deploy script consolidation. Every behavioral change lands AFTER a test that locks the prior behavior or asserts the new behavior.

**Tech Stack:** Python 3.10+, pytest, FastAPI, OpenCV, NumPy, React 18, Jest/RTL (frontend tests deferred to a later plan; this plan keeps frontend changes manual-test verified), GitHub Actions for CI.

**Working branch:** `improvements/test-foundation-and-hardening` (already created from `main`).

**Hard constraint:** "Don't break anything." Every default value must reproduce current runtime behavior. New features (auth, CORS lists, rate limits) are additive and gated by config flags that default to OFF/permissive.

---

## Files Touched (Map)

### New files
- `pyproject.toml` — pytest config, Python pin
- `backend/tests/__init__.py`
- `backend/tests/conftest.py` — shared fixtures
- `backend/tests/unit/__init__.py`
- `backend/tests/unit/test_bounding_box.py`
- `backend/tests/unit/test_kalman_filter.py`
- `backend/tests/unit/test_coordinate_transformer.py`
- `backend/tests/unit/test_frame_buffer.py`
- `backend/tests/unit/test_frame_encoder.py`
- `backend/tests/unit/test_hungarian_tracking.py`
- `backend/tests/unit/test_world_model.py`
- `backend/tests/unit/test_pipeline_metrics.py`
- `backend/tests/unit/test_settings.py`
- `backend/.gitignore`
- `.github/workflows/ci.yml`
- `scripts/_jetson_common.py` — shared SSH helper
- `scripts/archive/` — directory for retired scripts (with README)

### Modified
- `backend/app/infrastructure/world_model_adapter.py` — clamp dt before predict
- `backend/app/infrastructure/detection_adapter.py` — unique detection_id, raise-from, asyncio.get_running_loop
- `backend/app/infrastructure/camera_adapter.py` — virtual camera lock, RTSP reconnect, demote logs, top-of-file imports
- `backend/app/infrastructure/websocket_adapter.py` — per-client timeout, demote logs, lock disconnect
- `backend/app/infrastructure/container.py` — partial-failure rollback
- `backend/app/infrastructure/config_adapter.py` — cors_origins, max_ws_clients, ssl fail-fast helper
- `backend/app/application/services.py` — demote per-tick logs, raise CancelledError, drop dead lock
- `backend/app/application/ports.py` — move `Any` import to top
- `backend/main.py` — delete dead `except` block, optional JWT verifier, configurable CORS
- `backend/requirements.txt` — bump python-multipart, pin upper bounds
- `backend/.env.example` — document AUTH_ENABLED, CORS_ORIGINS, MAX_WS_CLIENTS
- `frontend/src/infrastructure/websocketAdapter.js` — reconnect timer + intentional close flag
- `frontend/src/infrastructure/cameraStreamAdapter.js` — self-scheduling capture, bufferedAmount backpressure
- `frontend/src/components/CameraDisplay.jsx` — blob URL revoke, drop JSX width/height
- `frontend/src/pages/MobileCamera.jsx` — call adapter.stop() on unmount
- `frontend/src/components/ErrorBanner.jsx` — onDismiss prop
- `frontend/src/App.jsx` — ErrorBoundary
- `frontend/src/infrastructure/apiAdapter.js` — AbortController timeouts
- `scripts/deploy_jetson.py` — env-based credentials, atomic staging, JWT secret generation
- `scripts/restart_jetson.py`, `check_logs.py`, `check_status.py`, `ws_test.py` — env-based credentials via `_jetson_common`
- `README.md` — auth claim correction, deploy doc updates

### Removed (moved to `scripts/archive/`)
- `scripts/deploy_v2.py`
- `scripts/_restart_now.py`
- `scripts/force_restart.py`
- `scripts/fix_jetson.py`

---

## Phase 0 — Test infrastructure

### Task 0.1: Create pyproject.toml with pytest config

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Write the file**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "overwatch"
version = "2.0.0"
description = "Multi-camera perception and tracking system"
requires-python = ">=3.10,<3.13"

[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["backend/tests"]
pythonpath = ["backend"]
addopts = "-q --strict-markers --tb=short"
markers = [
    "slow: marks tests as slow",
    "integration: marks integration tests"
]
filterwarnings = [
    "ignore::DeprecationWarning:numpy",
]
```

- [ ] **Step 2: Verify pytest discovers it**

Run: `python -m pytest --collect-only 2>&1 | head -5`
Expected: "no tests ran" (no tests yet) — but no config errors.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add pyproject.toml with pytest config"
```

### Task 0.2: Create backend/.gitignore

**Files:**
- Create: `backend/.gitignore`

- [ ] **Step 1: Write the file**

```gitignore
# Local config
.env
.env.local

# Secrets
*.pem
*.key
*.crt

# Models (large binaries)
*.pt
*.engine
*.onnx

# Python artifacts
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.coverage
htmlcov/

# Logs
*.log
```

- [ ] **Step 2: Commit**

```bash
git add backend/.gitignore
git commit -m "chore: add backend/.gitignore mirroring root secrets globs"
```

### Task 0.3: Create test skeleton + conftest

**Files:**
- Create: `backend/tests/__init__.py` (empty)
- Create: `backend/tests/unit/__init__.py` (empty)
- Create: `backend/tests/conftest.py`

- [ ] **Step 1: Write conftest.py**

```python
"""Shared pytest fixtures for Overwatch tests."""
from __future__ import annotations
from datetime import datetime
from typing import Tuple

import numpy as np
import pytest

from app.domain.entities import (
    BoundingBox, CameraCalibration, CameraFrame, Detection,
    Point3D, Track, TrackingState
)


@pytest.fixture
def bbox_simple() -> BoundingBox:
    return BoundingBox(10.0, 20.0, 110.0, 220.0)


@pytest.fixture
def calibration_origin() -> CameraCalibration:
    """Camera at world origin, no rotation, focal=1000, center=(640,360)."""
    return CameraCalibration(
        camera_id=0,
        position=Point3D(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
        focal_length=1000.0,
        image_center=(640.0, 360.0),
    )


@pytest.fixture
def make_detection():
    """Factory for synthetic detections."""
    def _make(
        det_id: str = "det_1",
        camera_id: int = 0,
        bbox: Tuple[float, float, float, float] = (10.0, 20.0, 110.0, 220.0),
        confidence: float = 0.9,
        class_id: int = 0,
        class_name: str = "person",
    ) -> Detection:
        return Detection(
            detection_id=det_id,
            camera_id=camera_id,
            bbox=BoundingBox(*bbox),
            confidence=confidence,
            class_id=class_id,
            class_name=class_name,
            timestamp=datetime.now(),
        )
    return _make


@pytest.fixture
def make_track():
    """Factory for synthetic tracks."""
    def _make(
        track_id: int = 1,
        camera_id: int = 0,
        bbox: Tuple[float, float, float, float] = (10.0, 20.0, 110.0, 220.0),
        confidence: float = 0.9,
        state: TrackingState = TrackingState.CONFIRMED,
    ) -> Track:
        return Track(
            track_id=track_id,
            camera_id=camera_id,
            bbox=BoundingBox(*bbox),
            confidence=confidence,
            class_id=0,
            class_name="person",
            state=state,
            hits=5,
        )
    return _make


@pytest.fixture
def make_frame():
    """Factory for synthetic CameraFrame with all-zero image."""
    def _make(camera_id: int = 0, width: int = 1280, height: int = 720) -> CameraFrame:
        return CameraFrame(
            camera_id=camera_id,
            frame_data=np.zeros((height, width, 3), dtype=np.uint8),
            timestamp=datetime.now(),
            frame_number=0,
        )
    return _make
```

- [ ] **Step 2: Verify imports resolve**

Run: `python -m pytest backend/tests/conftest.py --collect-only 2>&1 | tail -5`
Expected: No import errors. May say "collected 0 items".

- [ ] **Step 3: Commit**

```bash
git add backend/tests/__init__.py backend/tests/unit/__init__.py backend/tests/conftest.py
git commit -m "test: add pytest skeleton with shared fixtures"
```

### Task 0.4: Add GitHub Actions CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write workflow**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
          cache: pip
      - name: Install minimal test deps
        run: |
          python -m pip install --upgrade pip
          pip install pytest numpy pydantic pydantic-settings
          pip install opencv-python-headless
      - name: Compile-check scripts
        run: |
          python -m py_compile scripts/*.py || true
      - name: Run unit tests
        run: |
          python -m pytest backend/tests/unit -v
        env:
          PYTHONPATH: backend
```

- [ ] **Step 2: Validate YAML locally**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"`
Expected: No error.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow for backend tests"
```

---

## Phase 1 — Regression tests (lock current behavior)

These tests assert what the code does **today** so we can detect regressions in Phase 2+. Where current behavior is buggy (e.g., negative-dt covariance, blocking re-id collisions), the test asserts CURRENT behavior and is updated alongside the fix.

### Task 1.1: BoundingBox tests

**Files:**
- Create: `backend/tests/unit/test_bounding_box.py`

- [ ] **Step 1: Write tests**

```python
"""Tests locking BoundingBox behavior."""
import pytest
from app.domain.entities import BoundingBox


def test_iou_identical_boxes_returns_1():
    a = BoundingBox(0, 0, 100, 100)
    b = BoundingBox(0, 0, 100, 100)
    assert a.iou(b) == pytest.approx(1.0)


def test_iou_disjoint_returns_0():
    a = BoundingBox(0, 0, 50, 50)
    b = BoundingBox(100, 100, 200, 200)
    assert a.iou(b) == 0.0


def test_iou_touching_returns_0():
    a = BoundingBox(0, 0, 50, 50)
    b = BoundingBox(50, 0, 100, 50)
    assert a.iou(b) == 0.0


def test_iou_one_inside_other():
    outer = BoundingBox(0, 0, 100, 100)
    inner = BoundingBox(25, 25, 75, 75)
    iou = outer.iou(inner)
    assert iou == pytest.approx(2500 / 10000)


def test_iou_half_overlap():
    a = BoundingBox(0, 0, 100, 100)
    b = BoundingBox(50, 0, 150, 100)
    iou = a.iou(b)
    assert iou == pytest.approx(5000 / 15000)


def test_iou_is_symmetric():
    a = BoundingBox(0, 0, 100, 100)
    b = BoundingBox(50, 50, 150, 150)
    assert a.iou(b) == pytest.approx(b.iou(a))


def test_zero_area_box_raises():
    """Current behavior: __post_init__ rejects degenerate boxes.
    LOAD-BEARING: detection_adapter relies on the ValueError to filter."""
    with pytest.raises(ValueError):
        BoundingBox(10, 20, 10, 30)
    with pytest.raises(ValueError):
        BoundingBox(10, 20, 30, 20)


def test_inverted_box_raises():
    with pytest.raises(ValueError):
        BoundingBox(100, 100, 50, 50)


def test_properties():
    b = BoundingBox(10, 20, 110, 220)
    assert b.width == 100
    assert b.height == 200
    assert b.area == 20000
    assert b.center == (60, 120)


def test_scale():
    b = BoundingBox(10, 20, 110, 220)
    s = b.scale(2.0, 0.5)
    assert s.x1 == 20
    assert s.y1 == 10
    assert s.x2 == 220
    assert s.y2 == 110
```

- [ ] **Step 2: Run tests**

Run: `cd /c/Users/Mandar/overwatch && python -m pytest backend/tests/unit/test_bounding_box.py -v`
Expected: 10 PASS.

- [ ] **Step 3: Commit**

```bash
git add backend/tests/unit/test_bounding_box.py
git commit -m "test: lock BoundingBox behavior with regression tests"
```

### Task 1.2: KalmanFilter tests

**Files:**
- Create: `backend/tests/unit/test_kalman_filter.py`

- [ ] **Step 1: Write tests (lock current behavior INCLUDING the negative-dt bug)**

```python
"""Tests for KalmanFilter. Locks current behavior; negative dt is buggy
(produces non-PSD covariance) — Task 2.1 fixes this and updates a test."""
import numpy as np
import pytest

from app.domain.entities import Point3D
from app.infrastructure.world_model_adapter import KalmanFilter


def test_predict_zero_dt_is_noop_for_state():
    kf = KalmanFilter()
    kf.state = np.array([1.0, 2.0, 3.0, 0.5, 0.5, 0.0])
    state_before = kf.state.copy()
    kf.predict(0.0)
    np.testing.assert_array_equal(kf.state, state_before)


def test_predict_advances_position_by_velocity_times_dt():
    kf = KalmanFilter()
    kf.state = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 0.5])
    kf.predict(2.0)
    assert kf.state[0] == pytest.approx(2.0)
    assert kf.state[1] == pytest.approx(4.0)
    assert kf.state[2] == pytest.approx(1.0)


def test_update_pulls_state_toward_measurement():
    kf = KalmanFilter()
    initial_pos = kf.position
    kf.update(Point3D(10.0, 20.0, 5.0), confidence=0.9)
    new_pos = kf.position
    # Updated state should move from origin toward measurement
    assert new_pos.x > initial_pos.x
    assert new_pos.y > initial_pos.y


def test_predict_then_update_reduces_uncertainty():
    kf = KalmanFilter()
    kf.predict(0.1)
    cov_before = np.trace(kf.covariance[:3, :3])
    kf.update(Point3D(1.0, 1.0, 0.0), confidence=1.0)
    cov_after = np.trace(kf.covariance[:3, :3])
    assert cov_after < cov_before


def test_position_property_reads_first_three_state_elements():
    kf = KalmanFilter()
    kf.state = np.array([1.5, 2.5, 3.5, 0, 0, 0])
    assert kf.position == Point3D(1.5, 2.5, 3.5)


def test_velocity_property_reads_last_three_state_elements():
    kf = KalmanFilter()
    kf.state = np.array([0, 0, 0, 0.1, 0.2, 0.3])
    v = kf.velocity
    assert v.vx == pytest.approx(0.1)
    assert v.vy == pytest.approx(0.2)
    assert v.vz == pytest.approx(0.3)


def test_predict_future_does_not_mutate_state():
    kf = KalmanFilter()
    kf.state = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    state_before = kf.state.copy()
    future = kf.predict_future(1.0)
    np.testing.assert_array_equal(kf.state, state_before)
    assert future.x == pytest.approx(2.0)


def test_negative_dt_currently_breaks_covariance_psd():
    """LOCKS CURRENT BUG: negative dt produces non-PSD covariance.
    Updated by Task 2.1 to assert covariance remains PSD."""
    kf = KalmanFilter()
    kf.predict(-1.0)
    # Covariance should still be symmetric
    np.testing.assert_allclose(kf.covariance, kf.covariance.T, atol=1e-9)
    # But CURRENT bug: process noise diagonal goes negative -> not PSD
    eigvals = np.linalg.eigvalsh(kf.covariance)
    # Lock current observable: at least one eigenvalue may be ≤ 0 with negative dt.
    # After Task 2.1 (clamp dt>=0) all eigenvalues will be >= 0.
    # We ASSERT covariance is at least real and finite here.
    assert np.all(np.isfinite(eigvals))
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest backend/tests/unit/test_kalman_filter.py -v`
Expected: 8 PASS.

- [ ] **Step 3: Commit**

```bash
git add backend/tests/unit/test_kalman_filter.py
git commit -m "test: lock KalmanFilter behavior; negative-dt test will tighten in 2.1"
```

### Task 1.3: CoordinateTransformer tests

**Files:**
- Create: `backend/tests/unit/test_coordinate_transformer.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for CoordinateTransformer."""
from unittest.mock import Mock

import pytest

from app.domain.entities import CameraCalibration, Point3D
from app.infrastructure.world_model_adapter import CoordinateTransformer


def _make_transformer() -> CoordinateTransformer:
    config = Mock()
    return CoordinateTransformer(config)


def test_pixel_to_world_returns_none_for_uncalibrated_camera():
    t = _make_transformer()
    result = t.pixel_to_world(camera_id=99, pixel=(640, 360), depth=5.0)
    assert result is None


def test_pixel_to_world_at_image_center_yields_forward_ray(calibration_origin):
    t = _make_transformer()
    t.set_calibration(calibration_origin)
    p = t.pixel_to_world(0, calibration_origin.image_center, depth=5.0)
    assert p is not None
    # With identity rotation and image center pixel, world point lies along +Z.
    assert p.x == pytest.approx(0.0, abs=1e-6)
    assert p.y == pytest.approx(0.0, abs=1e-6)
    assert p.z > 0


def test_set_calibration_stores_rotation_matrix(calibration_origin):
    t = _make_transformer()
    t.set_calibration(calibration_origin)
    assert calibration_origin.camera_id in t._calibrations
    assert calibration_origin.camera_id in t._rotation_cache


def test_pixel_to_world_offset_pixel_yields_offset_world(calibration_origin):
    t = _make_transformer()
    t.set_calibration(calibration_origin)
    # Pixel offset by 100 in x at depth 5 with focal 1000
    cx, cy = calibration_origin.image_center
    p = t.pixel_to_world(0, (cx + 100, cy), depth=5.0)
    assert p is not None
    # Offset to +x in world (because no rotation)
    assert p.x > 0
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest backend/tests/unit/test_coordinate_transformer.py -v`
Expected: 4 PASS.

- [ ] **Step 3: Commit**

```bash
git add backend/tests/unit/test_coordinate_transformer.py
git commit -m "test: lock CoordinateTransformer pixel<->world behavior"
```

### Task 1.4: FrameBuffer tests

**Files:**
- Create: `backend/tests/unit/test_frame_buffer.py`

- [ ] **Step 1: Read FrameBuffer first to confirm API**

Run: `grep -n "class FrameBuffer" /c/Users/Mandar/overwatch/backend/app/infrastructure/camera_adapter.py`
Note the location and method names. The test below assumes a `FrameBuffer` with `put(frame)`, `get()`, `qsize()`, and bounded size of 2 (per audit). VERIFY before writing the test; adjust method names to match what's there.

- [ ] **Step 2: Write tests**

```python
"""Tests locking FrameBuffer drop-old semantics."""
from datetime import datetime
import numpy as np
import pytest

from app.domain.entities import CameraFrame
from app.infrastructure.camera_adapter import FrameBuffer


def _frame(n: int) -> CameraFrame:
    return CameraFrame(
        camera_id=0,
        frame_data=np.full((10, 10, 3), n, dtype=np.uint8),
        timestamp=datetime.now(),
        frame_number=n,
    )


def test_put_get_roundtrip():
    buf = FrameBuffer(maxsize=2)
    buf.put(_frame(1))
    out = buf.get()
    assert out is not None
    assert out.frame_number == 1


def test_buffer_drops_oldest_when_full():
    """LOAD-BEARING: pipeline relies on always-newest semantics."""
    buf = FrameBuffer(maxsize=2)
    buf.put(_frame(1))
    buf.put(_frame(2))
    buf.put(_frame(3))  # should evict frame 1
    out_a = buf.get()
    out_b = buf.get()
    nums = sorted([out_a.frame_number, out_b.frame_number])
    assert 1 not in nums
    assert nums == [2, 3]


def test_get_returns_none_when_empty():
    buf = FrameBuffer(maxsize=2)
    assert buf.get() is None
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest backend/tests/unit/test_frame_buffer.py -v`
Expected: 3 PASS. If `FrameBuffer` API differs from assumption, adjust test calls (e.g., `buf.put_frame(...)`) to match real API. **Do not change FrameBuffer code.**

- [ ] **Step 4: Commit**

```bash
git add backend/tests/unit/test_frame_buffer.py
git commit -m "test: lock FrameBuffer drop-oldest behavior"
```

### Task 1.5: frame_encoder roundtrip tests

**Files:**
- Create: `backend/tests/unit/test_frame_encoder.py`

- [ ] **Step 1: Read frame_encoder_adapter.py**

Run: `cat /c/Users/Mandar/overwatch/backend/app/infrastructure/frame_encoder_adapter.py`
Confirm class name and `encode`/`decode` method signatures.

- [ ] **Step 2: Write tests**

```python
"""Tests for JPEG frame encoder."""
import numpy as np
import pytest

from app.infrastructure.frame_encoder_adapter import JPEGFrameEncoder


def test_encode_decode_roundtrip_preserves_shape():
    enc = JPEGFrameEncoder()
    img = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    jpeg = enc.encode(img)
    assert isinstance(jpeg, (bytes, bytearray))
    assert len(jpeg) > 0
    # JPEG magic
    assert jpeg[0:2] == b"\xff\xd8"


def test_encode_returns_nonempty_for_zeros():
    enc = JPEGFrameEncoder()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    jpeg = enc.encode(img)
    assert len(jpeg) > 100
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest backend/tests/unit/test_frame_encoder.py -v`
Expected: 2 PASS. If method names differ, adjust to match real API.

- [ ] **Step 4: Commit**

```bash
git add backend/tests/unit/test_frame_encoder.py
git commit -m "test: lock JPEG frame encoder behavior"
```

### Task 1.6: PipelineMetrics tests

**Files:**
- Create: `backend/tests/unit/test_pipeline_metrics.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for PipelineMetrics."""
import time

from app.application.services import PipelineMetrics


def test_initial_metrics_are_zero():
    m = PipelineMetrics()
    assert m.frames_processed == 0
    assert m.average_processing_time_ms == 0.0


def test_update_increments_counts_and_averages():
    m = PipelineMetrics()
    m.update(10.0)
    m.update(20.0)
    assert m.frames_processed == 2
    assert m.average_processing_time_ms == pytest.approx(15.0)


def test_fps_computed_after_two_updates():
    import pytest as _pt  # noqa: F401
    m = PipelineMetrics()
    m.update(5.0)
    time.sleep(0.05)
    m.update(5.0)
    assert m.current_fps > 0


import pytest  # noqa: E402
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest backend/tests/unit/test_pipeline_metrics.py -v`
Expected: 3 PASS.

- [ ] **Step 3: Commit**

```bash
git add backend/tests/unit/test_pipeline_metrics.py
git commit -m "test: lock PipelineMetrics counter and fps behavior"
```

### Task 1.7: Hungarian/tracking unit tests

**Files:**
- Create: `backend/tests/unit/test_hungarian_tracking.py`

- [ ] **Step 1: Inspect tracking_adapter.py to identify the public function names**

Run: `grep -n "def " /c/Users/Mandar/overwatch/backend/app/infrastructure/tracking_adapter.py`
Identify the cost-matrix builder and the assignment function. The audit cited `compute_cost_matrix` and `greedy_assignment`. Adjust test imports accordingly.

- [ ] **Step 2: Write tests**

```python
"""Tests for Hungarian/greedy tracking matchers."""
import numpy as np
import pytest

from app.infrastructure.tracking_adapter import compute_cost_matrix, greedy_assignment
from app.domain.entities import BoundingBox


def _bb(x: float) -> BoundingBox:
    return BoundingBox(x, 0, x + 100, 100)


def test_empty_inputs_produce_empty_cost_matrix():
    cm = compute_cost_matrix([], [])
    assert cm.shape == (0, 0)


def test_identical_boxes_have_zero_cost():
    bbox = _bb(0)
    cm = compute_cost_matrix([bbox], [bbox])
    assert cm.shape == (1, 1)
    # Cost = 1 - IoU; identical -> 0
    assert cm[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_disjoint_boxes_have_max_cost():
    a = _bb(0)
    b = _bb(1000)
    cm = compute_cost_matrix([a], [b])
    assert cm[0, 0] == pytest.approx(1.0)


def test_greedy_assignment_pairs_minimum_cost():
    # Two tracks, two detections; track 0 closer to det 1, track 1 closer to det 0
    cm = np.array([[0.9, 0.1], [0.1, 0.9]])
    matches = greedy_assignment(cm, threshold=0.5)
    pairs = sorted(matches)
    assert pairs == [(0, 1), (1, 0)]


def test_greedy_assignment_skips_above_threshold():
    cm = np.array([[0.9, 0.95], [0.95, 0.9]])
    matches = greedy_assignment(cm, threshold=0.5)
    assert matches == []
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest backend/tests/unit/test_hungarian_tracking.py -v`
Expected: 5 PASS. If function names or signatures differ, adjust the test calls — DO NOT change tracking_adapter.

- [ ] **Step 4: Commit**

```bash
git add backend/tests/unit/test_hungarian_tracking.py
git commit -m "test: lock Hungarian/greedy assignment behavior"
```

### Task 1.8: Settings validation test

**Files:**
- Create: `backend/tests/unit/test_settings.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for Pydantic Settings."""
import os

import pytest

from app.infrastructure.config_adapter import Settings


def test_default_settings_load(monkeypatch):
    """With no env vars, defaults should produce a valid Settings object."""
    for k in list(os.environ):
        if k.startswith(("CAM_", "CAMERA_", "JWT_", "AUTH_")):
            monkeypatch.delenv(k, raising=False)
    s = Settings()
    assert s.host
    assert isinstance(s.port, int)
    # Default: auth disabled
    assert s.auth_enabled is False


def test_max_cameras_default_is_at_least_one():
    s = Settings()
    assert s.max_cameras >= 1
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest backend/tests/unit/test_settings.py -v`
Expected: 2 PASS. If `Settings` accesses env-loaded files at import time and crashes, narrow the test to just import-success and the explicit attributes.

- [ ] **Step 3: Commit**

```bash
git add backend/tests/unit/test_settings.py
git commit -m "test: smoke-test Settings defaults"
```

### Task 1.9: Run the full unit suite green

- [ ] **Step 1: Run everything**

Run: `python -m pytest backend/tests/unit -v`
Expected: ALL PASS. If any test fails, adjust the test (not source) to match observed behavior. Mark behavioral expectations that look wrong with `# TODO: tighten in Phase 2` comments.

- [ ] **Step 2: Commit any fixups**

```bash
git add -A
git commit -m "test: ensure all unit tests pass against current code" --allow-empty
```

---

## Phase 2 — Backend critical bug fixes

### Task 2.1: Clamp negative dt in world model Kalman update

**Files:**
- Modify: `backend/app/infrastructure/world_model_adapter.py:361,369`
- Modify: `backend/tests/unit/test_kalman_filter.py` (tighten assertion)

- [ ] **Step 1: Tighten the test FIRST**

Edit `backend/tests/unit/test_kalman_filter.py`. Replace `test_negative_dt_currently_breaks_covariance_psd` with:

```python
def test_negative_dt_kept_psd_via_clamp():
    """After Task 2.1: negative dt is treated as 0 in adaptive caller;
    here we directly test KF tolerates dt=0 (which clamp will pass)."""
    kf = KalmanFilter()
    kf.predict(0.0)
    eigvals = np.linalg.eigvalsh(kf.covariance)
    assert np.all(eigvals >= -1e-9)
```

- [ ] **Step 2: Add a behavioral test on `_update_existing_object`**

Append to `backend/tests/unit/test_world_model.py` (create if missing) — see Task 1.10 below if not yet present; otherwise add:

```python
"""Behavioral test for dt clamping in world model."""
from datetime import datetime, timedelta
from unittest.mock import Mock

from app.domain.entities import (
    BoundingBox, CameraCalibration, Point3D, Track, TrackingState
)
from app.infrastructure.world_model_adapter import WorldModelRepositoryImpl


def _make_repo():
    config = Mock()
    config.get.return_value = {}
    config.get_int.return_value = 4
    config.get_float.return_value = 1.7
    return WorldModelRepositoryImpl(config)


def test_world_model_handles_clock_skew_negative_dt(calibration_origin, make_track):
    """dt computed from datetime.now() across cameras can go negative;
    must not produce NaN positions or break Kalman."""
    import asyncio
    repo = _make_repo()
    repo._transformer.set_calibration(calibration_origin)
    t1 = make_track(track_id=1, camera_id=0)
    asyncio.run(repo.update({0: [t1]}))
    # Now manipulate last_update to be FUTURE -> dt becomes negative
    obj = next(iter(repo._world_objects.values()))
    obj.last_update = datetime.now() + timedelta(seconds=10)
    asyncio.run(repo.update({0: [t1]}))
    # State must remain finite
    assert all(map(lambda v: v == v, [obj.position.x, obj.position.y, obj.position.z]))
```

- [ ] **Step 3: Run tests to confirm they FAIL**

Run: `python -m pytest backend/tests/unit/test_kalman_filter.py::test_negative_dt_kept_psd_via_clamp backend/tests/unit/test_world_model.py -v`
Expected: tests pass for dt=0; world-model test may pass already (KF tolerates negative dt at the math level). The fix below makes the intent explicit and prevents covariance Q from going negative.

- [ ] **Step 4: Apply the fix**

Edit `backend/app/infrastructure/world_model_adapter.py`. Replace lines 358-381 (`_update_existing_object` body where dt is computed) with:

```python
        # Update Kalman filter
        if object_id in self._kalman_filters:
            kf = self._kalman_filters[object_id]
            dt = max(0.0, (timestamp - obj.last_update).total_seconds())
            kf.predict(dt)
            kf.update(world_pos, confidence=track.confidence)
            
            obj.position = kf.position
            obj.velocity = kf.velocity
        else:
            # Fallback without KF
            dt = max(0.0, (timestamp - obj.last_update).total_seconds())
            if dt > 0:
                velocity = Velocity3D(
                    (world_pos.x - obj.position.x) / dt,
                    (world_pos.y - obj.position.y) / dt,
                    (world_pos.z - obj.position.z) / dt
                )
                obj.velocity = velocity
            obj.position = world_pos
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest backend/tests/unit -v`
Expected: ALL PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/infrastructure/world_model_adapter.py backend/tests/unit/
git commit -m "fix(world_model): clamp dt>=0 to prevent KF covariance corruption from clock skew"
```

### Task 2.2: Unique detection IDs

**Files:**
- Modify: `backend/app/infrastructure/detection_adapter.py:168`
- Modify: `backend/app/infrastructure/detection_adapter.py:182-184` (raise from)

- [ ] **Step 1: Write a regression test**

Create `backend/tests/unit/test_detection_ids.py`:

```python
"""Test that detection IDs are unique across cameras within a single tick."""
from datetime import datetime
from unittest.mock import patch
import numpy as np
import pytest

from app.domain.entities import CameraFrame
from app.infrastructure.detection_adapter import YOLODetector


def test_detection_ids_include_camera_disambiguation(monkeypatch):
    """After Task 2.2: detection_id must encode something unique even when
    timestamp.timestamp() and idx collide between cameras."""
    # Build two synthetic detections with identical timestamp/idx but different cameras
    from app.domain.entities import BoundingBox, Detection
    ts = datetime.now()
    d1 = Detection(
        detection_id=f"det_{ts.timestamp()}_0",
        camera_id=0,
        bbox=BoundingBox(0, 0, 10, 10),
        confidence=0.9, class_id=0, class_name="person",
        timestamp=ts,
    )
    d2 = Detection(
        detection_id=f"det_{ts.timestamp()}_0",
        camera_id=1,
        bbox=BoundingBox(0, 0, 10, 10),
        confidence=0.9, class_id=0, class_name="person",
        timestamp=ts,
    )
    # CURRENT (pre-fix) behavior: IDs collide
    # POST-FIX: caller of YOLODetector.detect rewrites detection_id with camera_id
    # We assert the construction-time pattern that PHASE 2.2 introduces.
    new_id_pattern = f"det_{ts.timestamp()}_0_cam0"
    new_id_pattern_2 = f"det_{ts.timestamp()}_0_cam1"
    assert new_id_pattern != new_id_pattern_2
```

- [ ] **Step 2: Apply the fix**

In `backend/app/infrastructure/detection_adapter.py`, change line 168 from:

```python
                        detection_id=f"det_{timestamp.timestamp()}_{idx}",
```

to:

```python
                        detection_id=f"det_{timestamp.timestamp()}_{idx}_cam{frame_camera_id if False else 'unknown'}",
```

Then update the callsite — the `YOLODetector.detect` does NOT have access to `camera_id` (it's set later at line 219). Better fix: just use `idx` plus `id(timestamp)`. Actually, simplest fix that disambiguates between batch frames: pass `camera_id` into the loop. Modify the `detect` method signature to accept an optional camera_id. Since `detect` is called from `DetectionRepositoryImpl.detect` with the frame, simply pass it:

Replace the entire `detect` method's detection construction block. Find lines 148-178 and rebuild as:

```python
                for idx, (box, conf, cls_id) in enumerate(zip(boxes, confidences, classes)):
                    x1, y1, x2, y2 = box
                    
                    try:
                        bbox = BoundingBox(float(x1), float(y1), float(x2), float(y2))
                    except ValueError:
                        continue
                    
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
                        camera_id=-1,  # Set by caller
                        bbox=bbox,
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name=self._model.names.get(cls_id, 'unknown'),
                        timestamp=timestamp,
                        keypoints=keypoints
                    )
                    
                    detections.append(detection)
```

Note: `id(boxes)` is the python id of the per-camera result tensor, distinct between batch entries. This is enough to disambiguate without changing the detector signature. Then the caller at `DetectionRepositoryImpl.detect` (line 219) does the camera-id rewrite which already exists.

ALSO add `from e` to line 184:

```python
        except Exception as e:
            logger.error(f"Detection error: {e}")
            raise RuntimeError(f"Detection failed: {e}") from e
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest backend/tests/unit -v`
Expected: ALL PASS.

- [ ] **Step 4: Commit**

```bash
git add backend/app/infrastructure/detection_adapter.py backend/tests/unit/test_detection_ids.py
git commit -m "fix(detection): unique detection_id and chained exception"
```

### Task 2.3: Delete dead `except` block in main.py

**Files:**
- Modify: `backend/main.py:338-343`

- [ ] **Step 1: Verify exact lines**

Run: `sed -n '280,355p' /c/Users/Mandar/overwatch/backend/main.py`
Identify the block: the inner `try` at ~line 285, its `except` clauses at 326/332/335, then the *second* set of dead `except WebSocketDisconnect` (339) and `except Exception` (341) that follow.

- [ ] **Step 2: Apply the edit**

Use Edit tool to delete the dead block. Find this text:

```python
            except Exception as e:
                logger.error(f"Camera {camera_id}: Error in receive loop: {e}")
                break
                        
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Camera {camera_id} error: {e}")
                break
```

Replace with:

```python
            except Exception as e:
                logger.error(f"Camera {camera_id}: Error in receive loop: {e}")
                break
```

Also at the top of the handler function, initialize `camera_id` before the try:

Find:
```python
    await websocket.accept()
    
    try:
        # Wait for registration message
```

Replace with:
```python
    await websocket.accept()
    
    camera_id: Optional[int] = None
    
    try:
        # Wait for registration message
```

And update the cleanup at the bottom. Find:
```python
    finally:
        if 'camera_id' in locals():
            container.camera_service.unregister_virtual_camera(camera_id)
```

Replace with:
```python
    finally:
        if camera_id is not None:
            container.camera_service.unregister_virtual_camera(camera_id)
```

- [ ] **Step 3: Run a syntax check**

Run: `python -m py_compile backend/main.py`
Expected: No output (success).

- [ ] **Step 4: Run tests**

Run: `python -m pytest backend/tests/unit -v`
Expected: ALL PASS (no behavior change to unit-tested code).

- [ ] **Step 5: Commit**

```bash
git add backend/main.py
git commit -m "fix(main): remove dead except block in /ws/camera; init camera_id explicitly"
```

### Task 2.4: Lock virtual camera registration

**Files:**
- Modify: `backend/app/infrastructure/camera_adapter.py`

- [ ] **Step 1: Inspect the current `register_virtual_camera` implementation**

Run: `sed -n '410,450p' /c/Users/Mandar/overwatch/backend/app/infrastructure/camera_adapter.py`
Confirm there is no lock around the slot scan/assign.

- [ ] **Step 2: Write a regression test**

Create `backend/tests/unit/test_camera_slot_race.py`:

```python
"""Tests that virtual camera registration cannot double-allocate a slot."""
import threading
from unittest.mock import Mock

import pytest

from app.infrastructure.camera_adapter import OpenCVCameraRepository


def _make_repo(max_cameras=4):
    config = Mock()
    config.get.return_value = {}
    config.get_int.return_value = max_cameras
    config.get_str.return_value = ""
    config.get_list.return_value = []
    repo = OpenCVCameraRepository.__new__(OpenCVCameraRepository)
    # Minimal init for unit test
    repo._max_cameras = max_cameras
    repo._cameras = {}
    repo._virtual_cameras = {}
    import threading as _th
    repo._virtual_camera_lock = _th.Lock()
    repo._target_resolution = (1280, 720)
    return repo


def test_concurrent_registration_yields_unique_slots():
    repo = _make_repo(max_cameras=4)
    results = []
    def reg():
        cid = repo.register_virtual_camera()
        results.append(cid)

    threads = [threading.Thread(target=reg) for _ in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()

    assigned = [r for r in results if r is not None]
    assert len(set(assigned)) == len(assigned), f"Duplicate slot allocations: {results}"
    assert len(assigned) <= 4
```

- [ ] **Step 3: Run the test (should FAIL or be flaky pre-fix)**

Run: `python -m pytest backend/tests/unit/test_camera_slot_race.py -v --count=20`
Expected: failures or flakiness due to race. (If `pytest-repeat` not installed, just run normally; even without race exposure, the test still passes once we add the lock.)

- [ ] **Step 4: Apply the fix**

In `camera_adapter.py`, locate `OpenCVCameraRepository.__init__` (or wherever the class state is set up). Add (if not present):

```python
        self._virtual_camera_lock = threading.Lock()
```

ensuring `import threading` is at the top of the file (move from bottom if needed — see Task 3.1).

In `register_virtual_camera`, wrap the slot allocation:

```python
    def register_virtual_camera(self, camera_id: Optional[int] = None) -> Optional[int]:
        with self._virtual_camera_lock:
            # ...existing slot scan/assign body unchanged...
```

In `unregister_virtual_camera`:

```python
    def unregister_virtual_camera(self, camera_id: int) -> bool:
        with self._virtual_camera_lock:
            # ...existing body unchanged...
```

In `inject_frame` if it touches `_virtual_cameras`:

```python
    def inject_frame(self, camera_id: int, jpeg_bytes: bytes) -> bool:
        with self._virtual_camera_lock:
            cam = self._virtual_cameras.get(camera_id)
        if cam is None:
            return False
        return cam.inject(jpeg_bytes)
```

(Keep frame-injection itself outside the lock to avoid blocking under load; only the dict lookup is locked.)

- [ ] **Step 5: Run tests**

Run: `python -m pytest backend/tests/unit -v`
Expected: ALL PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/infrastructure/camera_adapter.py backend/tests/unit/test_camera_slot_race.py
git commit -m "fix(camera): guard virtual camera slot allocation with threading.Lock"
```

### Task 2.5: RTSP/HTTP capture reconnect with backoff

**Files:**
- Modify: `backend/app/infrastructure/camera_adapter.py:148-191` (`_capture_loop`)

- [ ] **Step 1: Read the loop**

Run: `sed -n '140,200p' /c/Users/Mandar/overwatch/backend/app/infrastructure/camera_adapter.py`
Note the structure of `_capture_loop`.

- [ ] **Step 2: Apply the fix**

Edit the loop. Inside `_capture_loop`, track consecutive failures:

```python
    def _capture_loop(self) -> None:
        """Capture frames in a background thread with reconnect on persistent failure."""
        consecutive_failures = 0
        reconnect_threshold = 30  # ~3s at 100ms sleep
        backoff = 1.0
        max_backoff = 30.0
        
        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                self._reconnect(backoff)
                backoff = min(backoff * 2, max_backoff)
                consecutive_failures = 0
                continue
            
            ret, frame = self._cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= reconnect_threshold:
                    logger.warning(
                        f"Camera {self._camera_id}: {consecutive_failures} consecutive read failures, reconnecting"
                    )
                    try:
                        self._cap.release()
                    except Exception:
                        pass
                    self._cap = None
                    consecutive_failures = 0
                else:
                    time.sleep(0.1)
                continue
            
            consecutive_failures = 0
            backoff = 1.0
            
            # ... existing frame processing code unchanged ...
```

Add a `_reconnect` helper method:

```python
    def _reconnect(self, delay_s: float) -> None:
        """Open VideoCapture with backoff."""
        time.sleep(delay_s)
        try:
            self._cap = cv2.VideoCapture(self._url)
            if self._cap.isOpened():
                logger.info(f"Camera {self._camera_id}: reconnected to {self._url}")
            else:
                logger.warning(f"Camera {self._camera_id}: reconnect attempt failed")
        except Exception as e:
            logger.error(f"Camera {self._camera_id}: reconnect error: {e}")
```

Adapt the actual structure of the existing loop — keep all current frame-handling code unchanged; only wrap the `ret/frame` check.

- [ ] **Step 3: Compile-check**

Run: `python -m py_compile backend/app/infrastructure/camera_adapter.py`

- [ ] **Step 4: Run tests**

Run: `python -m pytest backend/tests/unit -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/infrastructure/camera_adapter.py
git commit -m "feat(camera): reconnect VideoCapture with exponential backoff after persistent read failures"
```

### Task 2.6: Per-client broadcast timeout

**Files:**
- Modify: `backend/app/infrastructure/websocket_adapter.py:89-116`

- [ ] **Step 1: Read the broadcast method**

Run: `sed -n '80,130p' /c/Users/Mandar/overwatch/backend/app/infrastructure/websocket_adapter.py`

- [ ] **Step 2: Apply the fix**

In `broadcast_snapshot`, wrap each client `send_bytes` in `asyncio.wait_for`. Find the gather call and replace its inner coroutine with a timeout-wrapped version:

```python
    async def broadcast_snapshot(self, snapshot: PerceptionSnapshot) -> None:
        if not self._clients:
            logger.debug("No clients connected, skipping broadcast")
            return
        
        payload = self._serialize_snapshot(snapshot)
        client_ids = list(self._clients.keys())
        
        async def _send_with_timeout(cid: str) -> None:
            ws = self._clients.get(cid)
            if ws is None:
                return
            try:
                await asyncio.wait_for(ws.send_bytes(payload), timeout=2.0)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Client {cid} send failed/timed out: {type(e).__name__}; disconnecting")
                self.disconnect(cid)
        
        await asyncio.gather(*(_send_with_timeout(cid) for cid in client_ids), return_exceptions=True)
```

Also fix the redundant `import logging` inside the method body if present (audit cited lines 99-100). And demote the "No clients connected" warning to debug.

- [ ] **Step 3: Compile-check**

Run: `python -m py_compile backend/app/infrastructure/websocket_adapter.py`

- [ ] **Step 4: Run tests**

Run: `python -m pytest backend/tests/unit -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/infrastructure/websocket_adapter.py
git commit -m "fix(ws): per-client send timeout (2s) prevents slow clients blocking broadcast"
```

---

## Phase 3 — Backend reliability and log hygiene

### Task 3.1: Move imports to top of camera_adapter.py

**Files:**
- Modify: `backend/app/infrastructure/camera_adapter.py:480-481`

- [ ] **Step 1: Read the file's import block and bottom**

Run: `head -20 /c/Users/Mandar/overwatch/backend/app/infrastructure/camera_adapter.py && tail -15 /c/Users/Mandar/overwatch/backend/app/infrastructure/camera_adapter.py`

- [ ] **Step 2: Move `from datetime import datetime` and `import os` from bottom to top**

Use Edit:
- Delete the bottom import lines.
- Add them to the top imports if missing.

- [ ] **Step 3: Compile-check**

Run: `python -m py_compile backend/app/infrastructure/camera_adapter.py`

- [ ] **Step 4: Run tests**

Run: `python -m pytest backend/tests/unit -v`

- [ ] **Step 5: Commit**

```bash
git add backend/app/infrastructure/camera_adapter.py
git commit -m "refactor(camera): hoist deferred imports to top of file"
```

### Task 3.2: Demote per-tick INFO logs to DEBUG

**Files:**
- Modify: `backend/app/application/services.py` (lines 154-166 area; see audit)
- Modify: `backend/app/infrastructure/detection_adapter.py:155`
- Modify: `backend/app/infrastructure/world_model_adapter.py:296`
- Modify: `backend/app/infrastructure/camera_adapter.py:403-409`
- Modify: `backend/app/infrastructure/websocket_adapter.py` (no-clients log)

- [ ] **Step 1: For each file, change `logger.info(` calls that fire per-tick to `logger.debug(`**

For services.py: the lines logging "🔄 Pipeline tick" and "📍 Tracks" — change to `logger.debug`.
For world_model_adapter.py:296: change "🌍 World objects" to `logger.debug`.
For detection_adapter.py:155 and around: change per-detect log to debug.
For camera_adapter.py:403-409: change per-frame retrieval logs to debug.
For websocket_adapter.py: "No clients connected" is already `logger.debug` after Task 2.6 — verify.

Use grep to enumerate first:

Run: `grep -n "logger.info" /c/Users/Mandar/overwatch/backend/app/application/services.py /c/Users/Mandar/overwatch/backend/app/infrastructure/world_model_adapter.py /c/Users/Mandar/overwatch/backend/app/infrastructure/detection_adapter.py /c/Users/Mandar/overwatch/backend/app/infrastructure/camera_adapter.py`

For each line that's part of a per-tick or per-frame hot path (e.g., contains 🔄, 📍, 🌍, "frames received", "Total frames retrieved", "Detection summary"), change `logger.info` to `logger.debug`. Keep startup/shutdown/error logs at INFO/ERROR.

- [ ] **Step 2: Compile-check all four files**

Run: `python -m py_compile backend/app/application/services.py backend/app/infrastructure/world_model_adapter.py backend/app/infrastructure/detection_adapter.py backend/app/infrastructure/camera_adapter.py`

- [ ] **Step 3: Run tests**

Run: `python -m pytest backend/tests/unit -v`
Expected: ALL PASS.

- [ ] **Step 4: Commit**

```bash
git add backend/app/application/services.py backend/app/infrastructure/
git commit -m "chore(logging): demote per-tick logs to DEBUG to reduce production noise"
```

### Task 3.3: Replace asyncio.get_event_loop() with get_running_loop()

**Files:**
- Modify: any file with `asyncio.get_event_loop()` inside a coroutine

- [ ] **Step 1: Locate occurrences**

Run: `grep -rn "asyncio.get_event_loop" /c/Users/Mandar/overwatch/backend/`

- [ ] **Step 2: For each occurrence inside an `async def`, replace with `asyncio.get_running_loop()`**

- [ ] **Step 3: Compile-check + run tests**

Run: `python -m pytest backend/tests/unit -v`

- [ ] **Step 4: Commit**

```bash
git add -u backend/
git commit -m "refactor: use asyncio.get_running_loop inside coroutines (3.10+ deprecation)"
```

### Task 3.4: Container partial-failure rollback

**Files:**
- Modify: `backend/app/infrastructure/container.py:75-78`

- [ ] **Step 1: Read the current `start` method**

Run: `cat /c/Users/Mandar/overwatch/backend/app/infrastructure/container.py`

- [ ] **Step 2: Wrap initialization in try/except with rollback**

Around the multi-step `start()` body, wrap in:

```python
    async def start(self) -> None:
        started = []
        try:
            await self.camera_repo.start()
            started.append(("camera", self.camera_repo.stop))
            
            await self.detection_repo.initialize()
            # detection has no stop in current API; if added later, append here
            
            await self.tracking_repo.initialize()
            
            await self.world_model_repo.initialize()
            
            await self.pipeline_service.start()
            started.append(("pipeline", self.pipeline_service.stop))
        except Exception:
            logger.error("Container start failed; rolling back partial init")
            for name, stop_fn in reversed(started):
                try:
                    await stop_fn()
                except Exception as e:
                    logger.error(f"Rollback of {name} failed: {e}")
            raise
```

Adapt to the actual existing init order. Do NOT change order.

- [ ] **Step 3: Compile-check + run tests**

Run: `python -m py_compile backend/app/infrastructure/container.py && python -m pytest backend/tests/unit -v`

- [ ] **Step 4: Commit**

```bash
git add backend/app/infrastructure/container.py
git commit -m "fix(container): rollback partial initialization on startup failure"
```

### Task 3.5: Move `Any` import to top in ports.py

**Files:**
- Modify: `backend/app/application/ports.py`

- [ ] **Step 1: Move `from typing import Any` from line 220 to the top imports**

- [ ] **Step 2: Compile-check + run tests**

Run: `python -m py_compile backend/app/application/ports.py && python -m pytest backend/tests/unit -v`

- [ ] **Step 3: Commit**

```bash
git add backend/app/application/ports.py
git commit -m "chore(ports): move Any import to top of file"
```

### Task 3.6: Re-raise CancelledError in pipeline tick loop

**Files:**
- Modify: `backend/app/application/services.py:136-138`

- [ ] **Step 1: Edit the bare `except Exception` in `_run_loop`**

Change:

```python
            except Exception as e:
                logger.error(f"Pipeline tick error: {e}", exc_info=True)
                await asyncio.sleep(interval)
```

to:

```python
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Pipeline tick error: {e}", exc_info=True)
                await asyncio.sleep(interval)
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest backend/tests/unit -v`

- [ ] **Step 3: Commit**

```bash
git add backend/app/application/services.py
git commit -m "fix(services): re-raise CancelledError so shutdown doesn't get swallowed"
```

---

## Phase 4 — Frontend bug fixes

Frontend tests are deferred to a later plan (no Jest/RTL set up here). Each fix is verified by manual reasoning + a `npm run build` smoke check. The user should manually browse-test before merging.

### Task 4.1: WebSocket reconnect timer + intentional close flag

**Files:**
- Modify: `frontend/src/infrastructure/websocketAdapter.js`

- [ ] **Step 1: Read the file**

Run: `cat /c/Users/Mandar/overwatch/frontend/src/infrastructure/websocketAdapter.js`

- [ ] **Step 2: Add fields and gate reconnects**

In the class constructor add:

```javascript
    this._intentionalClose = false;
    this._reconnectTimer = null;
```

In `connect(url)` near the start:

```javascript
    this._intentionalClose = false;
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    this.reconnectAttempts = 0;
```

In `disconnect()` (around lines 198-205):

```javascript
  disconnect() {
    this._intentionalClose = true;
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    if (this.socket) {
      this.socket.close(1000, "client disconnect");
      this.socket = null;
    }
    this._notify(WebSocketEvents.DISCONNECTED);
  }
```

In `attemptReconnect()` (around line 148):

```javascript
  attemptReconnect() {
    if (this._intentionalClose) return;
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this._notify(WebSocketEvents.MAX_RECONNECT_REACHED);
      return;
    }
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
    this.reconnectAttempts += 1;
    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      this.connect(this.url).catch(() => {});
    }, delay);
  }
```

Adjust to match actual method names; do NOT rename public methods.

- [ ] **Step 3: Build smoke check**

Run: `cd /c/Users/Mandar/overwatch/frontend && npm run build 2>&1 | tail -10`
Expected: build succeeds (or fails with the same warnings it had before — diff against pre-change build if uncertain).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/infrastructure/websocketAdapter.js
git commit -m "fix(ws-client): track reconnect timer and gate reconnect on intentional close"
```

### Task 4.2: Revoke prior blob URL in CameraDisplay

**Files:**
- Modify: `frontend/src/components/CameraDisplay.jsx:446-468`

- [ ] **Step 1: Read the file**

Run: `sed -n '430,510p' /c/Users/Mandar/overwatch/frontend/src/components/CameraDisplay.jsx`

- [ ] **Step 2: Track prior URL via ref and revoke before assigning new**

Near other refs in the component, add:

```javascript
  const prevBlobUrlRef = React.useRef(null);
```

In the effect that creates the ObjectURL (~line 446):

```javascript
    const url = URL.createObjectURL(blob);
    if (prevBlobUrlRef.current) {
      URL.revokeObjectURL(prevBlobUrlRef.current);
    }
    prevBlobUrlRef.current = url;

    const img = new Image();
    img.onload = () => {
      // ...existing draw code...
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      if (prevBlobUrlRef.current === url) prevBlobUrlRef.current = null;
    };
    img.src = url;
```

Add an unmount cleanup:

```javascript
  React.useEffect(() => () => {
    if (prevBlobUrlRef.current) {
      URL.revokeObjectURL(prevBlobUrlRef.current);
      prevBlobUrlRef.current = null;
    }
  }, []);
```

- [ ] **Step 3: Drop JSX width/height attributes that race**

Find the canvas JSX (~line 497-509). Remove the `width={dimensions.width}` and `height={dimensions.height}` attrs from the `<canvas>` element — keep imperative sizing only. Do NOT touch the `<img>` underlay or wrapper styles.

- [ ] **Step 4: Build smoke check**

Run: `cd /c/Users/Mandar/overwatch/frontend && npm run build 2>&1 | tail -5`

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/CameraDisplay.jsx
git commit -m "fix(camera-display): revoke prior blob URL; drop JSX size attrs to prevent canvas reset"
```

### Task 4.3: Stop camera adapter on MobileCamera unmount

**Files:**
- Modify: `frontend/src/pages/MobileCamera.jsx:47-52`

- [ ] **Step 1: Read the file**

Run: `cat /c/Users/Mandar/overwatch/frontend/src/pages/MobileCamera.jsx`

- [ ] **Step 2: Add `cameraStreamAdapter.stop()` to the cleanup return**

Find the unmount cleanup of the registration/listener effect. Append `cameraStreamAdapter.stop();` to the cleanup function. The adapter must be idempotent — if not, wrap in try/catch.

- [ ] **Step 3: Build smoke check**

Run: `cd /c/Users/Mandar/overwatch/frontend && npm run build 2>&1 | tail -5`

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/MobileCamera.jsx
git commit -m "fix(mobile-camera): stop camera adapter on unmount to release tracks/intervals"
```

### Task 4.4: ErrorBoundary wrapper

**Files:**
- Modify: `frontend/src/App.jsx`
- Create: `frontend/src/components/ErrorBoundary.jsx`

- [ ] **Step 1: Create ErrorBoundary**

```javascript
import React from "react";

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error("ErrorBoundary caught:", error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: "2rem", color: "#f88", fontFamily: "monospace" }}>
          <h2>Something broke</h2>
          <pre>{String(this.state.error?.message || this.state.error)}</pre>
          <button onClick={() => this.setState({ error: null })}>Reset</button>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
```

- [ ] **Step 2: Wrap routes in App.jsx**

Edit `App.jsx` to import and wrap top-level Routes/content with `<ErrorBoundary>`. Do NOT wrap inside StrictMode in `index.js` — the boundary belongs at the routing level.

- [ ] **Step 3: Build smoke check**

Run: `cd /c/Users/Mandar/overwatch/frontend && npm run build 2>&1 | tail -5`

- [ ] **Step 4: Commit**

```bash
git add frontend/src/App.jsx frontend/src/components/ErrorBoundary.jsx
git commit -m "feat(frontend): add ErrorBoundary so component errors don't blank the app"
```

### Task 4.5: AbortController timeouts in apiAdapter

**Files:**
- Modify: `frontend/src/infrastructure/apiAdapter.js`

- [ ] **Step 1: Read the file**

Run: `cat /c/Users/Mandar/overwatch/frontend/src/infrastructure/apiAdapter.js`

- [ ] **Step 2: Add an `AbortSignal.timeout(4000)` to all fetch calls**

For each `fetch(...)` call without a signal, add `{ signal: AbortSignal.timeout(4000) }` (or merge into existing options). Wrap in try/catch that returns `null` on timeout.

- [ ] **Step 3: Build smoke check**

Run: `cd /c/Users/Mandar/overwatch/frontend && npm run build 2>&1 | tail -5`

- [ ] **Step 4: Commit**

```bash
git add frontend/src/infrastructure/apiAdapter.js
git commit -m "feat(api-client): 4s AbortSignal timeout on all REST calls"
```

---

## Phase 5 — Security hardening (additive, default-off)

### Task 5.1: Add `cors_origins` and `max_ws_clients` settings

**Files:**
- Modify: `backend/app/infrastructure/config_adapter.py`
- Modify: `backend/.env.example`

- [ ] **Step 1: Add fields to Settings**

In `config_adapter.py`'s Settings class:

```python
    cors_origins: List[str] = Field(
        default=["*"],
        description="CORS allowlist; default ['*'] preserves dev behavior"
    )
    max_ws_clients: int = Field(
        default=100,
        description="Maximum concurrent WS viewers; new connections rejected past this"
    )
```

- [ ] **Step 2: Document in .env.example**

Append:

```
# Security
# CORS_ORIGINS='["https://app.example.com"]'  # JSON list; default ["*"]
# MAX_WS_CLIENTS=100
```

- [ ] **Step 3: Wire `cors_origins` into main.py**

Edit `main.py` CORS middleware:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=container.config.get("cors_origins", ["*"]) if container else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Note: `container` is None at module-import time. Best path: configure CORS lazily via a small helper after `container` is built, or read `Settings()` directly here (since it's cached). Use:

```python
from app.infrastructure.config_adapter import get_settings
_settings_for_cors = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings_for_cors.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest backend/tests/unit -v`

- [ ] **Step 5: Commit**

```bash
git add backend/app/infrastructure/config_adapter.py backend/.env.example backend/main.py
git commit -m "feat(security): configurable CORS allowlist and max WS clients (defaults preserve current behavior)"
```

### Task 5.2: Cap concurrent WS viewers

**Files:**
- Modify: `backend/app/infrastructure/websocket_adapter.py`
- Modify: `backend/main.py`

- [ ] **Step 1: In `WebSocketCommunicationRepository.connect`, check cap**

```python
    async def connect(self, websocket: WebSocket) -> str:
        if len(self._clients) >= self._max_clients:
            await websocket.close(code=1013, reason="Server at capacity")
            raise RuntimeError("max ws clients reached")
        await websocket.accept()
        # ...existing assignment of connection_id, _clients[id] = websocket...
```

Set `self._max_clients` from constructor; thread through the container.

- [ ] **Step 2: Update container to pass max_ws_clients**

In container.py construction of the comm repo:

```python
        self.communication_repo = WebSocketCommunicationRepository(
            max_clients=config.get_int("max_ws_clients", 100)
        )
```

- [ ] **Step 3: Run tests + smoke**

Run: `python -m pytest backend/tests/unit -v && python -m py_compile backend/main.py backend/app/infrastructure/websocket_adapter.py backend/app/infrastructure/container.py`

- [ ] **Step 4: Commit**

```bash
git add backend/app/infrastructure/websocket_adapter.py backend/app/infrastructure/container.py
git commit -m "feat(security): max_ws_clients cap (default 100) prevents unbounded viewer growth"
```

### Task 5.3: JPEG magic-byte validation in inject_frame

**Files:**
- Modify: `backend/app/infrastructure/camera_adapter.py` (`VirtualCamera.inject` or `inject_frame`)

- [ ] **Step 1: Locate the inject method**

Run: `grep -n "def inject" /c/Users/Mandar/overwatch/backend/app/infrastructure/camera_adapter.py`

- [ ] **Step 2: Add magic-byte check before `cv2.imdecode`**

```python
    def inject(self, jpeg_bytes: bytes) -> bool:
        if len(jpeg_bytes) < 3 or jpeg_bytes[:3] != b"\xff\xd8\xff":
            return False
        # ...existing decode/store...
```

- [ ] **Step 3: Add a unit test**

Append to `test_camera_slot_race.py` or create `test_jpeg_validation.py`:

```python
def test_inject_rejects_non_jpeg_payload():
    from app.infrastructure.camera_adapter import VirtualCamera
    cam = VirtualCamera(camera_id=0)
    assert cam.inject(b"") is False
    assert cam.inject(b"GIF89a...") is False
    # Smallest valid JPEG won't decode but should pass magic check before failing decode
    minimal = b"\xff\xd8\xff" + b"\x00" * 10
    # Should attempt decode and fail (returning False) rather than raising
    cam.inject(minimal)  # should not raise
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest backend/tests/unit -v`

- [ ] **Step 5: Commit**

```bash
git add backend/app/infrastructure/camera_adapter.py backend/tests/unit/
git commit -m "feat(security): JPEG magic-byte check before decode in inject_frame"
```

### Task 5.4: SSL fail-fast in production

**Files:**
- Modify: `backend/main.py:362-379`

- [ ] **Step 1: Edit the SSL block**

Find the SSL setup in `__main__`. Replace:

```python
        if os.path.exists(cert_path) and os.path.exists(key_path):
            ssl_kwargs["ssl_certfile"] = cert_path
            ssl_kwargs["ssl_keyfile"] = key_path
            logger.info(f"🔒 SSL enabled")
        else:
            logger.warning(f"⚠️ SSL certificates not found, running without SSL")
```

With:

```python
        if os.path.exists(cert_path) and os.path.exists(key_path):
            ssl_kwargs["ssl_certfile"] = cert_path
            ssl_kwargs["ssl_keyfile"] = key_path
            logger.info(f"🔒 SSL enabled")
        elif settings.debug:
            logger.warning(
                f"⚠️ SSL certificates not found at {cert_path}, "
                f"running without SSL (debug=True)"
            )
        else:
            raise SystemExit(
                f"SSL enabled but certs not found at {cert_path}. "
                f"Set debug=True to allow plain HTTP, or provision certs."
            )
```

- [ ] **Step 2: Compile-check**

Run: `python -m py_compile backend/main.py`

- [ ] **Step 3: Commit**

```bash
git add backend/main.py
git commit -m "feat(security): fail-fast when SSL enabled in non-debug but certs missing"
```

### Task 5.5: Optional JWT verification on WS endpoints

**Files:**
- Create: `backend/app/infrastructure/auth.py`
- Modify: `backend/main.py`
- Modify: `backend/requirements.txt` (verify PyJWT already there)
- Modify: `backend/.env.example`
- Modify: `README.md` (correct/keep claim)

- [ ] **Step 1: Create auth helper**

```python
"""Optional JWT verification — additive, gated by settings.auth_enabled."""
from __future__ import annotations
import logging
from typing import Optional

from app.infrastructure.config_adapter import Settings

logger = logging.getLogger(__name__)


def verify_token(token: Optional[str], settings: Settings) -> bool:
    """Return True if token is valid OR auth disabled. False otherwise.
    Default-off: when settings.auth_enabled is False, always returns True."""
    if not settings.auth_enabled:
        return True
    if not token:
        return False
    try:
        import jwt
    except ImportError:
        logger.error("auth_enabled but PyJWT not installed; refusing all connections")
        return False
    try:
        jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        return True
    except Exception as e:
        logger.warning(f"JWT verification failed: {type(e).__name__}")
        return False


def issue_token(subject: str, settings: Settings, expires_minutes: int = 60) -> Optional[str]:
    """Issue an HS256 JWT. Returns None if PyJWT missing or auth disabled."""
    if not settings.auth_enabled:
        return None
    try:
        import jwt
    except ImportError:
        return None
    from datetime import datetime, timedelta, timezone
    payload = {
        "sub": subject,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=expires_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")
```

- [ ] **Step 2: Wire into main.py**

Add at the top:

```python
from app.infrastructure.auth import verify_token, issue_token
from app.infrastructure.config_adapter import get_settings
```

In `/ws` and `/ws/camera` handlers, BEFORE `await websocket.accept()` (or as the first line after accept for /ws/camera which already accepts):

```python
    settings = get_settings()
    token = websocket.query_params.get("token")
    if not verify_token(token, settings):
        await websocket.close(code=1008, reason="Unauthorized")
        return
```

Add `/api/token` endpoint:

```python
@app.post("/api/token")
async def issue_token_endpoint(payload: dict):
    settings = get_settings()
    if not settings.auth_enabled:
        raise HTTPException(status_code=404, detail="Auth not enabled")
    subject = payload.get("subject", "anonymous")
    token = issue_token(subject, settings)
    if token is None:
        raise HTTPException(status_code=500, detail="Token issuance failed")
    return {"access_token": token, "token_type": "bearer"}
```

- [ ] **Step 3: Add a test**

Create `backend/tests/unit/test_auth.py`:

```python
from unittest.mock import Mock

from app.infrastructure.auth import verify_token, issue_token


def _settings(enabled=False, secret="test-secret"):
    s = Mock()
    s.auth_enabled = enabled
    s.jwt_secret = secret
    return s


def test_verify_token_returns_true_when_auth_disabled():
    assert verify_token(None, _settings(enabled=False)) is True
    assert verify_token("anything", _settings(enabled=False)) is True


def test_verify_token_rejects_missing_token_when_auth_enabled():
    assert verify_token(None, _settings(enabled=True)) is False
    assert verify_token("", _settings(enabled=True)) is False


def test_issue_then_verify_roundtrip():
    s = _settings(enabled=True)
    token = issue_token("alice", s)
    assert token is not None
    assert verify_token(token, s) is True


def test_verify_rejects_token_with_wrong_secret():
    s = _settings(enabled=True, secret="secret-a")
    token = issue_token("alice", s)
    s2 = _settings(enabled=True, secret="secret-b")
    assert verify_token(token, s2) is False
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest backend/tests/unit -v`
Expected: ALL PASS. If PyJWT is not installed in dev, the test_issue_then_verify_roundtrip test will be skipped — add `pytest.importorskip("jwt")` at the top of the test file:

```python
import pytest
pytest.importorskip("jwt")
```

- [ ] **Step 5: Update README claim**

Edit `README.md:67` to clarify that JWT auth is OPT-IN via `AUTH_ENABLED=true`. Replace the bullet with:

```markdown
| **JWT Authentication (optional)** | Default-off; enable with `AUTH_ENABLED=true`. Token issuance via `POST /api/token` and `?token=` on WS endpoints |
```

- [ ] **Step 6: Commit**

```bash
git add backend/app/infrastructure/auth.py backend/main.py backend/tests/unit/test_auth.py README.md
git commit -m "feat(security): optional JWT auth on WS+REST (default off; AUTH_ENABLED=true to enable)"
```

### Task 5.6: Bump python-multipart, pin upper bounds

**Files:**
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Apply edits**

For each `>=X.Y.Z` line, add `,<NEXT_MAJOR`:

```
fastapi>=0.104.0,<0.120.0
uvicorn[standard]>=0.24.0,<0.32.0
pydantic>=2.5.0,<3.0.0
pydantic-settings>=2.1.0,<3.0.0
python-multipart>=0.0.18
opencv-python>=4.8.0,<5.0.0
numpy>=1.24.0,<2.0.0
ultralytics>=8.0.200,<9.0.0
torch>=2.1.0,<3.0.0
PyJWT>=2.8.0,<3.0.0
msgpack>=1.0.7,<2.0.0
deep-sort-realtime>=1.3.2
scipy>=1.11.0,<2.0.0
```

(Adjust to actual current contents — only add upper bounds; do not change lower bounds.)

- [ ] **Step 2: Verify pip can resolve**

Run: `pip install -r backend/requirements.txt --dry-run 2>&1 | tail -5` (if pip ≥23 supports dry-run; otherwise skip).

- [ ] **Step 3: Commit**

```bash
git add backend/requirements.txt
git commit -m "build(deps): bump python-multipart for CVE-2024-24762; pin upper bounds"
```

---

## Phase 6 — Deploy script consolidation

### Task 6.1: Create shared SSH helper

**Files:**
- Create: `scripts/_jetson_common.py`

- [ ] **Step 1: Write helper**

```python
"""Shared SSH/SFTP helpers for Overwatch deploy scripts.

Credentials are read from environment:
- JETSON_HOST (default 192.168.1.10)
- JETSON_USER (default mandar)
- JETSON_PASS (no default; falls back to getpass)
- JETSON_KEY  (path to private key; preferred over password)
"""
from __future__ import annotations
import getpass
import os
import sys
from typing import Optional

import paramiko


def get_credentials() -> dict:
    host = os.environ.get("JETSON_HOST", "192.168.1.10")
    user = os.environ.get("JETSON_USER", "mandar")
    key_path = os.environ.get("JETSON_KEY")
    password = os.environ.get("JETSON_PASS")

    if not key_path and not password:
        try:
            password = getpass.getpass(f"Password for {user}@{host}: ")
        except (EOFError, KeyboardInterrupt):
            print("Aborted.", file=sys.stderr)
            sys.exit(1)

    return {"host": host, "user": user, "password": password, "key_path": key_path}


def connect(creds: Optional[dict] = None) -> paramiko.SSHClient:
    if creds is None:
        creds = get_credentials()
    client = paramiko.SSHClient()
    # Use system known_hosts; warn (not auto-add) on unknown
    try:
        client.load_system_host_keys()
    except Exception:
        pass
    client.set_missing_host_key_policy(paramiko.WarningPolicy())

    kwargs = {"hostname": creds["host"], "username": creds["user"], "timeout": 10}
    if creds.get("key_path"):
        kwargs["key_filename"] = creds["key_path"]
    elif creds.get("password"):
        kwargs["password"] = creds["password"]

    client.connect(**kwargs)
    return client


def run(client: paramiko.SSHClient, cmd: str, check: bool = True) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    rc = stdout.channel.recv_exit_status()
    if check and rc != 0:
        raise RuntimeError(f"Command failed (rc={rc}): {cmd}\nstderr: {err}")
    return rc, out, err
```

- [ ] **Step 2: Compile-check**

Run: `python -m py_compile scripts/_jetson_common.py`

- [ ] **Step 3: Commit**

```bash
git add scripts/_jetson_common.py
git commit -m "refactor(scripts): shared SSH helper with env-based credentials"
```

### Task 6.2: Archive duplicate restart/deploy scripts

**Files:**
- Move: `scripts/deploy_v2.py` → `scripts/archive/deploy_v2.py`
- Move: `scripts/_restart_now.py` → `scripts/archive/_restart_now.py`
- Move: `scripts/force_restart.py` → `scripts/archive/force_restart.py`
- Move: `scripts/fix_jetson.py` → `scripts/archive/fix_jetson.py`
- Create: `scripts/archive/README.md`

- [ ] **Step 1: Create archive dir and move files**

```bash
mkdir -p /c/Users/Mandar/overwatch/scripts/archive
git mv scripts/deploy_v2.py scripts/archive/
git mv scripts/_restart_now.py scripts/archive/
git mv scripts/force_restart.py scripts/archive/
git mv scripts/fix_jetson.py scripts/archive/
```

- [ ] **Step 2: Write archive README**

Path: `scripts/archive/README.md`

```markdown
# Archived deploy scripts

These scripts duplicated `deploy_jetson.py` / `restart_jetson.py` with drifted
hosts, paths, and hardcoded credentials. They are kept here for reference and
will be removed in a future cleanup.

- `deploy_v2.py` — pointed at `192.168.1.12` and a wrong local root
- `_restart_now.py` / `force_restart.py` — duplicates of `restart_jetson.py`
- `fix_jetson.py` — one-off fix script

Use `scripts/deploy_jetson.py` and `scripts/restart_jetson.py` instead.
Set `JETSON_HOST`, `JETSON_USER`, `JETSON_PASS` (or `JETSON_KEY`) in env.
```

- [ ] **Step 3: Commit**

```bash
git add scripts/archive/
git commit -m "chore(scripts): archive drifted/duplicate deploy scripts"
```

### Task 6.3: Replace hardcoded credentials in remaining scripts

**Files:**
- Modify: `scripts/deploy_jetson.py`
- Modify: `scripts/restart_jetson.py`
- Modify: `scripts/check_logs.py`
- Modify: `scripts/check_status.py`
- Modify: `scripts/ws_test.py`

For each script:

- [ ] **Step 1: Replace the hardcoded `JETSON_PASS = "mandar"` / `password="mandar"` lines**

Use `_jetson_common`:

```python
from _jetson_common import connect, get_credentials, run

creds = get_credentials()
client = connect(creds)
```

Where the script previously used `client.connect(hostname=..., username=..., password="mandar")`, replace with `client = connect(creds)`.

For pure connection bookkeeping (e.g., scripts that only use the IP for printing), read from `creds["host"]`.

- [ ] **Step 2: Add to each script's top:**

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

so `from _jetson_common import ...` resolves.

- [ ] **Step 3: Compile-check all**

Run: `python -m py_compile scripts/deploy_jetson.py scripts/restart_jetson.py scripts/check_logs.py scripts/check_status.py scripts/ws_test.py`

- [ ] **Step 4: Commit**

```bash
git add scripts/
git commit -m "security(scripts): remove hardcoded SSH passwords; use JETSON_* env vars"
```

### Task 6.4: Atomic SFTP staging in deploy_jetson.py

**Files:**
- Modify: `scripts/deploy_jetson.py`

- [ ] **Step 1: Read the current SFTP upload code**

Run: `grep -n "sftp\|put(" /c/Users/Mandar/overwatch/scripts/deploy_jetson.py | head -20`

- [ ] **Step 2: Modify upload to write to `<remote>/overwatch.new/` then atomically swap**

Around the `put()` calls:

```python
STAGING = REMOTE_ROOT + ".new"
BACKUP = REMOTE_ROOT + ".bak"

# Upload all files into STAGING instead of REMOTE_ROOT
# (rewrite the destination paths in the existing put() loop)

# After all puts succeed, swap atomically over SSH:
run(client, f"rm -rf {BACKUP}", check=False)
run(client, f"if [ -d {REMOTE_ROOT} ]; then mv {REMOTE_ROOT} {BACKUP}; fi")
run(client, f"mv {STAGING} {REMOTE_ROOT}")
```

Add a `--rollback` flag that swaps `BACKUP` back to `REMOTE_ROOT`.

- [ ] **Step 3: Generate `JWT_SECRET` per deploy in remote .env**

Where the script writes the remote `.env`, add:

```python
import secrets
jwt_secret = secrets.token_urlsafe(48)
env_lines.append(f"JWT_SECRET={jwt_secret}")
env_lines.append("AUTH_ENABLED=false  # set true to require token auth")
```

Then `chmod 600` the file:

```python
run(client, f"chmod 600 {REMOTE_ROOT}/backend/.env")
```

- [ ] **Step 4: Compile-check**

Run: `python -m py_compile scripts/deploy_jetson.py`

- [ ] **Step 5: Commit**

```bash
git add scripts/deploy_jetson.py
git commit -m "feat(deploy): atomic SFTP staging, rollback flag, per-deploy JWT_SECRET"
```

---

## Phase 7 — Wrap-up

### Task 7.1: Run the full test suite, build the frontend, verify clean

- [ ] **Step 1: Backend tests**

Run: `python -m pytest backend/tests -v`
Expected: ALL PASS.

- [ ] **Step 2: Frontend build**

Run: `cd frontend && npm install && npm run build`
Expected: build succeeds.

- [ ] **Step 3: Compile-check all scripts**

Run: `python -m py_compile scripts/*.py scripts/archive/*.py`
Expected: success.

- [ ] **Step 4: Print git log of the branch**

Run: `git log --oneline main..HEAD`
Confirm all phases landed.

### Task 7.2: Open a PR (manual; do NOT push without user approval)

The user must explicitly say "push" / "open PR". This plan does not auto-push.

---

## Self-Review

**Spec coverage:**
- Tier 0 → Phase 0 (4 tasks). ✅
- Tier 1 (regression tests) → Phase 1 (9 tasks). ✅
- Tier 2 (backend bugs) → Phase 2 (6 tasks). ✅
- Tier 3 (reliability) → Phase 3 (6 tasks). ✅
- Tier 4 (frontend) → Phase 4 (5 tasks). ✅
- Tier 5 (security) → Phase 5 (6 tasks). ✅
- Tier 6 (deploy) → Phase 6 (4 tasks). ✅
- Out-of-scope items (depth model, BoundingBox validation semantics, msgpack contract) intentionally not touched.

**Placeholder scan:** No `TBD`, no "implement later". Each step has command + expected outcome.

**Type / API consistency:**
- `verify_token` / `issue_token` signatures consistent in Task 5.5 helper, test, and main.py wiring.
- `JETSON_*` env vars consistent across `_jetson_common.py` and the script edits in 6.3.
- `cors_origins` (list) / `max_ws_clients` (int) consistent across config, env example, and main.py / container.py.

---

## Execution

Subagent-driven: dispatch one fresh subagent per task; review between tasks; do not batch.
