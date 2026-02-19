<p align="center">
  <img src="https://img.shields.io/badge/OVERWATCH-v2.0.0-00ffc8?style=for-the-badge&labelColor=0a0a0a" alt="Version" />
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/react-18-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React" />
  <img src="https://img.shields.io/badge/TensorRT-FP16-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="TensorRT" />
  <img src="https://img.shields.io/badge/Jetson_Orin_Nano-Edge-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="Jetson" />
  <img src="https://img.shields.io/badge/license-Proprietary-red?style=for-the-badge" alt="License" />
</p>

<h1 align="center">🎯 OVERWATCH</h1>

<p align="center">
  <strong>Real-time multi-agent collaborative perception system</strong><br/>
  <em>Multi-camera tracking · AI-powered sensor fusion · Augmented reality overlays · Edge deployment</em>
</p>

<p align="center">
  <a href="#-demo-video">Demo</a> ·
  <a href="#-features">Features</a> ·
  <a href="#-architecture">Architecture</a> ·
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-deployment">Deployment</a> ·
  <a href="#-api-reference">API Reference</a> ·
  <a href="#-troubleshooting">Troubleshooting</a>
</p>

---

## Demo Video

[Watch Demo on YouTube](https://youtu.be/L_jDzPQBXO8)

---

## Overview

OVERWATCH is a real-time multi-camera situational awareness platform built for edge deployment on NVIDIA Jetson hardware. It fuses video from IP cameras and mobile phones into a unified world model using YOLOv8 detection, Hungarian-assignment tracking, adaptive Kalman filtering, and cross-camera appearance re-identification — all at TensorRT FP16 speeds.

The system runs a **singleton perception pipeline** — detection, tracking, and fusion execute **once per tick** regardless of how many viewers are connected, then broadcast pre-serialized snapshots to all clients over binary WebSocket.

---

## 🚀 Features

### Core Perception
| Capability | Implementation |
|---|---|
| **Person Detection** | YOLOv8n with NMS-level class filter (`classes=[0]`) — person-only |
| **TensorRT FP16** | `.engine` export on Jetson — ~8 MiB, sub-10ms inference |
| **Hungarian Tracking** | `scipy.optimize.linear_sum_assignment` — 0.6×IoU + 0.4×cosine appearance cost |
| **Tracker Fallback Chain** | DeepSORT (MobileNet) → Hungarian (scipy) → Simple (centroid) |
| **Adaptive Kalman Filter** | 6-state `[x,y,z,vx,vy,vz]` — measurement noise scales by confidence, bbox area, sensor trust |
| **Cross-Camera Re-ID** | 64-dim HSV histogram descriptors, L2-normalized, EMA-smoothed (α=0.3) |
| **Sensor Trust Scoring** | Per-sensor trust ∈ [0.1, 1.0] — increases for consistent measurements, decays for innovation outliers |
| **Cross-Camera Homography** | Self-calibrating ground-plane H from shared foot-point observations via `cv2.findHomography` + RANSAC — projects person positions across camera views in <0.1ms |
| **3-Path Ghost Predictions** | Path A: homography projection from ANY source camera (green). Path B: pixel extrapolation with adaptive budget (red). Path C: world-coordinate pinhole projection fallback (orange). Ensures ghosts appear even when no homography exists and the target camera has never seen the person. |

### Platform
| Capability | Implementation |
|---|---|
| **Multi-Camera** | Up to 4 concurrent streams (physical MJPEG/RTSP + mobile virtual cameras) |
| **Mobile Streaming** | Phone browsers → `getUserMedia` → binary JPEG over WebSocket → `VirtualCamera` |
| **GPS + IMU Fusion** | Mobile geolocation → equirectangular projection; `DeviceOrientationEvent` → camera rotation |
| **AR Overlays** | Canvas-based: cyan detection brackets, yellow track boxes, green homography ghosts, red extrapolation ghosts |
| **Binary Protocol** | msgpack-serialized snapshots — zero-copy broadcast to all viewers |
| **SSL/TLS** | Self-signed certificates with SAN for LAN IP access (required for `getUserMedia`) |
| **JWT Authentication** | Token issuance endpoint (`POST /api/token`) with configurable expiry |
| **Edge Deployment** | Automated SSH/SFTP deployment to Jetson Orin Nano via paramiko |

---

## 🏗️ Architecture

```
                          ┌─────────────────────────────────┐
                          │       OVERWATCH  v2.0.0         │
                          └─────────────────────────────────┘

  ╔═══════════════╗       ╔═══════════════════════════════════════════════════╗
  ║  DATA SOURCES ║       ║          JETSON ORIN NANO  (backend :8000)       ║
  ╠═══════════════╣       ╠═══════════════════════════════════════════════════╣
  ║               ║       ║                                                   ║
  ║  📷 IP Camera ─────────►  CameraCapture (OpenCV, MJPEG/RTSP)            ║
  ║               ║       ║       │                                           ║
  ║  📱 Mobile    ─────────►  VirtualCamera (binary JPEG push)               ║
  ║   Phone       ║ws/cam ║       │         + GPS/IMU sensor data             ║
  ║               ║       ║       ▼                                           ║
  ║               ║       ║  ┌──────────────────────────────────────────┐     ║
  ║               ║       ║  │     PerceptionPipeline  (singleton)     │     ║
  ║               ║       ║  │                                          │     ║
  ║               ║       ║  │  1. DETECT   YOLOv8n TensorRT FP16     │     ║
  ║               ║       ║  │              + HSV appearance features   │     ║
  ║               ║       ║  │                     │                    │     ║
  ║               ║       ║  │  2. TRACK    Hungarian assignment       │     ║
  ║               ║       ║  │              IoU + cosine appearance     │     ║
  ║               ║       ║  │                     │                    │     ║
  ║               ║       ║  │  3. FUSE     Adaptive Kalman 6-state    │     ║
  ║               ║       ║  │              Cross-camera matching      │     ║
  ║               ║       ║  │              Sensor trust scoring       │     ║
  ║               ║       ║  │                     │                    │     ║
  ║               ║       ║  │  4. SNAPSHOT Pre-serialized msgpack     │     ║
  ║               ║       ║  └──────────────┬───────────────────────────┘     ║
  ║               ║       ║                 │                                 ║
  ╚═══════════════╝       ║                 ▼  broadcast                      ║
                          ║     WebSocketManager (/ws, msgpack binary)        ║
                          ║         │           │           │                  ║
                          ╚═════════╪═══════════╪═══════════╪═════════════════╝
                                    │           │           │
                          ┌─────────▼──┐  ┌─────▼──┐  ┌────▼───┐
                          │  Viewer 1  │  │Viewer 2│  │Viewer N│
                          │  React     │  │  React │  │  React │
                          │  AR Canvas │  │  ...   │  │  ...   │
                          └────────────┘  └────────┘  └────────┘
```

### Pipeline Design

Unlike traditional per-viewer architectures, OVERWATCH runs a **single shared pipeline**. The `PerceptionPipeline` singleton executes detect → track → fuse **once per tick**, produces a `PerceptionSnapshot` with pre-serialized msgpack packets, and all connected viewers simply read from the latest snapshot. This means:

- **1 camera + 10 viewers = 1 GPU inference** (not 10)
- Zero-copy broadcast via pre-serialized binary packets
- Slow viewers gracefully skip intermediate frames

---

## 📁 Project Structure

```
OVERWATCH/
│
├── backend/                              # FastAPI + Perception Engine
│   ├── main.py                           # App entry, lifespan, REST + WS endpoints
│   ├── requirements.txt                  # Python dependencies (CPU/Windows)
│   ├── requirements-jetson.txt           # Jetson Orin Nano dependencies
│   ├── yolov8n.pt                        # YOLOv8 nano weights (~6 MB)
│   ├── yolov8n.engine                    # TensorRT FP16 engine (Jetson, ~8.9 MB)
│   ├── .env                              # Runtime configuration
│   ├── certs/                            # SSL certificates
│   └── app/
│       ├── application/                  # Use cases & business logic
│       │   ├── ports.py                  # Repository interfaces
│       │   └── services.py               # Perception pipeline service
│       ├── domain/                       # Core entities
│       │   └── entities.py               # Detection, Track, WorldObject, etc.
│       └── infrastructure/               # Adapters
│           ├── camera_adapter.py          # OpenCV camera capture
│           ├── detection_adapter.py      # YOLO wrapper
│           ├── tracking_adapter.py        # Hungarian/DeepSORT
│           ├── world_model_adapter.py     # Kalman filter fusion
│           └── websocket_adapter.py       # Binary msgpack broadcast
│
├── frontend/                             # React 18 Admin Dashboard
│   ├── package.json
│   ├── .env                              # REACT_APP_BACKEND_HOST / PORT
│   ├── build/                            # Production build
│   └── src/
│       ├── pages/
│       │   ├── AdminDashboard.jsx        # Main camera grid view
│       │   └── MobileCamera.jsx          # Phone camera streaming UI
│       ├── components/
│       │   ├── CameraDisplay.jsx         # Canvas AR overlay renderer
│       │   ├── StatsPanel.jsx            # System statistics
│       │   └── ConnectionStatus.jsx      # WS connection indicator
│       ├── application/hooks/            # React hooks
│       │   ├── useCameraData.js         # Frame/detection handling
│       │   ├── useWebSocket.js           # WebSocket connection
│       │   └── useSystemStats.js         # Backend status polling
│       └── infrastructure/
│           ├── websocketAdapter.js       # msgpack binary WS client
│           └── cameraStreamAdapter.js     # getUserMedia → WS
│
├── scripts/                              # Deployment & Operations
│   ├── deploy_jetson.py                  # Full SSH/SFTP deployment to Jetson
│   └── restart_jetson.py                  # Quick restart backend
│
├── certs/                                # SSL certificates (self-signed)
│   ├── cert.pem
│   └── key.pem
│
└── README.md
```

---

## ⚡ Quick Start

### Prerequisites

- **Python 3.10+** with pip
- **Node.js 18+** with npm
- **NVIDIA Jetson Orin Nano** (production) or any machine with CUDA *(development)*

### 1. Clone

```bash
git clone https://github.com/mandarwagh9/overwatch.git
cd overwatch
```

### 2. Build & Run

```bash
# Build frontend
cd frontend
npm install
npm run build
cd ..

# Start backend (serves frontend + API on port 8000)
cd backend
python main.py
```

That's it! Backend serves both the React app and API at **https://localhost:8000**

### 3. Deploy to Jetson

```bash
python scripts/deploy_jetson.py
```

### 4. Access

| Service | URL |
|---|---|
| Admin Dashboard | https://192.168.1.12:8000 |
| Mobile Camera | https://192.168.1.12:8000/mobile |

### 2. Local Development (Windows/Mac/Linux)

```bash
# Backend
cd backend
pip install -r requirements.txt
python main.py

# Frontend (new terminal)
cd frontend
npm install
npm start
```

Open **https://localhost:3000** — accept the self-signed certificate warning.

### 3. Deploy to Jetson

```bash
# One-command deployment - uploads code, builds frontend, starts backend
python scripts/deploy_jetson.py
```

This script will:
- Connect to Jetson at `192.168.1.12`
- Upload backend, frontend build, and SSL certs
- Install Python dependencies
- Create optimized `.env` config
- Start the backend

### 4. Access

| Service | URL |
|---|---|
| Admin Dashboard | https://192.168.1.12:8000 |
| Mobile Camera | https://192.168.1.12:8000/mobile |

### Quick Operations

```bash
# Restart backend (without redeploying)
python scripts/restart_jetson.py

# View logs
ssh mandar@192.168.1.12 'tail -50 /tmp/overwatch.log'
```

---

## 🚀 Deployment

### Deploy to Jetson

```bash
python scripts/deploy_jetson.py
```

This uploads everything (backend + frontend build + certs), installs deps, and starts the backend.

### Quick Operations

```bash
# Restart backend (without redeploying)
python scripts/restart_jetson.py

# View logs
ssh mandar@192.168.1.12 'tail -50 /tmp/overwatch.log'
```

---

## 📡 API Reference

### REST Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check — returns version, status |
| `GET` | `/health` | Detailed health status |
| `GET` | `/status` | System status (cameras, clients, detection engine, pipeline metrics) |
| `GET` | `/cameras` | Active camera list |
| `GET` | `/mobile` | Standalone mobile camera HTML page |
| `POST` | `/cameras/{camera_id}/start` | Start a physical camera |
| `POST` | `/cameras/{camera_id}/stop` | Stop a camera |

#### Health Check Response

```json
{
  "message": "Overwatch API is running",
  "version": "2.0.0",
  "status": "operational",
  "capabilities": [
    "perception_pipeline",
    "hungarian_tracking",
    "adaptive_kalman",
    "appearance_reid",
    "gps_imu_fusion",
    "world_update_broadcast"
  ]
}
```

### WebSocket Endpoints

| Endpoint | Direction | Format | Purpose |
|---|---|---|---|
| `/ws` | Server → Client | msgpack binary | Viewer stream (frames + detections + tracks + predictions) |
| `/ws/camera` | Client → Server | Binary JPEG + JSON | Mobile camera source |

### WebSocket Message Types

<details>
<summary><strong>Frame Message</strong> (server → viewer)</summary>

```json
{
  "type": "frame",
  "camera_id": 0,
  "timestamp": 1706745600.123,
  "frame_data": "<JPEG bytes>",
  "detections": [
    { "bbox": [x1, y1, x2, y2], "confidence": 0.87, "class_name": "person", "feature_vector": "..." }
  ],
  "tracks": [
    { "track_id": 1, "bbox": [x1, y1, x2, y2], "velocity": [dx, dy], "confidence": 0.9 }
  ],
  "predictions": [
    { "object_id": 1, "bbox": [x1, y1, x2, y2], "time_since_seen": 1.2, "confidence": 0.6, "inferred": true }
  ]
}
```
</details>

<details>
<summary><strong>World Update</strong> (server → viewer)</summary>

```json
{
  "type": "world_update",
  "timestamp": 1706745600.123,
  "objects": [
    {
      "id": "obj_1",
      "class_id": 0,
      "position": [2.3, 1.1, 0.0],
      "velocity": [0.5, -0.2, 0.0],
      "confidence": 0.85,
      "last_seen_camera": 0,
      "position_uncertainty": 0.12,
      "bbox_size": [45, 120]
    }
  ],
  "stats": { "tick_ms": 42.3, "avg_tick_ms": 38.7, "cameras": 2 }
}
```
</details>

<details>
<summary><strong>Mobile Registration</strong> (camera source handshake)</summary>

```
Client → { "type": "register", "role": "camera_source", "camera_id": null }
Server → { "type": "registered", "camera_id": 0, "target_fps": 15 }
Client → [binary JPEG frames at target FPS]
Client → { "type": "sensor_data", "gps": {...}, "orientation": {...} }
```
</details>

---

## 🔧 Configuration

### Backend `.env`

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `yolov8n.pt` | Model file — `.pt`, `.engine` (TensorRT), or `.onnx` |
| `DEVICE` | `auto` | Compute device — `auto`, `cpu`, `cuda:0` |
| `HALF_PRECISION` | `false` | FP16 inference (set `true` on Jetson with `.engine`) |
| `DETECTION_CLASSES` | `[0]` | COCO class IDs to detect (`0` = person) |
| `CONFIDENCE_THRESHOLD` | `0.5` | Detection confidence threshold |
| `IOU_THRESHOLD` | `0.45` | NMS IOU threshold |
| `TARGET_FPS` | `24` | Processing framerate target |
| `MAX_CAMERAS` | `4` | Maximum concurrent camera streams |
| `TRACKING_MAX_AGE` | `30` | Max frames to keep lost tracks |
| `TRACKING_MIN_HITS` | `3` | Min hits to confirm track |
| `TRACKING_IOU_THRESHOLD` | `0.25` | IoU threshold for tracking |
| `MOBILE_CAMERA_FPS` | `15` | Mobile camera target FPS |
| `MOBILE_CAMERA_MAX_WIDTH` | `640` | Mobile camera max width |
| `SSL_ENABLED` | `true` | Enable HTTPS/WSS |
| `SSL_CERTFILE` | `certs/cert.pem` | SSL certificate path |
| `SSL_KEYFILE` | `certs/key.pem` | SSL private key path |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |

### Frontend `.env`

| Variable | Default | Description |
|---|---|---|
| `REACT_APP_BACKEND_HOST` | `window.location.hostname` | Backend IP address |
| `REACT_APP_BACKEND_PORT` | `8000` | Backend port |
| `REACT_APP_BACKEND_PROTOCOL` | `wss` (https) / `ws` (http) | WebSocket protocol |
| `REACT_APP_MAX_CAMERAS` | `4` | Maximum cameras to display |
| `REACT_APP_CAMERA_INACTIVITY_TIMEOUT` | `3000` | ms before marking camera offline |
| `REACT_APP_MOBILE_TARGET_FPS` | `15` | Mobile streaming FPS |
| `REACT_APP_MOBILE_JPEG_QUALITY` | `0.5` | Mobile JPEG quality (0-1) |
| `REACT_APP_MOBILE_MAX_WIDTH` | `640` | Mobile frame width |

---

## 🎯 AR Overlay System — EagleEye Tactical HUD

The frontend renders an Anduril EagleEye-inspired tactical overlay with diamond markers, compass ribbon, and threat rings:

| Layer | Color | Elements |
|---|---|---|
| **Detections** | Slate-blue `#64b5f6` | Diamond markers, corner brackets, `PERSON` confidence pill, BLOS indicators |
| **Tracks** | Amber `#ffd740` | Diamond/chevron markers, velocity vector arrows, track ID callouts |
| **Predictions (H-PROJ)** | Green `#00ff82` solid | Homography-projected ghost — accurate, real-time cross-camera |
| **Predictions (EXTRAP)** | Red `#ff5050` dashed | Pixel-extrapolated ghost — time-decaying dead-reckoning |
| **Predictions (WORLD)** | Orange `#ff9800` dashed | World-coordinate projection — rough pinhole-model fallback |
| **Compass Ribbon** | — | Heading ribbon with N/E/S/W and bearing tick marks |
| **Threat Ring** | Per-IFF color | Inner ring around camera feed showing bearing to off-screen predictions |

Detection overlays show what the model sees *right now*. Track overlays show persistent identity across frames. Predictions show cross-camera projections — green for homography (most accurate), orange for world-model fallback (rough but always available), red for pixel extrapolation (last resort).

---

## 🌍 World Model & Sensor Fusion

### Kalman Filter
Each fused world object maintains a 6-state Kalman filter: `[x, y, z, vx, vy, vz]` with constant-velocity dynamics. Measurement noise **R** adapts per-update based on detection confidence, bounding box area, and sensor trust — higher-quality observations tighten the filter, while noisy or untrusted sensors widen it.

### Cross-Camera Association
Objects from different cameras are matched when:
- Euclidean distance < 2 meters
- Same `class_id`
- Appearance cosine similarity > 0.5 (when feature vectors available)

### Sensor Trust
Each camera/sensor earns trust through consistency:
- **Consistent measurements** → trust increases (capped at 1.0)
- **Innovation outliers** → trust decays (floored at 0.1)

### Appearance Re-ID
- 64-dimensional HSV histogram descriptors computed per detection (~0.1ms each)
- L2-normalized for cosine similarity
- Exponential moving average (α=0.3) for descriptor stability across frames

---

## 📱 Mobile Camera Streaming

Any phone on the same LAN can become a camera source:

**Via React app**: `https://<frontend-ip>:3001/mobile`
**Standalone page**: `https://<jetson-ip>:8000/mobile`

The mobile client:
1. Opens rear camera via `getUserMedia` (1280×720)
2. Renders to offscreen canvas → extracts JPEG blob
3. Sends binary frames over WebSocket to `/ws/camera`
4. Captures GPS (`watchPosition`, high accuracy) and IMU (`DeviceOrientationEvent`) at 2 Hz
5. Sends sensor data as JSON for camera calibration fusion

> **Note**: `getUserMedia` requires HTTPS — this is why SSL certificates are mandatory even for LAN deployments.

---

## 🐛 Troubleshooting

<details>
<summary><strong>WebSocket won't connect</strong></summary>

1. Visit `https://<jetson-ip>:8000` in your browser and accept the self-signed certificate
2. Verify `REACT_APP_BACKEND_HOST` in `frontend/.env` matches the backend IP
3. Check the backend is running: `curl -sk https://<jetson-ip>:8000/`
</details>

<details>
<summary><strong>Mobile camera shows black screen</strong></summary>

- HTTPS is required for `getUserMedia` — ensure `SSL_ENABLED=true`
- Phone must be on the same LAN as the backend
- Allow camera permission when the browser prompts
- Try the standalone page: `https://<jetson-ip>:8000/mobile`
</details>

<details>
<summary><strong>TensorRT .to() error</strong></summary>

TensorRT `.engine` files are already GPU-bound. The detection engine correctly skips `.to()` for these models. If you see this error, ensure you're using the latest `detection_engine.py`.
</details>

<details>
<summary><strong>Pydantic "Config and model_config" error</strong></summary>

Use only the `model_config = SettingsConfigDict(...)` dict pattern — do not define an inner `class Config`. This is the Pydantic v2 convention.
</details>

<details>
<summary><strong>Port already in use on Jetson</strong></summary>

```bash
python scripts/restart_jetson.py
# Or manually:
pkill -9 -f 'python3 main.py'
sleep 2
cd /home/mandar/overwatch/backend && nohup python3 main.py > /tmp/overwatch.log 2>&1 &
```
</details>

<details>
<summary><strong>Checking Jetson logs</strong></summary>

```bash
# Quick restart
python scripts/restart_jetson.py

# Or via SSH
ssh mandar@192.168.1.12 'tail -50 /tmp/overwatch.log'
```
</details>
<details>
<summary><strong>Ghost predictions not appearing on a camera</strong></summary>

If a person is visible in Camera 0 but no ghost appears in Camera 1:

1. **Check homography status** — look for `H learned: cam0→cam1` in logs. If missing, walk through both camera FOVs simultaneously to collect correspondence pairs.
2. **Check world projection** — Path C (orange ghost) should always work. If missing, verify `_simple_world_to_pixel()` camera positions match your physical setup.
3. **Check prediction horizon** — if `time_since_seen > prediction_horizon` (default 5s), the object is pruned. The person must be actively tracked by at least one camera.
4. **Check source_tracks** — if Camera 1 is currently tracking the person (in `source_tracks`), no prediction is generated (it's a live track, not a ghost).
</details>

<details>
<summary><strong>Orange (WORLD) ghosts are in the wrong position</strong></summary>

Path C world projection uses hardcoded camera positions. If ghosts land far from the actual person:
1. Edit `_simple_world_to_pixel()` in `world_model.py`
2. Set `camera_positions` dict to match your physical camera locations (x, y, z in meters)
3. Adjust `fov_deg` (default 60°) to match your camera lens
4. Redeploy: `python scripts/deploy_jetson.py && python scripts/restart_jetson.py`
</details>

<details>
<summary><strong>Ghosts flicker between green and orange</strong></summary>

This happens when the homography is borderline — sometimes projection succeeds (green), sometimes it fails and falls through to Path C (orange). Causes:
- Homography was learned from too few correspondence pairs (minimum 4, but 8+ is more stable)
- Person is near the edge of the overlap zone where reprojection error is highest
- Walk more paths through the camera overlap to collect additional pairs and improve $H$ stability
</details>

<details>
<summary><strong>Two people merged into one ghost</strong></summary>

Cross-camera re-ID matched two different people as the same world object. This can happen with:
- Identical clothing (same HSV histogram)
- People standing < 2m apart in world coordinates
- Temporary occlusion causing track ID swap

The system should self-correct once the people separate spatially. If persistent, the appearance descriptor EMA (α=0.3) will gradually diverge.
</details>
---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **Detection** | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (nano) |
| **Inference** | NVIDIA TensorRT FP16 / ONNX Runtime / PyTorch |
| **Tracking** | [DeepSORT](https://github.com/levan92/deep_sort_realtime) / Hungarian (scipy) / Centroid |
| **Fusion** | Custom 6-state Kalman filter with adaptive noise |
| **Cross-Camera** | Ground-plane homography via [OpenCV `findHomography`](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780) + RANSAC |
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn (ASGI) |
| **Protocol** | [msgpack](https://msgpack.org/) binary over WebSocket |
| **Frontend** | [React 18](https://react.dev/) + Canvas 2D API |
| **Auth** | [PyJWT](https://pyjwt.readthedocs.io/) (HS256) |
| **Hardware** | NVIDIA Jetson Orin Nano (JetPack 6.x, R36) |
| **Deployment** | [paramiko](https://www.paramiko.org/) SSH/SFTP automation |

---

## 📐 Cross-Camera Homography — How It Works

OVERWATCH's core feature is **ghost prediction**: when Camera 0 can't see a person but Camera 1 can, the system renders a ghost overlay on Camera 0's feed showing where that person is.

### The Problem with Naive Extrapolation

Simply sliding a person's last-known pixel position forward in time (dead-reckoning) fails within seconds because:
- Different cameras have completely different pixel coordinate systems
- The mapping between camera views is a **projective transformation**, not a linear offset
- A person at pixel `(400, 300)` in Camera 1 might correspond to `(800, 500)` in Camera 0

### The Solution: Learn the Camera-to-Camera Transform

When both cameras simultaneously observe the **same person** (matched via appearance re-ID), the system records **foot-point correspondence pairs** — the bottom-center of the bounding box in each camera's view. These foot points project to the same physical ground-plane location.

With ≥4 such pairs, `cv2.findHomography()` + RANSAC computes a 3×3 **homography matrix** $H$ that maps any ground-plane point from one camera's pixel space to another's:

$$\begin{pmatrix} x' \\ y' \\ w \end{pmatrix} = H \cdot \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}$$

### Self-Calibrating Pipeline

1. **Collect**: When re-ID matches a person across Camera 0 and Camera 1, record `(foot_cam0, foot_cam1)` pair
2. **Estimate**: After 4+ pairs, compute $H_{0→1}$ and $H_{1→0}$ via RANSAC (re-estimated every 5 new pairs)
3. **Project**: When Camera 0 loses a person but Camera 1 still sees them, apply $H_{1→0}$ to Camera 1's current foot point → get the position on Camera 0's feed
4. **Validate**: Monitor reprojection error; if it spikes (camera moved), flush and re-learn

### Computational Cost

- Homography estimation: **< 0.1ms** (called every 5 new pairs, not every frame)
- Per-prediction projection: **< 0.001ms** (one 3×3 matrix multiply)
- Total overhead per frame: **effectively zero** on Jetson Orin Nano

### Visual Indicators

| Ghost Color | Tag | Source | Meaning |
|---|---|---|---|
| 🟢 Green solid | `H-PROJ` | Path A — Homography | Cross-camera ground-plane projection. Tries **all** source cameras with valid $H$ to the target camera, picks the freshest. Most accurate. |
| 🟠 Orange dashed | `WORLD` | Path C — World projection | Fused 3D world position → pinhole camera model. Rough but **always works** even when no homography exists and target camera has never seen the person. |
| 🔴 Red dashed | `EXTRAP` | Path B — Pixel extrapolation | Slides last-known pixel position by velocity × time. Adaptive budget: `min(250px, 80 + 40 × t)`. Only works if the target camera previously saw the person. |

---

## ⚠️ Edge Cases & Known Limitations

### Cross-Camera Prediction

| Edge Case | Behavior | Mitigation |
|---|---|---|
| **No homography learned yet** | Path A fails silently. System falls through to Path B (extrap) or Path C (world projection). Ghost appears orange instead of green. | Walk through overlapping camera FOVs to collect ≥4 foot-point correspondence pairs. Homography auto-learns within ~5 seconds of co-visibility. |
| **Camera moved after calibration** | Homography reprojection error spikes. Stale $H$ matrix produces offset ghosts. | The system monitors reprojection error and flushes the homography when error exceeds 50px. Walk through overlap again to re-learn. |
| **Person only seen by one camera ever** | Path A has no source camera to project from. Path B has no pixel history for the target camera. Path C is the only option — ghost is orange and position is approximate. | This is the primary reason Path C (world projection) was added. Accuracy depends on how well the hardcoded camera positions in `_simple_world_to_pixel()` match physical reality. |
| **Cameras with no overlapping FOV** | No co-visible observations → no foot-point pairs → no homography learned. Path A never activates between these cameras. | Path C world projection still works. For better accuracy, calibrate camera extrinsics in `_simple_world_to_pixel()` (currently hardcoded positions). |
| **Object behind the camera (world projection)** | Pinhole model projects negative-depth points to invalid pixels. | Path C checks that projected pixel is within `[-0.5×W, 1.5×W]` and `[-0.5×H, 1.5×H]`. Out-of-bounds projections are silently dropped. |
| **Rapid camera switching (DHCP IP change)** | If the Jetson's IP changes, `frontend/.env` and all deploy scripts point to the wrong address. | Update `REACT_APP_BACKEND_HOST` in `frontend/.env` and run `grep -r '192.168.1' scripts/` to catch all references. Consider using mDNS hostname instead. |

### Tracking & Re-ID

| Edge Case | Behavior | Mitigation |
|---|---|---|
| **Identical clothing (twins/uniforms)** | HSV histogram descriptors are nearly identical. Re-ID may merge two people into one world object. | The system uses spatial distance (< 2m) AND appearance similarity (> 0.5 cosine) for cross-camera matching. If two people are far apart, they stay separate even with identical appearance. |
| **Person temporarily fully occluded** | Track coasts for `prediction_horizon` seconds (default 5s). Prediction confidence decays linearly. After timeout, track is pruned. | Increase `prediction_horizon` in config if longer persistence is needed. Kalman velocity estimate keeps the ghost moving during occlusion. |
| **Crowded scenes (>10 people)** | Hungarian assignment cost matrix grows as $O(n \times m)$. Appearance feature extraction adds ~0.1ms per detection. | YOLOv8n NMS already limits detections. The pipeline runs single-threaded on GPU — throughput may drop below target FPS with many detections. |
| **Person enters from off-screen** | No pixel history, no world object yet. First detection creates a new track with high measurement noise. | Kalman filter initializes with large uncertainty. Trust builds over 5–10 consistent frames. Ghost predictions only appear after the person is fused into the world model. |

### Sensor Fusion

| Edge Case | Behavior | Mitigation |
|---|---|---|
| **Mobile GPS jitter indoors** | GPS accuracy can be 10–50m indoors. Kalman filter receives noisy position updates. | Sensor trust scoring automatically down-weights GPS sources with high innovation. The trust floor (0.1) prevents complete rejection. |
| **Mobile phone loses WebSocket** | Virtual camera stream stops. Existing tracks from that camera coast via Kalman prediction. | Tracks persist for `prediction_horizon` seconds. Phone auto-reconnects and gets a new camera ID. |
| **Clock drift between cameras** | Frame timestamps from different cameras may not be synchronized. Co-visibility matching uses a 0.5s window. | The 0.5s co-visibility window is generous enough for typical LAN latency. NTP sync across devices is recommended for sub-100ms accuracy. |

### Network & Deployment

| Edge Case | Behavior | Mitigation |
|---|---|---|
| **Self-signed cert rejected by browser** | WebSocket connection fails silently. Frontend shows no camera feeds. | Navigate to `https://<jetson-ip>:8000` directly and accept the certificate. This must be done once per browser session. |
| **Jetson runs out of GPU memory** | TensorRT engine uses ~30 MiB. With 4 cameras at 640×640, CUDA memory usage is ~200 MiB total. Orin Nano has 8 GB shared. | Monitor with `tegrastats`. If memory is tight, reduce `MAX_CAMERAS` or input resolution. |
| **Backend crash / watchdog** | Uvicorn runs with `--reload` (WatchFiles). File changes trigger auto-restart. Crash requires manual `python scripts/force_restart.py`. | Consider adding systemd service with `Restart=always` for production. |
| **Multiple viewers cause lag** | The singleton pipeline runs once per tick regardless of viewers. However, msgpack serialization + WebSocket send scales linearly with viewer count. | Pre-serialized snapshots minimize per-viewer cost. For >10 viewers, consider adding a pub/sub layer (Redis, NATS). |

### World Projection (Path C) Accuracy

Path C uses a simplified pinhole camera model with **hardcoded camera positions**:

```python
camera_positions = {
    0: (0, 0, 2),    # Origin, 2m height
    1: (5, 0, 2),    # 5m to the right
    2: (0, 5, 2),    # 5m forward
    3: (-5, 0, 2),   # 5m to the left
}
```

These positions assume a rectangular room setup. If your cameras are arranged differently:
1. Edit `_simple_world_to_pixel()` in [world_model_adapter.py](backend/app/infrastructure/world_model_adapter.py) with actual camera positions
2. Adjust the FOV constant (currently 60°) to match your cameras
3. Path C accuracy improves dramatically with correct extrinsics — ghosts land within ~50px of true position vs ~200px with wrong positions

---

## 📚 References & Sources

The cross-camera homography system is built on established multi-view geometry principles and inspired by several academic works and open-source implementations:

### Foundational Theory

| Source | Relevance |
|---|---|
| Hartley, R. & Zisserman, A. (2004). **"Multiple View Geometry in Computer Vision"**, 2nd ed. Cambridge University Press. | Chapter 13: ground-plane homography between uncalibrated camera pairs. The mathematical foundation for projecting points across views via a 3×3 matrix. |
| Faugeras, O. (1993). **"Three-Dimensional Computer Vision: A Geometric Viewpoint"**, MIT Press. | Projective geometry fundamentals used in the homography estimation pipeline. |

### Research Papers

| Paper | Venue | Contribution |
|---|---|---|
| Hou, Y., Zheng, L., & Gould, S. (2020). **"Multiview Detection with Feature Perspective Transformation"** | ECCV 2020 | Ground-plane projection of CNN feature maps via homography for multi-view pedestrian detection. 88.2% MODA on Wildtrack. Demonstrated that planar homography is sufficient for pedestrian ground-plane mapping. |
| Hou, Y. & Zheng, L. (2021). **"Multiview Detection with Shadow Transformer"** (MVDeTr) | ACM Multimedia 2021 | Deformable transformer extension of MVDet with deformable attention across multi-view projected features. 91.5% MODA on Wildtrack. |
| Psaltis, A. et al. (2021). **"Tracking Grow-Finish Pigs Across Large Pens Using Multiple Cameras"** (AIFARMS) | CVPR 2021 Workshop on CV4Animals | Production homography-based cross-camera tracking with DeepSORT + YOLOv4. Demonstrated `cv2.findHomography` + `cv2.perspectiveTransform` for mapping bounding boxes between angled and ceiling cameras. |
| Ristani, E. et al. (2016). **"Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking"** | ECCV 2016 Workshop | Defined standard MCMT evaluation metrics (IDF1, IDP, IDR) using world-plane ground truth. Established the DukeMTMC benchmark. |
| Jeon, Y. et al. (2023). **"Leveraging Future Trajectory Prediction for Multi-Camera People Tracking"** (SCIT-MCMT) | CVPR 2023 Workshop | Spatial-temporal cross-camera graph for multi-camera multi-target tracking. Learns cross-camera topology from shared observations. |
| Chen, C. et al. (2023). **"ReST: A Reconfigurable Spatial-Temporal Graph Model for Multi-Camera Multi-Object Tracking"** | ICCV 2023 | Graph-based cross-camera association that learns spatial topology from observations. Reconfigurable structure adapts to changing camera layouts. |
| Fischler, M.A. & Bolles, R.C. (1981). **"Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography"** | Communications of the ACM, 24(6) | The RANSAC algorithm used in `cv2.findHomography` to robustly estimate the homography despite outlier correspondences. |

### Open-Source Implementations

| Repository | Usage |
|---|---|
| [hou-yz/MVDet](https://github.com/hou-yz/MVDet) | Reference for `get_worldcoord_from_imgcoord()` projection utilities and `kornia.warp_perspective()` multi-view feature fusion architecture. |
| [hou-yz/MVDeTr](https://github.com/hou-yz/MVDeTr) | Reference for deformable transformer attention across multi-view projected features. |
| [AIFARMS/multi-camera-pig-tracking](https://github.com/AIFARMS/multi-camera-pig-tracking) | Direct inspiration for the homography-based cross-camera approach. Their `transform_polygon(H, poly)` pattern using `cv2.perspectiveTransform` validated the production viability of this approach. |
| [yuntaeJ/SCIT-MCMT-Tracking](https://github.com/yuntaeJ/SCIT-MCMT-Tracking) | Reference for spatial-temporal cross-camera association graphs. |
| [chengche6230/ReST](https://github.com/chengche6230/ReST) | Reference for reconfigurable spatial-temporal graphs in MCMT. |
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | YOLOv8 detection model used for person detection (nano variant with TensorRT FP16). |
| [levan92/deep_sort_realtime](https://github.com/levan92/deep_sort_realtime) | DeepSORT tracker implementation used as primary tracking backend. |

### Key OpenCV Functions Used

| Function | Purpose |
|---|---|
| [`cv2.findHomography()`](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780) | Estimates the 3×3 ground-plane homography from foot-point correspondences using RANSAC for outlier rejection. |
| [`cv2.perspectiveTransform()`](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#gad327659ac03e5fd6894b90025e6900a7) | Applies the homography to transform point arrays (used internally in projection). |

### Datasets Referenced

| Dataset | Citation |
|---|---|
| **Wildtrack** | Chavdarova, T. et al. (2018). "Wildtrack: A Multi-Camera HD Dataset for Dense Unscripted Pedestrian Detection." CVPR 2018. |
| **MultiviewX** | Hou, Y. et al. (2020). Synthetic multi-view pedestrian detection dataset introduced with MVDet. |
| **DukeMTMC** | Ristani, E. et al. (2016). Multi-camera multi-target tracking benchmark at Duke University. |

---

## 📄 License

This project is **proprietary software**. Copyright © 2024–2026 Mandar Wagh. All rights reserved.

Unauthorized use, copying, modification, or distribution is strictly prohibited. See [LICENSE](LICENSE) for full terms.

---

<p align="center">
  <strong>Built for connected situational awareness</strong> 🎯
</p>
