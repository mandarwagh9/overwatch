<p align="center">
  <img src="https://img.shields.io/badge/OVERWATCH-v2.0.0-00ffc8?style=for-the-badge&labelColor=0a0a0a" alt="Version" />
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/react-18-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React" />
  <img src="https://img.shields.io/badge/TensorRT-FP16-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="TensorRT" />
  <img src="https://img.shields.io/badge/Jetson_Orin_Nano-Edge-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="Jetson" />
  <img src="https://img.shields.io/badge/license-MIT-blue?style=for-the-badge" alt="License" />
</p>

<h1 align="center">🎯 OVERWATCH</h1>

<p align="center">
  <strong>Real-time multi-agent collaborative perception system</strong><br/>
  <em>Multi-camera tracking · AI-powered sensor fusion · Augmented reality overlays · Edge deployment</em>
</p>

<p align="center">
  <a href="#-features">Features</a> ·
  <a href="#-architecture">Architecture</a> ·
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-deployment">Deployment</a> ·
  <a href="#-api-reference">API Reference</a> ·
  <a href="#-troubleshooting">Troubleshooting</a>
</p>

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
| **Ghost Predictions** | Primary: homography projection (green, real-time). Fallback: pixel extrapolation (red, up to 5s) |

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
│   ├── static/
│   │   └── mobile.html                   # Standalone mobile camera page
│   └── app/
│       ├── config.py                     # Pydantic settings with .env support
│       ├── api/
│       │   └── websocket_handler.py      # WS connection manager, frame encoding
│       └── core/
│           ├── perception_pipeline.py    # Singleton detect→track→fuse loop
│           ├── detection_engine.py       # YOLOv8 wrapper + appearance features
│           ├── tracking_manager.py       # Hungarian / DeepSORT / Simple tracker
│           ├── world_model.py            # Kalman filter, cross-cam fusion, trust
│           └── camera_manager.py         # Physical + virtual camera management
│
├── frontend/                             # React 18 Admin Dashboard
│   ├── package.json
│   ├── .env                              # REACT_APP_BACKEND_HOST / PORT
│   └── src/
│       ├── App.jsx                       # Dashboard layout, WS event handling
│       ├── App.css                       # Dark tactical theme
│       ├── components/
│       │   └── CameraDisplay.jsx         # Canvas AR overlay renderer
│       ├── pages/
│       │   ├── MobileCamera.jsx          # Phone camera streaming UI
│       │   └── MobileCamera.css
│       └── services/
│           ├── websocket.js              # msgpack binary WS client
│           └── cameraStream.js           # getUserMedia → WS + GPS/IMU capture
│
├── scripts/                              # Deployment & Operations
│   ├── deploy_v2.py                      # Full SSH/SFTP deployment to Jetson
│   ├── restart_jetson.py                 # Force-kill and restart backend
│   ├── check_logs.py                     # Verify Jetson logs and imports
│   ├── check_status.py                   # Monitor API health, connections, GPU
│   └── ws_test.py                        # CLI WebSocket test client
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
- **OpenSSL** (for certificate generation)
- NVIDIA Jetson Orin Nano *(production)* or any machine with CUDA *(development)*

### 1. Clone

```bash
git clone https://github.com/mandarwagh9/overwatch.git
cd overwatch
```

### 2. Generate SSL Certificates

```bash
mkdir certs
openssl req -x509 -newkey rsa:2048 -keyout certs/key.pem -out certs/cert.pem \
  -days 365 -nodes -subj "/CN=overwatch" \
  -addext "subjectAltName=IP:192.168.1.12,IP:127.0.0.1,DNS:localhost"
```

### 3. Backend Setup

```bash
cd backend
pip install -r requirements.txt        # CPU/Windows
# pip install -r requirements-jetson.txt  # Jetson Orin Nano
```

Create `backend/.env`:

```env
MODEL_PATH=yolov8n.pt
DEVICE=auto
HALF_PRECISION=false
DETECTION_CLASSES=[0]
SSL_ENABLED=true
HOST=0.0.0.0
PORT=8000
```

### 4. Frontend Setup

```bash
cd frontend
npm install
```

Create `frontend/.env`:

```env
REACT_APP_BACKEND_HOST=192.168.1.12
REACT_APP_BACKEND_PORT=8000
```

### 5. Run

```bash
# Terminal 1 — Backend
cd backend && python main.py

# Terminal 2 — Frontend
cd frontend && npm start
```

Open **https://localhost:3001** — accept the self-signed certificate warning.

---

## 🚀 Deployment

### Jetson Orin Nano (Production)

#### Automated Deployment

```bash
python scripts/deploy_v2.py
```

This script handles the full lifecycle via SSH:
1. Kills any existing backend process
2. Uploads all 8 backend files via SFTP
3. Installs `scipy` and `PyJWT` dependencies
4. Verifies all imports
5. Starts the backend with `nohup`
6. Validates health check returns v2.0.0

#### Manual Deployment

```bash
# On Jetson — Export TensorRT engine (one-time)
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='engine', half=True, imgsz=640)"

# Create .env
cat > .env << 'EOF'
MODEL_PATH=yolov8n.engine
DEVICE=cuda:0
HALF_PRECISION=true
DETECTION_CLASSES=[0]
SSL_ENABLED=true
HOST=0.0.0.0
PORT=8000
EOF

# Start
nohup python3 main.py > /tmp/overwatch.log 2>&1 &
```

#### Operations Scripts

| Script | Purpose |
|---|---|
| `scripts/deploy_v2.py` | Full deployment (upload, deps, restart, verify) |
| `scripts/restart_jetson.py` | Force-kill all processes and restart |
| `scripts/check_logs.py` | Read startup logs, verify scipy + pipeline imports |
| `scripts/check_status.py` | API health, active WebSocket connections, GPU stats |

---

## 📡 API Reference

### REST Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check — returns version, status, capabilities |
| `GET` | `/status` | System status (cameras, clients, model info) |
| `GET` | `/cameras` | Active camera list |
| `GET` | `/mobile` | Standalone mobile camera HTML page |
| `POST` | `/camera/{id}/start` | Start a physical camera |
| `POST` | `/camera/{id}/stop` | Stop a camera |
| `POST` | `/api/token` | Issue JWT authentication token |

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
| `SSL_ENABLED` | `true` | Enable HTTPS/WSS |
| `SSL_CERTFILE` | `certs/cert.pem` | SSL certificate path |
| `SSL_KEYFILE` | `certs/key.pem` | SSL private key path |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |
| `MAX_CAMERAS` | `4` | Maximum concurrent camera streams |
| `TARGET_FPS` | `24` | Processing framerate target |
| `JWT_SECRET` | `overwatch-secret` | JWT signing secret |
| `JWT_ALGORITHM` | `HS256` | JWT algorithm |
| `JWT_EXPIRE_MINUTES` | `60` | Token expiry |
| `AUTH_ENABLED` | `false` | Enforce JWT on WebSocket connections |
| `TRUST_DECAY_RATE` | `0.01` | Per-tick trust decay for sensors |
| `TRUST_MIN` | `0.1` | Minimum sensor trust floor |

### Frontend `.env`

| Variable | Description |
|---|---|
| `REACT_APP_BACKEND_HOST` | Jetson/backend IP address |
| `REACT_APP_BACKEND_PORT` | Backend port (default `8000`) |

---

## 🎯 AR Overlay System

The frontend renders three distinct visualization layers on a canvas overlay:

| Layer | Color | Elements |
|---|---|---|
| **Detections** | Cyan `#00ffc8` | Corner brackets, center crosshair, `PERSON` confidence pill with pointer |
| **Tracks** | Yellow `#ffff00` | Bounding box, center dot, velocity vector arrow with arrowhead, track ID label |
| **Predictions** | Red `#ff4444` dashed | Ghost bounding box, pulsing center dot (sine animation), confidence + time-ago label |

Detection overlays show what the model sees *right now*. Track overlays show persistent identity across frames. Prediction overlays show cross-camera dead-reckoning — objects last seen by another camera, projected into the current view.

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
cd /home/mandar/OVERWATCH/backend && nohup python3 main.py > /tmp/overwatch.log 2>&1 &
```
</details>

<details>
<summary><strong>Checking Jetson logs</strong></summary>

```bash
# From your development machine
python scripts/check_logs.py
python scripts/check_status.py

# Or via SSH
ssh mandar@192.168.1.12 'tail -50 /tmp/overwatch.log'
```
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

| Ghost Color | Source | Meaning |
|---|---|---|
| 🟢 Green solid box | `H-PROJ` | Homography projection — **real-time, accurate** |
| 🔴 Red dashed box | `EXTRAP` | Pixel extrapolation — **time-decaying guess** |

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

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  <strong>Built for connected situational awareness</strong> 🎯
</p>
