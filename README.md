<p align="center">
  <img src="https://img.shields.io/badge/OVERWATCH-v2.0.0-00ffc8?style=for-the-badge&labelColor=0a0a0a" alt="Version" />
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/react-18-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React" />
  <img src="https://img.shields.io/badge/TensorRT-FP16-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="TensorRT" />
  <img src="https://img.shields.io/badge/Jetson_Orin_Nano-Edge-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="Jetson" />
  <img src="https://github.com/mandarwagh9/overwatch/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI" />
  <img src="https://img.shields.io/badge/license-MIT-00ffc8?style=for-the-badge" alt="License: MIT" />
</p>

<h1 align="center">🎯 OVERWATCH</h1>

<p align="center">
  <strong>An open, hackable take on connected-warfare-style perception — running on a $500 dev kit.</strong><br/>
  <em>Multi-sensor fusion · Cross-camera tracking · Tactical AR HUD · Edge inference on Jetson Orin Nano</em>
</p>

<p align="center">
  Inspired by <a href="https://www.anduril.com/connected-warfare">Anduril Connected Warfare</a> &amp; the Lattice OS concept —<br/>
  built as a community reference implementation, not affiliated with Anduril Industries.
</p>

<p align="center">
  <a href="#demo-video">Demo</a> ·
  <a href="#-inspiration--scope">Inspiration</a> ·
  <a href="#-features">Features</a> ·
  <a href="#-architecture">Architecture</a> ·
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-deployment">Deployment</a> ·
  <a href="#-testing">Testing</a> ·
  <a href="#-api-reference">API</a> ·
  <a href="#-troubleshooting">Troubleshooting</a>
</p>

---

## Demo Video

[Watch on YouTube](https://youtu.be/L_jDzPQBXO8) — prototype iteration 1.

---

## 🛰️ Inspiration & scope

OVERWATCH is a **publicly-available reference implementation** of the multi-sensor situational-awareness concept popularised by [Anduril's Connected Warfare](https://www.anduril.com/connected-warfare) and its **Lattice** software platform — the idea that a network of low-cost, heterogeneous sensors can be fused at the edge into a single, AI-driven view of the battlespace.

This project takes that idea and runs with it on commodity hardware:

- A **$500 NVIDIA Jetson Orin Nano** instead of a hardened tactical server
- **IP webcams + mobile phone cameras** instead of dedicated military-grade sensors
- **YOLOv8 + Kalman + homography** instead of classified perception stacks
- **A FastAPI/React stack** instead of proprietary tactical software

The visual language — diamond IFF markers, compass ribbon, threat rings, ghost predictions — is **inspired by Anduril's EagleEye HUD aesthetic** (one of the publicly-shown UI surfaces of Lattice). It is *not* a clone, *not* affiliated with or endorsed by Anduril Industries, and *not* a substitute for their products. Trademarks belong to their respective owners.

> **Scope honesty:** this is a research/educational project. It demonstrates the *principles* of connected sensing — sensor fusion, cross-camera re-ID, edge inference, real-time broadcast — at a scale that fits in a backpack. It is not military-grade, not C2-system-grade, and not certified for any operational use.

### What's the same idea, what's different

| | Anduril Lattice / Connected Warfare | OVERWATCH (this repo) |
|---|---|---|
| **Goal** | Unified situational awareness across heterogeneous sensors | Same — at hobbyist scale |
| **Sensor mix** | Cameras, radar, RF, sonar, drones, ground vehicles, … | IP cameras + phone cameras (extensible) |
| **Fusion** | Proprietary, classified | Open: Kalman + Hungarian + homography |
| **Edge compute** | Hardened tactical hardware | Jetson Orin Nano dev kit |
| **HUD style** | EagleEye tactical UI | EagleEye-*inspired* canvas overlay |
| **Autonomy** | Multi-asset autonomous teaming | Single-pipeline perception only |
| **Use** | Defense / national security | Research, learning, civilian situational awareness |
| **Source** | Closed | Public on GitHub (license: see [LICENSE](LICENSE)) |

If you're building something in this space — researchers, students, civilian defense-tech tinkerers, public-safety folks — this repo is meant to be a starting point you can fork, hack on, and learn from.

---

## Overview

OVERWATCH is a real-time multi-camera situational awareness platform built for edge deployment on NVIDIA Jetson Orin Nano. It fuses video from IP cameras and mobile phones into a unified world model using YOLOv8 detection, Hungarian-assignment tracking, adaptive Kalman filtering, and cross-camera appearance re-identification — all at TensorRT FP16 speeds.

The system runs a **singleton perception pipeline**: detection, tracking, and fusion execute **once per tick** regardless of how many viewers are connected, then broadcast pre-serialized snapshots to all clients over binary WebSocket.

**1 camera + 10 viewers = 1 GPU inference**, not 10.

---

## ✅ Implementation status

OVERWATCH ships a working perception core today. Several headline capabilities are
**designed and on the roadmap but not yet wired into the live pipeline** — this table is
the source of truth. Rows marked 🔭 are *planned*, not implemented; the sections below
that describe them are forward-looking design.

| Capability | Status |
|---|---|
| YOLOv8 person detection (TensorRT / ONNX / PyTorch) | ✅ Implemented |
| Per-camera Hungarian / IoU tracking | ✅ Implemented |
| 6-state Kalman world fusion (confidence-adaptive) | ✅ Implemented |
| Cross-camera merge by world-space distance | ✅ Implemented |
| World-projection ghost predictions (orange `WORLD`) | ✅ Implemented |
| msgpack zero-copy WebSocket broadcast | ✅ Implemented |
| Canvas tactical HUD (brackets, diamonds, velocity, ghosts) | ✅ Implemented |
| Mobile phone camera source + standalone page | ✅ Implemented |
| Optional JWT auth · SSL · atomic Jetson deploy | ✅ Implemented |
| Cross-camera homography ghosts (green `H-PROJ`) | ✅ Implemented |
| Pixel-extrapolation ghosts (red `EXTRAP`) | 🔭 Planned |
| Appearance re-ID (HSV histograms) wired into tracking / fusion | ✅ Implemented |
| Sensor-trust scoring | 🔭 Planned |
| Adaptive Kalman noise by bbox area | 🔭 Planned |
| GPS + IMU fusion into the world model | 🔭 Planned |
| DeepSORT / centroid tracker fallback chain | 🔭 Planned |

> Config and helper scaffolding for the 🔭 items already exists (`compute_appearance()`,
> the `HOMOGRAPHY_*` / `GPS_REFERENCE_*` env vars), but those paths are **not yet active**
> in the pipeline. The build order is tracked in [`docs/superpowers/specs/`](docs/superpowers/specs/).

---

## 🚀 Features

### Core perception

| Capability | Implementation |
|---|---|
| **Person detection** | YOLOv8n with NMS-level class filter (`classes=[0]`) — person-only |
| **TensorRT FP16** | `.engine` export on Jetson — ~8 MiB, sub-10 ms inference |
| **Hungarian tracking** | `scipy.optimize.linear_sum_assignment` — cost `0.6 × IoU + 0.4 × cosine appearance`; appearance descriptors are now computed per detection, so the appearance term is active |
| **Tracker fallback chain** 🔭 | *Planned* — only Hungarian (with a greedy fallback) is active today; DeepSORT/Centroid are not implemented |
| **Adaptive Kalman filter** | 6-state `[x, y, z, vx, vy, vz]` — measurement noise scales by **confidence** (bbox-area & sensor-trust scaling 🔭 planned) |
| **Cross-camera re-ID** | 64-dim HSV histogram descriptors, L2-normalized, EMA-smoothed (α = 0.3); computed per detection and used to gate cross-camera association |
| **Sensor trust scoring** 🔭 | *Planned* — per-sensor trust ∈ [0.1, 1.0]; the Kalman update accepts the param but it is fixed at 1.0 today |
| **Cross-camera homography** | Self-calibrating ground-plane H from shared foot-point observations via `cv2.findHomography` + RANSAC; projects Path-A green `H-PROJ` ghosts |
| **Ghost predictions** | Path A (homography/green `H-PROJ`) and Path C (world projection/orange `WORLD`) are **active**; Path B (pixel extrapolation/red `EXTRAP`) is 🔭 planned |

### Platform

| Capability | Implementation |
|---|---|
| **Multi-camera** | Up to 4 concurrent streams (physical MJPEG/RTSP + mobile virtual cameras) |
| **Mobile streaming** | Phone browsers → `getUserMedia` → binary JPEG over WebSocket → `VirtualCamera` |
| **GPS + IMU fusion** 🔭 | *Planned* — the mobile client sends GPS (`watchPosition`) + IMU (`DeviceOrientationEvent`), but the backend currently receives and discards `sensor_data`; fusion into the world model is not wired yet |
| **AR overlays** | Canvas-based: cyan detection brackets, amber track boxes, green/orange/red ghost predictions |
| **Binary protocol** | msgpack-serialized snapshots — zero-copy broadcast to all viewers |
| **SSL/TLS** | Self-signed certificates with SAN for LAN IP access (required for `getUserMedia`) |
| **Optional JWT auth** | Default-off; enable with `AUTH_ENABLED=true`. Token issuance via `POST /api/token`; WS endpoints accept `?token=...` query param |
| **Edge deployment** | Automated SSH/SFTP deployment to Jetson Orin Nano via paramiko, with atomic staging swap and `--rollback` |

---

## 🏗️ Architecture

```
                          ┌─────────────────────────────────┐
                          │       OVERWATCH  v2.0.0         │
                          └─────────────────────────────────┘

  ╔═══════════════╗       ╔═══════════════════════════════════════════════════╗
  ║  DATA SOURCES ║       ║          JETSON ORIN NANO  (backend :8000)        ║
  ╠═══════════════╣       ╠═══════════════════════════════════════════════════╣
  ║               ║       ║                                                   ║
  ║  📷 IP Camera ─────────►  CameraCapture (OpenCV, MJPEG/RTSP)              ║
  ║               ║       ║       │                                           ║
  ║  📱 Mobile    ─────────►  VirtualCamera (binary JPEG push)                ║
  ║   Phone       ║ws/cam ║       │         + GPS/IMU sensor data             ║
  ║               ║       ║       ▼                                           ║
  ║               ║       ║  ┌──────────────────────────────────────────┐     ║
  ║               ║       ║  │     PerceptionPipeline  (singleton)      │     ║
  ║               ║       ║  │                                          │     ║
  ║               ║       ║  │  1. DETECT   YOLOv8n TensorRT FP16       │     ║
  ║               ║       ║  │              + HSV appearance features   │     ║
  ║               ║       ║  │              │                           │     ║
  ║               ║       ║  │  2. TRACK    Hungarian assignment        │     ║
  ║               ║       ║  │              IoU + cosine appearance     │     ║
  ║               ║       ║  │              │                           │     ║
  ║               ║       ║  │  3. FUSE     Adaptive Kalman 6-state     │     ║
  ║               ║       ║  │              Cross-camera matching       │     ║
  ║               ║       ║  │              Sensor trust scoring        │     ║
  ║               ║       ║  │              │                           │     ║
  ║               ║       ║  │  4. SNAPSHOT Pre-serialized msgpack      │     ║
  ║               ║       ║  └──────────────┬───────────────────────────┘     ║
  ║               ║       ║                 │                                 ║
  ╚═══════════════╝       ║                 ▼  broadcast                      ║
                          ║     WebSocketManager (/ws, msgpack binary)        ║
                          ║         │           │           │                 ║
                          ╚═════════╪═══════════╪═══════════╪═════════════════╝
                                    │           │           │
                          ┌─────────▼──┐  ┌─────▼──┐  ┌─────▼─────┐
                          │  Viewer 1  │  │Viewer 2│  │ Viewer N  │
                          │  React     │  │  React │  │  React    │
                          │  AR Canvas │  │  ...   │  │  ...      │
                          └────────────┘  └────────┘  └───────────┘
```

### Pipeline design

OVERWATCH runs a **single shared pipeline** rather than per-viewer. The `PerceptionPipeline` singleton executes detect → track → fuse **once per tick**, produces a `PerceptionSnapshot` with pre-serialized msgpack packets, and all connected viewers read from the latest snapshot.

- 1 camera + 10 viewers = 1 GPU inference
- Zero-copy broadcast via pre-serialized binary packets
- Slow viewers gracefully skip intermediate frames (per-client 2 s send timeout)

---

## 📁 Project structure

```
OVERWATCH/
├── backend/                              # FastAPI + perception engine
│   ├── main.py                           # App entry, lifespan, REST + WS endpoints
│   ├── requirements.txt                  # Python deps (CPU/Windows dev)
│   ├── requirements-jetson.txt           # Jetson Orin Nano deps (pinned)
│   ├── .env.example                      # Config template
│   └── app/
│       ├── domain/entities.py            # Detection, Track, WorldObject, ...
│       ├── application/
│       │   ├── ports.py                  # Repository interfaces
│       │   └── services.py               # PerceptionPipelineService
│       └── infrastructure/
│           ├── auth.py                   # Optional JWT verify/issue
│           ├── camera_adapter.py         # OpenCV capture + virtual cameras
│           ├── config_adapter.py         # Pydantic settings
│           ├── container.py              # DI container
│           ├── detection_adapter.py      # YOLO wrapper
│           ├── frame_encoder_adapter.py  # JPEG encode
│           ├── tracking_adapter.py       # Hungarian + DeepSORT
│           ├── websocket_adapter.py      # msgpack broadcast
│           └── world_model_adapter.py    # Kalman fusion + homography
│   └── tests/
│       ├── conftest.py                   # Shared fixtures
│       └── unit/                         # 57 unit tests
│
├── frontend/                             # React 18 admin dashboard
│   ├── package.json
│   └── src/
│       ├── pages/
│       │   ├── AdminDashboard.jsx        # Main camera grid
│       │   └── MobileCamera.jsx          # Phone camera streaming UI
│       ├── components/
│       │   ├── CameraDisplay.jsx         # Canvas AR overlay renderer
│       │   ├── ErrorBoundary.jsx         # Top-level error fallback
│       │   ├── StatsPanel.jsx
│       │   └── ConnectionStatus.jsx
│       ├── application/hooks/            # useCameraData, useWebSocket, useSystemStats
│       └── infrastructure/               # websocketAdapter, cameraStreamAdapter, apiAdapter
│
├── scripts/                              # Deployment & ops
│   ├── _jetson_common.py                 # Shared SSH/SFTP helper (env-driven creds)
│   ├── deploy_jetson.py                  # Atomic deploy with --rollback
│   ├── restart_jetson.py                 # Quick backend restart
│   ├── check_logs.py / check_status.py
│   ├── ws_test.py
│   └── archive/                          # Retired/duplicate scripts (reference only)
│
├── certs/                                # SSL certificates (gitignored)
├── .github/workflows/ci.yml              # GitHub Actions test runner
├── pyproject.toml                        # pytest + project metadata
└── README.md
```

---

## ⚡ Quick start

### Prerequisites

- **Python 3.10+** with pip
- **Node.js 18+** with npm
- **NVIDIA Jetson Orin Nano** for production, or any machine with CUDA for development

### 1. Clone

```bash
git clone https://github.com/mandarwagh9/overwatch.git
cd overwatch
```

### 2. Local development

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

### 3. Single-binary mode (frontend served by backend)

```bash
cd frontend && npm install && npm run build && cd ..
cd backend && python main.py
```

Backend serves both the React app and the API at **https://localhost:8000**.

---

## 🚀 Deployment

### Deploy to Jetson

Credentials are read from environment — never hardcoded:

```bash
export JETSON_HOST=192.168.1.10        # default if unset
export JETSON_USER=mandar              # default if unset
export JETSON_PASS=...                 # or use JETSON_KEY=/path/to/id_rsa
python scripts/deploy_jetson.py
```

The script:
1. Connects via SSH/SFTP using paramiko
2. Uploads backend, frontend build, certs to `<remote>.new/` (staging)
3. Atomically swaps `<remote>.new` → `<remote>`, keeping previous version at `<remote>.bak`
4. Generates a fresh `JWT_SECRET` and writes a `chmod 600` `.env`
5. Installs Python dependencies
6. Starts the backend

### Rollback

```bash
python scripts/deploy_jetson.py --rollback
```

Swaps the last `.bak` directory back into place. Use after a bad deploy.

### Quick operations

```bash
# Restart backend without redeploying
python scripts/restart_jetson.py

# Tail logs
python scripts/check_logs.py

# Quick status
python scripts/check_status.py
```

### Access (replace with your `JETSON_HOST`)

| Service | URL |
|---|---|
| Admin Dashboard | `https://<jetson-host>:8000` |
| Mobile Camera (standalone) | `https://<jetson-host>:8000/mobile` |

---

## 🧪 Testing

**Backend** — 57 unit tests covering domain primitives, Kalman filtering, coordinate transforms, tracking, and configuration:

```bash
python -m pytest backend/tests/unit -v
```

Tests are pure-Python and do not require CUDA, ultralytics, or torch (heavy deps are import-skipped).

**Frontend** — Jest + React Testing Library smoke tests for domain helpers and components:

```bash
cd frontend && npm run test:ci
```

**CI** runs the full gate on every push and PR — a backend job (`ruff` + `mypy` + `pytest` with a coverage floor) and a frontend job (`jest`/RTL + `eslint` via the production build).

---

## 📡 API reference

### REST endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the React app (or returns API status if no build present) |
| `GET` | `/health` | Detailed health status |
| `GET` | `/status` | System status (cameras, clients, detection engine, pipeline metrics) |
| `GET` | `/cameras` | Active camera list |
| `POST` | `/cameras/{id}/start` | Start a physical camera |
| `POST` | `/cameras/{id}/stop` | Stop a camera |
| `POST` | `/api/token` | Issue a JWT (only when `AUTH_ENABLED=true`) |

### WebSocket endpoints

| Endpoint | Direction | Format | Purpose |
|---|---|---|---|
| `/ws` | Server → client | msgpack binary | Viewer stream (frames + detections + tracks + predictions) |
| `/ws/camera` | Client → server | Binary JPEG + JSON | Mobile camera source |

When `AUTH_ENABLED=true`, both endpoints require a `?token=<jwt>` query parameter; unauthorized connections close with 1008.

### Mobile registration handshake

```
Client → { "type": "register", "role": "camera_source", "camera_id": null }
Server → { "type": "registered", "camera_id": 0 }
Client → [binary JPEG frames]
Client → { "type": "sensor_data", "gps": {...}, "orientation": {...} }   # received, not yet fused 🔭
```

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
| `IOU_THRESHOLD` | `0.45` | NMS IoU threshold |
| `TARGET_FPS` | `24` | Processing framerate target |
| `MAX_CAMERAS` | `4` | Maximum concurrent camera streams |
| `TRACKING_MAX_AGE` | `30` | Max frames to keep lost tracks |
| `TRACKING_MIN_HITS` | `3` | Min hits to confirm a track |
| `TRACKING_IOU_THRESHOLD` | `0.25` | IoU threshold for tracking |
| `MOBILE_CAMERA_FPS` | `15` | Mobile camera target FPS |
| `MOBILE_CAMERA_MAX_WIDTH` | `640` | Mobile camera max width |
| `SSL_ENABLED` | `true` | Enable HTTPS/WSS |
| `SSL_CERTFILE` | `certs/cert.pem` | SSL certificate path |
| `SSL_KEYFILE` | `certs/key.pem` | SSL private key path |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |
| `AUTH_ENABLED` | `false` | Require JWT on WS + REST when `true` |
| `JWT_SECRET` | _(empty)_ | HS256 signing key — required when `AUTH_ENABLED=true` |
| `CORS_ORIGINS` | `["*"]` | JSON list of allowed origins |
| `MAX_WS_CLIENTS` | `100` | Hard cap on concurrent viewer WebSocket connections |

### Frontend `.env`

| Variable | Default | Description |
|---|---|---|
| `REACT_APP_BACKEND_HOST` | `window.location.hostname` | Backend IP or hostname |
| `REACT_APP_BACKEND_PORT` | `8000` | Backend port |
| `REACT_APP_BACKEND_PROTOCOL` | `wss` (https) / `ws` (http) | WebSocket protocol |
| `REACT_APP_MAX_CAMERAS` | `4` | Maximum cameras to display |
| `REACT_APP_CAMERA_INACTIVITY_TIMEOUT` | `3000` | ms before marking camera offline |
| `REACT_APP_MOBILE_TARGET_FPS` | `15` | Mobile streaming FPS |
| `REACT_APP_MOBILE_JPEG_QUALITY` | `0.5` | Mobile JPEG quality (0–1) |
| `REACT_APP_MOBILE_MAX_WIDTH` | `640` | Mobile frame width |

### Deploy environment

| Variable | Default | Description |
|---|---|---|
| `JETSON_HOST` | `192.168.1.10` | Jetson IP or hostname |
| `JETSON_USER` | `mandar` | SSH user |
| `JETSON_PASS` | _(prompt)_ | SSH password — fallback to `getpass` if unset |
| `JETSON_KEY` | _(unset)_ | Path to private key (preferred over password) |

---

## 🎯 AR overlay system — EagleEye-inspired tactical HUD

The frontend renders a tactical HUD inspired by Anduril's EagleEye UI, implemented entirely in HTML5 Canvas — corner brackets, diamond IFF markers, velocity vectors, ghost chevrons, and HUD corner ticks. (Compass ribbon and threat ring are 🔭 planned, not in the current renderer. Visual style only; rendered from open code, no Anduril assets used.)

| Layer | Color | Elements |
|---|---|---|
| **Detections** | Slate-blue `#64b5f6` | Diamond markers, corner brackets, `PERSON` confidence pill |
| **Tracks** | Amber `#ffd740` | Diamond/chevron markers, velocity vector arrows, track ID callouts |
| **Predictions (H-PROJ)** | Green `#00ff82` solid | ✅ Active — homography-projected ghost; emitted once a camera-pair homography has been learned |
| **Predictions (EXTRAP)** 🔭 | Red `#ff5050` dashed | *Planned* — pixel-extrapolated ghost (renderer-ready; not emitted by the backend yet) |
| **Predictions (WORLD)** | Orange `#ff9800` dashed | ✅ Active — world-coordinate pinhole projection; the only ghost path emitted today |
| **Compass ribbon** 🔭 | — | *Planned* — not in the current canvas renderer |
| **Threat ring** 🔭 | Per-IFF color | *Planned* — not in the current canvas renderer |

Detection overlays show what the model sees *right now*. Track overlays show persistent identity across frames. Predictions show cross-camera projections — green `H-PROJ` (homography, most accurate) and orange `WORLD` (world-model fallback) are emitted today; red `EXTRAP` (pixel extrapolation) is 🔭 planned.

---

## 🌍 World model & sensor fusion

### Kalman filter

Each fused world object maintains a 6-state Kalman filter `[x, y, z, vx, vy, vz]` with constant-velocity dynamics. Measurement noise **R** adapts per-update based on detection confidence, bounding box area, and sensor trust — higher-quality observations tighten the filter, while noisy or untrusted sensors widen it. `dt` is clamped to `≥ 0` to defend against cross-camera clock skew.

### Cross-camera association

Objects from different cameras are matched when:
- Euclidean distance < 2 m
- Same `class_id`
- Appearance cosine similarity ≥ 0.5 when both observations carry a descriptor (differently-dressed people at the same ground position stay separate)

### Sensor trust 🔭 *(planned)*

The intended design earns per-sensor trust through consistency:
- **Consistent measurements** → trust increases (capped at 1.0)
- **Innovation outliers** → trust decays (floored at 0.1)

Today the Kalman update accepts a `sensor_trust` argument but it is fixed at `1.0`.

### Appearance re-ID

- 64-dimensional HSV histogram descriptors computed per detection (~0.1 ms each)
- L2-normalized for cosine similarity
- Exponential moving average (α = 0.3) on the fused world object for descriptor stability

Used to gate cross-camera association: two people at the same ground position but with
different appearance are kept as distinct world objects.

---

## 📐 Cross-camera homography — how it works

> **Status: implemented** (`app/infrastructure/homography.py`). The estimator self-calibrates
> from foot-point correspondences and projects Path-A green `H-PROJ` ghosts, falling back to
> Path C (world projection) when no homography has been learned for a camera pair yet.

The signature feature is **homography ghost prediction**: when Camera 0 can't see a person but Camera 1 can, the system renders a ghost overlay on Camera 0's feed showing where that person is.

### The problem with naive extrapolation

Sliding a person's last-known pixel position forward in time fails within seconds because:
- Different cameras have completely different pixel coordinate systems
- The mapping between camera views is a **projective transformation**, not a linear offset
- A person at pixel `(400, 300)` in Camera 1 might correspond to `(800, 500)` in Camera 0

### The solution: learn the camera-to-camera transform

When both cameras simultaneously observe the **same person** (matched via appearance re-ID), the system records **foot-point correspondence pairs** — the bottom-center of the bounding box in each view. These foot points project to the same physical ground-plane location.

With ≥ 4 such pairs, `cv2.findHomography()` + RANSAC computes a 3×3 **homography matrix** $H$ that maps any ground-plane point from one camera's pixel space to another's:

$$\begin{pmatrix} x' \\ y' \\ w \end{pmatrix} = H \cdot \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}$$

### Self-calibrating pipeline

1. **Collect**: when re-ID matches a person across Camera 0 and Camera 1, record `(foot_cam0, foot_cam1)` pair
2. **Estimate**: after 4+ pairs, compute $H_{0\to 1}$ and $H_{1\to 0}$ via RANSAC (re-estimated every 5 new pairs)
3. **Project**: when Camera 0 loses a person but Camera 1 still sees them, apply $H_{1\to 0}$ to Camera 1's current foot point → position on Camera 0's feed
4. **Validate**: monitor reprojection error; if it spikes (camera moved), flush and re-learn

### Computational cost

- Homography estimation: **< 0.1 ms** (called every 5 new pairs, not every frame)
- Per-prediction projection: **< 0.001 ms** (one 3×3 matrix multiply)
- Total overhead per frame: effectively zero on Jetson Orin Nano

### Visual indicators

| Ghost color | Tag | Source | Meaning |
|---|---|---|---|
| 🟢 Green solid | `H-PROJ` | Path A — homography | Cross-camera ground-plane projection. Tries all source cameras with valid $H$ to the target, picks the freshest. Most accurate. |
| 🟠 Orange dashed | `WORLD` | Path C — world projection | Fused 3D world position → pinhole camera model. Rough but **always works** even when no homography exists and the target camera has never seen the person. |
| 🔴 Red dashed | `EXTRAP` | Path B — pixel extrapolation | Slides last-known pixel position by velocity × time. Adaptive budget: `min(250 px, 80 + 40 × t)`. Only works if the target camera previously saw the person. |

---

## 📱 Mobile camera streaming

Any phone on the same LAN can become a camera source:

- **Via React app**: `https://<frontend-ip>:3000/mobile`
- **Standalone page**: `https://<jetson-ip>:8000/mobile`

The mobile client:
1. Opens the rear camera via `getUserMedia` (1280×720)
2. Renders to an offscreen canvas, extracts a JPEG blob
3. Sends binary frames over WebSocket to `/ws/camera`
4. Captures GPS (`watchPosition`, high-accuracy) and IMU (`DeviceOrientationEvent`) at 2 Hz
5. Sends sensor data as JSON for camera calibration fusion

> `getUserMedia` requires HTTPS — this is why SSL certificates are mandatory even on LAN.

---

## ⚠️ Edge cases & known limitations

### Cross-camera prediction

> Path A (homography) and Path C (world projection) are active; rows mentioning Path B
> (pixel extrapolation) describe 🔭 **planned** behavior (see the
> [Implementation status](#-implementation-status) table).

| Edge case | Behavior | Mitigation |
|---|---|---|
| **No homography learned yet** | Path A fails silently; falls through to Path B (extrap) or Path C (world projection). Ghost appears orange instead of green. | Walk through overlapping camera FOVs to collect ≥ 4 foot-point pairs. Homography auto-learns within ~5 s of co-visibility. |
| **Camera moved after calibration** | Reprojection error spikes; stale $H$ produces offset ghosts. | The system monitors error and flushes the homography when it exceeds 50 px. Walk through overlap again to re-learn. |
| **Person only seen by one camera ever** | Path A has no source to project from; Path B has no pixel history for the target. Path C is the only option. | Path C accuracy depends on calibrated camera positions in the `CoordinateTransformer` (set via `CAMERA_POSITIONS` env). |
| **Cameras with no overlapping FOV** | No co-visible observations → no foot-point pairs → no homography. Path A never activates between these cameras. | Path C still works. For better accuracy, set `CAMERA_POSITIONS` to your physical camera extrinsics. |

### Tracking & re-ID

| Edge case | Behavior | Mitigation |
|---|---|---|
| **Identical clothing** | HSV histograms are nearly identical; re-ID may merge two people into one world object. | The system uses spatial distance (< 2 m) AND appearance similarity (> 0.5 cosine). If two people are spatially separated, they stay separate even with identical appearance. |
| **Person temporarily fully occluded** | Track coasts for `prediction_horizon` seconds (default 5 s); confidence decays linearly. After timeout, the track is pruned. | Increase `prediction_horizon` if longer persistence is needed. Kalman velocity keeps the ghost moving during occlusion. |
| **Crowded scenes (> 10 people)** | Hungarian cost matrix grows as `O(n × m)`; appearance feature extraction adds ~0.1 ms per detection. | Throughput may drop below target FPS. YOLOv8n NMS already limits detections. |
| **Person enters from off-screen** | No pixel history, no world object yet. First detection creates a new track with high measurement noise. | Kalman initializes with large uncertainty; trust builds over 5–10 consistent frames. |

### Sensor fusion

| Edge case | Behavior | Mitigation |
|---|---|---|
| **Mobile GPS jitter indoors** | GPS accuracy can be 10–50 m indoors; Kalman receives noisy position updates. | Sensor trust scoring down-weights high-innovation sources. Trust floor (0.1) prevents complete rejection. |
| **Mobile phone loses WebSocket** | Virtual camera stream stops; existing tracks coast via Kalman prediction. | Tracks persist for `prediction_horizon`. Phone auto-reconnects (with intentional-close handling) and gets a new camera ID. |
| **Clock drift between cameras** | Frame timestamps may not be synchronized. Co-visibility matching uses a 0.5 s window. | The 0.5 s window is generous for typical LAN latency. NTP sync is recommended for sub-100 ms accuracy. The Kalman `dt` is clamped to ≥ 0 so negative skew can't corrupt covariance. |

### Network & deployment

| Edge case | Behavior | Mitigation |
|---|---|---|
| **Self-signed cert rejected by browser** | WebSocket connection fails silently; frontend shows no feeds. | Visit `https://<jetson-ip>:8000` directly and accept the certificate (once per browser session). |
| **Jetson runs out of GPU memory** | TensorRT engine uses ~30 MiB. With 4 cameras at 640×640, CUDA memory ≈ 200 MiB total. Orin Nano has 8 GB shared. | Monitor with `tegrastats`. Reduce `MAX_CAMERAS` or input resolution if tight. |
| **Backend crash** | Run via `nohup` in deploy script; no auto-restart. | Add a systemd unit with `Restart=always` for production. `python scripts/restart_jetson.py` brings it back manually. |
| **Many viewers cause lag** | The singleton pipeline runs once per tick regardless of viewers, but msgpack serialize + send scales linearly. Per-client send timeout is 2 s. | Pre-serialized snapshots minimize per-viewer cost. For > 10 viewers, consider a pub/sub layer (Redis, NATS). |
| **Mid-deploy network drop** | Atomic SFTP staging means previous version stays at `<remote>` until the swap. | If a deploy partially fails, run `python scripts/deploy_jetson.py --rollback` to restore the last `.bak`. |

---

## 🐛 Troubleshooting

<details>
<summary><strong>WebSocket won't connect</strong></summary>

1. Visit `https://<jetson-ip>:8000` in your browser and accept the self-signed certificate
2. Verify `REACT_APP_BACKEND_HOST` in `frontend/.env` matches the backend IP
3. Check the backend is running: `curl -sk https://<jetson-ip>:8000/health`
4. If `AUTH_ENABLED=true`, ensure the client supplies `?token=<jwt>` from `POST /api/token`
</details>

<details>
<summary><strong>Mobile camera shows black screen</strong></summary>

- HTTPS is required for `getUserMedia` — ensure `SSL_ENABLED=true`
- Phone must be on the same LAN as the backend
- Allow camera permission when the browser prompts
- Try the standalone page: `https://<jetson-ip>:8000/mobile`
</details>

<details>
<summary><strong>Port already in use on Jetson</strong></summary>

```bash
python scripts/restart_jetson.py
# Or manually over SSH:
JETSON_HOST=192.168.1.10 JETSON_PASS=... ssh "$JETSON_USER@$JETSON_HOST" \
  'pkill -9 -f "python3 main.py"; sleep 2; cd /home/$USER/overwatch/backend && nohup python3 main.py > /tmp/overwatch.log 2>&1 &'
```
</details>

<details>
<summary><strong>Checking Jetson logs</strong></summary>

```bash
python scripts/check_logs.py        # Tails the last 50 lines
python scripts/check_status.py      # Backend status snapshot
```
</details>

<details>
<summary><strong>Bad deploy — roll back</strong></summary>

```bash
python scripts/deploy_jetson.py --rollback
```

Swaps the last `<remote>.bak` directory back into place.
</details>

<details>
<summary><strong>Ghost predictions not appearing on a camera</strong></summary>

1. **Check homography status** — look for `H learned: cam0→cam1` in logs. If missing, walk through both camera FOVs simultaneously to collect correspondence pairs.
2. **Check world projection** — Path C (orange ghost) should always work if camera positions are configured. If missing, verify the `CoordinateTransformer` calibration matches your physical setup.
3. **Check prediction horizon** — if `time_since_seen > prediction_horizon` (default 5 s), the object is pruned. The person must be actively tracked by at least one camera.
4. **Check `source_tracks`** — if Camera 1 is currently tracking the person, no prediction is generated for it (live track, not a ghost).
</details>

<details>
<summary><strong>Ghosts flicker between green and orange</strong></summary>

The homography is borderline — sometimes projection succeeds (green), sometimes it fails and falls through to Path C (orange):

- Homography learned from too few correspondence pairs (minimum 4, but 8+ is more stable)
- Person is near the edge of the overlap zone where reprojection error is highest
- Walk more paths through the camera overlap to improve $H$ stability
</details>

<details>
<summary><strong>Two people merged into one ghost</strong></summary>

Cross-camera re-ID matched two different people as the same world object. This happens with:
- Identical clothing (same HSV histogram)
- People standing < 2 m apart in world coordinates
- Temporary occlusion causing track ID swap

The system self-corrects once the people separate spatially. The appearance descriptor EMA (α = 0.3) gradually diverges.
</details>

<details>
<summary><strong>Pydantic <code>Config</code> error</strong></summary>

Use only the `model_config = SettingsConfigDict(...)` dict pattern — do not define an inner `class Config`. This is the Pydantic v2 convention.
</details>

---

## 🧰 Tech stack

| Layer | Technology |
|---|---|
| **Detection** | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (nano) |
| **Inference** | NVIDIA TensorRT FP16 / ONNX Runtime / PyTorch |
| **Tracking** | [DeepSORT](https://github.com/levan92/deep_sort_realtime) / Hungarian (scipy) / Centroid |
| **Fusion** | Custom 6-state Kalman filter with adaptive noise |
| **Cross-camera** | Ground-plane homography via [OpenCV `findHomography`](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780) + RANSAC |
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn (ASGI) |
| **Protocol** | [msgpack](https://msgpack.org/) binary over WebSocket |
| **Frontend** | [React 18](https://react.dev/) + Canvas 2D API |
| **Auth (optional)** | [PyJWT](https://pyjwt.readthedocs.io/) (HS256) |
| **Hardware** | NVIDIA Jetson Orin Nano (JetPack 6.x, R36) |
| **Deployment** | [paramiko](https://www.paramiko.org/) SSH/SFTP automation |
| **Tests + CI** | pytest + GitHub Actions |

---

## 📚 References & sources

The cross-camera homography system is built on established multi-view geometry principles and inspired by several academic works and open-source implementations.

### Foundational theory

| Source | Relevance |
|---|---|
| Hartley, R. & Zisserman, A. (2004). **Multiple View Geometry in Computer Vision**, 2nd ed. Cambridge University Press. | Chapter 13: ground-plane homography between uncalibrated camera pairs. |
| Faugeras, O. (1993). **Three-Dimensional Computer Vision: A Geometric Viewpoint**, MIT Press. | Projective geometry fundamentals used in the homography estimation pipeline. |

### Research papers

| Paper | Venue | Contribution |
|---|---|---|
| Hou, Y., Zheng, L., & Gould, S. (2020). **Multiview Detection with Feature Perspective Transformation** | ECCV 2020 | Ground-plane projection of CNN feature maps via homography for multi-view pedestrian detection. 88.2% MODA on Wildtrack. |
| Hou, Y. & Zheng, L. (2021). **MVDeTr: Multiview Detection with Shadow Transformer** | ACM MM 2021 | Deformable transformer extension of MVDet. 91.5% MODA on Wildtrack. |
| Psaltis, A. et al. (2021). **Tracking Grow-Finish Pigs Across Large Pens Using Multiple Cameras** | CVPR 2021 Workshop | Production homography-based cross-camera tracking with DeepSORT + YOLOv4. |
| Ristani, E. et al. (2016). **Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking** | ECCV 2016 Workshop | Defined IDF1, IDP, IDR metrics. Established the DukeMTMC benchmark. |
| Jeon, Y. et al. (2023). **Leveraging Future Trajectory Prediction for Multi-Camera People Tracking** | CVPR 2023 Workshop | Spatial-temporal cross-camera graph for MCMT. |
| Chen, C. et al. (2023). **ReST: A Reconfigurable Spatial-Temporal Graph Model for MCMT** | ICCV 2023 | Graph-based cross-camera association that learns spatial topology from observations. |
| Fischler, M.A. & Bolles, R.C. (1981). **Random Sample Consensus** | CACM 24(6) | The RANSAC algorithm used in `cv2.findHomography`. |

### Open-source implementations

| Repository | Usage |
|---|---|
| [hou-yz/MVDet](https://github.com/hou-yz/MVDet) | Reference for `get_worldcoord_from_imgcoord()` and multi-view feature fusion. |
| [hou-yz/MVDeTr](https://github.com/hou-yz/MVDeTr) | Reference for deformable transformer attention across multi-view projected features. |
| [AIFARMS/multi-camera-pig-tracking](https://github.com/AIFARMS/multi-camera-pig-tracking) | Direct inspiration for the homography-based cross-camera approach. |
| [yuntaeJ/SCIT-MCMT-Tracking](https://github.com/yuntaeJ/SCIT-MCMT-Tracking) | Reference for spatial-temporal cross-camera association graphs. |
| [chengche6230/ReST](https://github.com/chengche6230/ReST) | Reference for reconfigurable spatial-temporal graphs in MCMT. |
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | YOLOv8 detection model. |
| [levan92/deep_sort_realtime](https://github.com/levan92/deep_sort_realtime) | DeepSORT tracker implementation. |

### Datasets referenced

| Dataset | Citation |
|---|---|
| **Wildtrack** | Chavdarova, T. et al. (2018). CVPR. |
| **MultiviewX** | Hou, Y. et al. (2020). Synthetic multi-view pedestrian dataset introduced with MVDet. |
| **DukeMTMC** | Ristani, E. et al. (2016). |

---

## 📄 License & trademarks

OVERWATCH is released under the **[MIT License](LICENSE)** — copyright © 2024–2026 Mandar Wagh. You're free to use, copy, modify, merge, publish, distribute, sublicense, and sell copies of the software, subject to the conditions in the license file. If you build something cool with it, a link back is appreciated but not required.

**"Anduril," "Lattice," "Connected Warfare," and "EagleEye"** are trademarks of Anduril Industries, Inc. This project is an independent community implementation **inspired by publicly-shown concepts** of those products. It is not affiliated with, endorsed by, or sponsored by Anduril Industries. No proprietary information, code, or assets from Anduril are used.

All other third-party trademarks (NVIDIA, Jetson, TensorRT, React, FastAPI, etc.) belong to their respective owners.

---

<p align="center">
  <strong>Connected sensing, at hackathon scale.</strong> 🎯<br/>
  <em>Inspired by <a href="https://www.anduril.com/connected-warfare">Anduril Connected Warfare</a> · Built on open tools.</em>
</p>
