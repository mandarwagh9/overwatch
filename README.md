# 🎯 OVERWATCH

**Real-time multi-agent collaborative perception system with multi-camera tracking, AI-powered sensor fusion, and augmented reality visualization.**

Built on a Python/FastAPI backend with YOLOv8 person detection, DeepSORT tracking, Kalman-filtered world model, and a React frontend with tactical AR overlays. Designed for edge deployment on NVIDIA Jetson Orin Nano with TensorRT FP16 acceleration.

---

## Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Data Flow Pipeline](#-data-flow-pipeline)
- [Project Structure](#-project-structure)
- [Backend Components](#-backend-components)
- [Frontend Components](#-frontend-components)
- [WebSocket Protocol](#-websocket-protocol)
- [REST API](#-rest-api)
- [Configuration Reference](#-configuration-reference)
- [Installation](#-installation)
- [Deployment](#-deployment)
- [SSL / HTTPS](#-ssl--https)
- [Mobile Camera Streaming](#-mobile-camera-streaming)
- [AR Overlay System](#-ar-overlay-system)
- [World Model & Sensor Fusion](#-world-model--sensor-fusion)
- [Troubleshooting](#-troubleshooting)

---

## 🚀 Features

| Capability | Description |
|---|---|
| **Multi-Camera Processing** | Up to 4 concurrent camera streams (physical + mobile virtual cameras) |
| **Person Detection** | YOLOv8n with COCO class filter (`classes=[0]`) — person-only at NMS level |
| **TensorRT Acceleration** | FP16 `.engine` export for NVIDIA Jetson — sub-10ms inference |
| **Predictive Tracking** | DeepSORT (with MobileNet embedder) or centroid-based SimpleTracker fallback |
| **Sensor Fusion** | 6-state Kalman filter world model with cross-camera object matching |
| **Cross-Camera Predictions** | Objects seen by camera A appear as predicted ghost markers on camera B |
| **AR Visualization** | Canvas-based tactical overlays — cyan detection brackets, yellow tracks, red predictions |
| **Mobile Camera Streaming** | Phone browsers stream camera via WebSocket binary JPEG to the detection pipeline |
| **SSL/TLS** | Self-signed certificates with SAN for LAN IP access (required for `getUserMedia`) |
| **Edge Deployment** | Automated SSH deployment to Jetson Orin Nano via paramiko/SFTP |
| **Binary Protocol** | msgpack serialization for all viewer WebSocket communication |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     OVERWATCH SYSTEM ARCHITECTURE                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  📱 Mobile Phones ──► /ws/camera  (binary JPEG frames over WS)      │
│  📷 IP Cameras    ──► OpenCV VideoCapture (MJPEG/RTSP)              │
│         │                                                            │
│         ▼                                                            │
│  ┌───────────────── FastAPI Backend (port 8000, SSL) ─────────────┐  │
│  │                                                                │  │
│  │  CameraManager          DetectionEngine      TrackingManager   │  │
│  │  ├─ CameraCapture[]     ├─ YOLODetector      ├─ SimpleTracker  │  │
│  │  ├─ VirtualCamera[]     │  ├─ YOLOv8n .pt    │  (centroid)     │  │
│  │  └─ FrameQueue          │  ├─ .engine (TRT)  ├─ DeepSORTTrack  │  │
│  │     (thread-safe)       │  └─ .onnx          │  (MobileNet)    │  │
│  │         │               │  classes=[0]        └─ GlobalRegistry │  │
│  │         └───────────────┴───────┴────────────────────┘          │  │
│  │                              │                                  │  │
│  │                         WorldModel                              │  │
│  │                    ├─ CoordinateTransform                       │  │
│  │                    ├─ KalmanFilter (6-state per object)         │  │
│  │                    ├─ Cross-camera association                  │  │
│  │                    └─ Prediction generation                     │  │
│  │                              │                                  │  │
│  │                     WebSocketManager                            │  │
│  │                    ├─ FrameEncoder (JPEG q=40, 640px max)       │  │
│  │                    └─ msgpack binary serialization               │  │
│  └──────────────────────────┬─────────────────────────────────────┘  │
│                             │ /ws  (msgpack binary frames)           │
│                             ▼                                        │
│  ┌──────────────── React Frontend (port 3000, HTTPS) ─────────────┐  │
│  │  App.jsx ──► AdminDashboard (2×2 camera grid + stats)          │  │
│  │  /mobile ──► MobileCamera.jsx (phone camera streamer)          │  │
│  │  Services: websocket.js, cameraStream.js                       │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow Pipeline

Each viewer connection triggers this per-frame cycle:

```
1. INGEST        CameraCapture thread OR VirtualCamera.inject_frame(jpeg)
                      │
2. DETECT         detection_engine.process_frames() → YOLOv8n (classes=[0])
                      │
3. TRACK          tracking_manager.process_detections() → DeepSORT/SimpleTracker
                      │
4. FUSE           world_model.update_with_tracking_results() → Kalman filter
                      │
5. PREDICT        world_model.generate_predictions_for_camera() → cross-camera ghosts
                      │
6. ENCODE         FrameEncoder.encode_frame() → JPEG (q=40, 640px max)
                      │
7. SERIALIZE      MessageBuilder.build_frame_message() → msgpack
                      │
8. TRANSMIT       WebSocket binary → Frontend
                      │
9. RENDER         CameraDisplay canvas → AR overlays
```

---

## 📁 Project Structure

```
OVERWATCH/
├── README.md
├── certs/                             # SSL certificates (self-signed)
│   ├── cert.pem
│   └── key.pem
│
├── backend/                           # Python FastAPI backend
│   ├── main.py                        # Entry point (196 lines)
│   ├── requirements.txt               # Windows/CPU dependencies
│   ├── requirements-jetson.txt        # Jetson Orin Nano deps
│   ├── yolov8n.pt                     # YOLOv8 nano weights (~6 MB)
│   ├── yolov8n.engine                 # TensorRT FP16 (Jetson, ~8.9 MB)
│   ├── .env                           # Environment config
│   ├── static/
│   │   └── mobile.html                # Standalone mobile camera page (283 lines)
│   └── app/
│       ├── __init__.py
│       ├── config.py                  # Pydantic settings (66 lines)
│       ├── api/
│       │   └── websocket_handler.py   # WS manager (605 lines)
│       └── core/
│           ├── camera_manager.py      # Camera capture (508 lines)
│           ├── detection_engine.py    # YOLOv8 detector (275 lines)
│           ├── tracking_manager.py    # DeepSORT/SimpleTracker (316 lines)
│           └── world_model.py         # Kalman fusion (556 lines)
│
├── frontend/                          # React 18 frontend
│   ├── package.json
│   ├── .env                           # REACT_APP_BACKEND_HOST/PORT
│   ├── public/index.html
│   └── src/
│       ├── index.js
│       ├── App.jsx                    # Admin dashboard (225 lines)
│       ├── App.css                    # Dark tactical theme (360 lines)
│       ├── components/
│       │   └── CameraDisplay.jsx      # AR overlay renderer (348 lines)
│       ├── pages/
│       │   ├── MobileCamera.jsx       # Mobile streaming (184 lines)
│       │   └── MobileCamera.css
│       └── services/
│           ├── websocket.js           # msgpack client (199 lines)
│           └── cameraStream.js        # getUserMedia streamer (349 lines)
│
└── scripts/                           # Deployment utilities
    ├── deploy_jetson.py               # Full SSH deployment (229 lines)
    ├── fix_jetson.py                  # Quick restart helper (63 lines)
    └── ws_test.py                     # CLI WebSocket test (66 lines)
```

---

## ⚙️ Backend Components

### main.py (196 lines)
FastAPI entry point with lifespan management, two WebSocket endpoints (`/ws` for viewers, `/ws/camera` for mobile sources), REST API, and SSL configuration.

### config.py (66 lines)
Pydantic `BaseSettings` loading from `.env`:

| Setting | Default | Description |
|---|---|---|
| `model_path` | `yolov8n.pt` | Model file (.pt / .engine) |
| `detection_classes` | `[0]` | COCO classes to detect (0=person) |
| `device` | `auto` | Compute: auto / cpu / cuda:0 |
| `half_precision` | `False` | FP16 inference (True on Jetson) |
| `max_cameras` | `4` | Maximum camera slots |
| `target_fps` | `24` | Processing FPS target |
| `ssl_enabled` | `True` | Enable HTTPS/WSS |

### detection_engine.py (275 lines)
YOLOv8 wrapper with GPU/TensorRT/CPU auto-detection. Filters to person-only (`classes=[0]`) at NMS level. Skips `.to()` for TensorRT engines (already GPU-bound).

### tracking_manager.py (316 lines)
- **SimpleTracker**: Greedy nearest-centroid (100px threshold), velocity from frame-to-frame displacement
- **DeepSORTTracker**: MobileNet appearance embeddings (if library available)
- Per-camera trackers with global track registry

### world_model.py (556 lines)
- **CoordinateTransform**: Pixel ↔ world projection (simplified pinhole, depth=1.0m)
- **KalmanFilter**: 6-state `[x, y, z, vx, vy, vz]` constant-velocity model
- **Cross-camera fusion**: Euclidean distance < 2m + same class_id
- **Prediction generation**: Extrapolate unseen objects to other camera views

### camera_manager.py (508 lines)
- **CameraCapture**: OpenCV threaded capture with FPS limiting
- **VirtualCamera**: Push-based for mobile browser streams
- **FrameQueue**: Thread-safe, drop-oldest policy

### websocket_handler.py (605 lines)
- **ConnectionManager**: Active connection tracking and broadcast
- **FrameEncoder**: JPEG encoding (q=40, 640px max)
- **MessageBuilder**: Serialize detections/tracks/predictions
- **WebSocketManager**: Orchestrates the per-viewer processing loop

---

## 🖥️ Frontend Components

### App.jsx (225 lines)
Admin dashboard with 2×2 camera grid, stats sidebar, camera start/stop controls. Routes: `/mobile` → MobileCamera, `/*` → AdminDashboard.

### CameraDisplay.jsx (348 lines)
Canvas AR overlay renderer:
- **Detections**: Cyan corner brackets + PERSON pill label
- **Tracks**: Yellow bboxes + velocity arrows + track ID
- **Predictions**: Red dashed boxes + pulsing center dot

### MobileCamera.jsx (184 lines)
Phone camera streaming page with live preview, start/stop, front/rear toggle, FPS stats.

### websocket.js (199 lines)
Singleton msgpack WebSocket client with event emitter and exponential-backoff reconnect.

### cameraStream.js (349 lines)
`getUserMedia` → offscreen canvas → JPEG blob → binary WebSocket to `/ws/camera`.

---

## 📡 WebSocket Protocol

### Endpoints

| Endpoint | Role | Format |
|---|---|---|
| `/ws` | Viewer (admin dashboard) | msgpack binary |
| `/ws/camera` | Camera source (mobile) | Binary JPEG frames |

### Message Types

**Frame Message** (`type: 'frame'`):
```json
{
  "type": "frame",
  "camera_id": 0,
  "timestamp": 1706745600.123,
  "frame_data": "<JPEG bytes>",
  "detections": [{"bbox": [x1,y1,x2,y2], "confidence": 0.87, "class_name": "person"}],
  "tracks": [{"track_id": 1, "bbox": [...], "velocity": [dx, dy]}],
  "predictions": [{"object_id": 1, "bbox": [...], "time_since_seen": 1.2, "inferred": true}]
}
```

**Mobile Registration** (JSON):
```
Client → {"type": "register", "role": "camera_source", "camera_id": null}
Server → {"type": "registered", "camera_id": 0, "target_fps": 15}
```

---

## 🌐 REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/status` | System status (cameras, clients, model) |
| `GET` | `/cameras` | Camera info list |
| `GET` | `/mobile` | Standalone mobile camera HTML page |
| `POST` | `/camera/{id}/start` | Start physical camera |
| `POST` | `/camera/{id}/stop` | Stop camera |

---

## 🔧 Configuration Reference

### Backend `.env`

```bash
MODEL_PATH=yolov8n.engine
DEVICE=cuda:0
HALF_PRECISION=true
DETECTION_CLASSES=[0]
SSL_ENABLED=true
SSL_CERTFILE=certs/cert.pem
SSL_KEYFILE=certs/key.pem
MAX_CAMERAS=4
TARGET_FPS=24
```

### Frontend `.env`

```bash
REACT_APP_BACKEND_HOST=192.168.1.12
REACT_APP_BACKEND_PORT=8000
```

---

## 📦 Installation

### Backend

```bash
cd backend

# Windows / CPU
pip install -r requirements.txt

# Jetson Orin Nano (PyTorch from NVIDIA wheels)
pip install -r requirements-jetson.txt
```

### Frontend

```bash
cd frontend
npm install
```

### SSL Certificates

```bash
mkdir certs
openssl req -x509 -newkey rsa:2048 -keyout certs/key.pem -out certs/cert.pem \
  -days 365 -nodes -subj "/CN=overwatch" \
  -addext "subjectAltName=IP:192.168.1.12,IP:127.0.0.1,DNS:localhost"
```

---

## 🚀 Deployment

### Development (Windows/CPU)

```bash
# Terminal 1
cd backend && python main.py

# Terminal 2
cd frontend && npm start
```

### Production (Jetson Orin Nano)

```bash
# Export TensorRT engine
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='engine', half=True, imgsz=640)"

# Start backend
nohup python3 main.py > /tmp/overwatch.log 2>&1 &
```

### Automated SSH Deployment

```bash
python scripts/deploy_jetson.py
```

Handles: SSH → upload files → install deps → export TensorRT → create .env → start backend.

---

## 🔒 SSL / HTTPS

Required for mobile camera `getUserMedia()` over LAN. Self-signed certs with SAN entries for LAN IPs. Accept certificate warning by visiting `https://<ip>:8000` in browser.

---

## 📱 Mobile Camera Streaming

**React App**: `https://<frontend>:3000/mobile`  
**Standalone**: `https://<backend>:8000/mobile`

Both use: `getUserMedia` → canvas → JPEG blob → binary WebSocket → VirtualCamera → detection pipeline.

---

## 🎯 AR Overlay System

| Layer | Style | Elements |
|---|---|---|
| Detections | Cyan `#00ffc8` | Corner brackets, PERSON pill, crosshair |
| Tracks | Yellow `#ffff00` | Bounding box, center dot, velocity arrow, track ID |
| Predictions | Red dashed | Ghost box, pulsing center, "X.Xs ago" label |

---

## 🌍 World Model & Sensor Fusion

- **Kalman Filter**: 6-state `[x, y, z, vx, vy, vz]` per object
- **Cross-camera matching**: Euclidean < 2m, same class, 100ms recency gate
- **Prediction horizon**: 5 seconds max dead-reckoning
- **Object cleanup**: Remove if unseen > 5 seconds

---

## 🐛 Troubleshooting

### WebSocket won't connect
- Accept self-signed cert at `https://<ip>:8000`
- Check `REACT_APP_BACKEND_HOST` in frontend `.env`

### Mobile camera not working
- Requires HTTPS (SSL enabled)
- Same LAN as backend
- Allow camera permission when prompted

### Jetson issues
```bash
ssh mandar@192.168.1.12 'tail -100 /tmp/overwatch.log'
python scripts/fix_jetson.py
```

### Pydantic "Config and model_config" error
- Use only `model_config = {...}` dict, not inner `class Config`

### TensorRT `.to()` error
- `.engine` files are GPU-bound — `detection_engine.py` skips `.to()` for them

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [DeepSORT](https://github.com/levan92/deep_sort_realtime)
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://react.dev/)
- [Three.js](https://threejs.org/) (installed for future 3D tactical map)

---

**Built for connected situational awareness** 🎯
