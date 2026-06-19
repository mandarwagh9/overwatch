# Overwatch Backend - Clean Architecture

This is a complete rewrite of the Overwatch backend following **Clean Architecture** principles.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Interface Layer                          │
│  (FastAPI, WebSocket handlers, API routes)                  │
├─────────────────────────────────────────────────────────────┤
│                  Application Layer                          │
│  (Services, Use Cases, Ports/Interfaces)                    │
├─────────────────────────────────────────────────────────────┤
│                    Domain Layer                             │
│  (Entities, Value Objects, Domain Logic)                    │
├─────────────────────────────────────────────────────────────┤
│                 Infrastructure Layer                        │
│  (Adapters, External Dependencies, Frameworks)              │
└─────────────────────────────────────────────────────────────┘
```

## Key Improvements

### 1. **No Hardcoded Values**
- All camera URLs come from configuration
- All camera positions come from configuration
- No hardcoded IP addresses or credentials

### 2. **Dependency Injection**
- All dependencies injected through constructors
- No global state or singletons
- Proper separation of concerns

### 3. **Clean Architecture**
- Domain layer has ZERO external dependencies
- Application layer defines ports (interfaces)
- Infrastructure layer implements adapters

### 4. **Proper Error Handling**
- Custom domain exceptions
- No silent failures
- Structured logging throughout

### 5. **Type Safety**
- Full type hints throughout
- Pydantic validation for all config
- No `Any` types in domain

### 6. **No Temporary Solutions**
- No mock detectors - requires ultralytics
- No simple tracker fallbacks - proper Hungarian algorithm
- No monkey-patching - clean class design

## Project Structure

```
backend/
├── app/
│   ├── domain/               # Pure business logic
│   │   └── entities.py       # Domain entities and value objects
│   │
│   ├── application/          # Use cases and ports
│   │   ├── ports.py          # Repository interfaces
│   │   └── services.py       # Application services
│   │
│   ├── infrastructure/       # Adapters and frameworks
│   │   ├── config_adapter.py        # Pydantic settings
│   │   ├── camera_adapter.py        # OpenCV camera capture
│   │   ├── detection_adapter.py     # YOLO detection
│   │   ├── tracking_adapter.py      # Hungarian tracking
│   │   ├── world_model_adapter.py   # Sensor fusion
│   │   ├── frame_encoder_adapter.py # JPEG encoding
│   │   ├── websocket_adapter.py     # WebSocket communication
│   │   └── container.py             # DI container
│   │
│   └── __init__.py
│
├── main.py                   # FastAPI application entry point
├── requirements.txt          # Dependencies
└── .env                      # Configuration (create from .env.example)
```

## Configuration

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

### Required Configuration

**Camera URLs** (at least one):
```env
CAMERA_URLS=["http://192.168.1.100:8080/video", "http://192.168.1.101:8080/video"]
# OR
CAM_0_URL=0  # Local camera device
CAM_1_URL=http://192.168.1.100:8080/video
```

**Camera Positions** (recommended for accurate world coordinates; auto-defaulted if omitted):
```env
CAMERA_POSITIONS=[[0, 0, 2], [5, 0, 2], [0, 5, 2]]
```
> If omitted, the world model synthesizes a default calibration per camera (spread along
> the x-axis) so it still produces world objects out of the box, logging a one-time warning.
> Set real positions for accurate cross-camera coordinates.

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your camera URLs and positions

# Run
python main.py
```

## API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health status
- `GET /status` - System status
- `GET /cameras` - List cameras
- `POST /cameras/{id}/start` - Start camera
- `POST /cameras/{id}/stop` - Stop camera
- `WS /ws` - Main WebSocket for real-time data
- `WS /ws/camera` - Mobile camera streaming

## Migration from Old Code

The old files have been replaced:
- `app/config.py` → `app/infrastructure/config_adapter.py`
- `app/core/camera_manager.py` → `app/infrastructure/camera_adapter.py`
- `app/core/detection_engine.py` → `app/infrastructure/detection_adapter.py`
- `app/core/tracking_manager.py` → `app/infrastructure/tracking_adapter.py`
- `app/core/world_model.py` → `app/infrastructure/world_model_adapter.py`
- `app/core/perception_pipeline.py` → `app/application/services.py`
- `app/api/websocket_handler.py` → `app/infrastructure/websocket_adapter.py`

## Testing

```bash
# Run tests
pytest

# Type checking
mypy app/

# Formatting
black app/
```
