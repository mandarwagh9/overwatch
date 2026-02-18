# Overwatch Frontend - Clean Architecture

This is a complete rewrite of the Overwatch frontend following **Clean Architecture** principles.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  Presentation Layer                         │
│  (React Components, Pages, CSS)                            │
├─────────────────────────────────────────────────────────────┤
│                  Application Layer                          │
│  (Custom Hooks, State Management)                          │
├─────────────────────────────────────────────────────────────┤
│                  Infrastructure Layer                       │
│  (WebSocket Adapter, API Adapter, Camera Stream)          │
├─────────────────────────────────────────────────────────────┤
│                    Domain Layer                             │
│  (Entities, Types, Constants)                              │
├─────────────────────────────────────────────────────────────┤
│                  Configuration Layer                        │
│  (Environment-based Configuration)                         │
└─────────────────────────────────────────────────────────────┘
```

## Key Improvements

### 1. **No Hardcoded Values**
- All configuration centralized in `src/config/`
- Backend URLs computed from environment variables
- No hardcoded IP addresses or magic numbers
- All constants in domain layer

### 2. **Layer Separation**
- Domain: Pure entities and business logic
- Application: Hooks and state management
- Infrastructure: External dependencies (WebSocket, API, Camera)
- Presentation: React components

### 3. **Custom Hooks**
- `useWebSocket`: WebSocket connection management
- `useCameraData`: Camera feed state management
- `useCameraControl`: Camera start/stop operations
- `useSystemStats`: System statistics fetching

### 4. **Clean Adapters**
- `WebSocketAdapter`: Manages WebSocket lifecycle
- `ApiAdapter`: HTTP API client with error handling
- `CameraStreamAdapter`: Mobile camera streaming

### 5. **Configuration Management**
- Environment-based configuration
- Validation at startup
- Sensible defaults
- No proxy configuration mismatches

## Project Structure

```
src/
├── config/                    # Configuration management
│   └── index.js              # Config with validation
│
├── domain/                    # Domain layer
│   └── entities.js           # Types, constants, entities
│
├── application/               # Application layer
│   └── hooks/                # Custom React hooks
│       ├── useWebSocket.js
│       ├── useCameraData.js
│       ├── useCameraControl.js
│       └── useSystemStats.js
│
├── infrastructure/            # Infrastructure layer
│   ├── websocketAdapter.js   # WebSocket client
│   ├── apiAdapter.js         # HTTP API client
│   └── cameraStreamAdapter.js # Camera streaming
│
├── components/                # Presentation components
│   ├── CameraDisplay.jsx     # Camera feed with overlays
│   ├── StatsPanel.jsx        # Statistics panel
│   ├── ConnectionStatus.jsx  # Connection indicator
│   └── ErrorBanner.jsx       # Error display
│
├── pages/                     # Page components
│   ├── AdminDashboard.jsx    # Main dashboard
│   └── MobileCamera.jsx      # Mobile camera page
│
├── App.jsx                    # Main app component
├── index.js                   # Entry point
└── index.css                  # Global styles
```

## Configuration

Create a `.env` file in the frontend directory:

```env
# Backend Configuration
REACT_APP_BACKEND_HOST=localhost
REACT_APP_BACKEND_PORT=8000
REACT_APP_BACKEND_PROTOCOL=ws
REACT_APP_API_PROTOCOL=http

# WebSocket Configuration
REACT_APP_WS_MAX_RECONNECT=5
REACT_APP_WS_RECONNECT_DELAY=1000

# Camera Configuration
REACT_APP_MAX_CAMERAS=4
REACT_APP_CAMERA_INACTIVITY_TIMEOUT=3000

# Mobile Camera Configuration
REACT_APP_MOBILE_TARGET_FPS=15
REACT_APP_MOBILE_JPEG_QUALITY=0.5
REACT_APP_MOBILE_MAX_WIDTH=640
```

## Running

```bash
# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm build
```

## Removed Dependencies

The following unused dependencies have been removed:
- `@react-three/fiber` (Three.js was never used)
- `@react-three/drei` (Three.js was never used)
- `three` (Three.js was never used)

## Features

### Admin Dashboard (`/`)
- Real-time camera grid (configurable number of cameras)
- Tactical HUD overlays (EagleEye-style)
- System statistics panel
- Camera start/stop controls
- FPS monitoring
- World object tracking
- Detection and tracking visualization

### Mobile Camera (`/mobile`)
- Stream phone camera to backend
- Front/rear camera switching
- GPS and orientation sensor fusion
- Real-time streaming stats

## API Endpoints Used

- `WS /ws` - Main WebSocket for real-time data
- `WS /ws/camera` - Mobile camera streaming
- `GET /health` - Health check
- `GET /status` - System status
- `GET /cameras` - List cameras
- `POST /cameras/{id}/start` - Start camera
- `POST /cameras/{id}/stop` - Stop camera

## Migration from Old Code

The old files have been replaced:
- `services/websocket.js` → `infrastructure/websocketAdapter.js`
- `services/cameraStream.js` → `infrastructure/cameraStreamAdapter.js`
- `App.jsx` (old) → `pages/AdminDashboard.jsx` + `App.jsx` (new)
- `components/CameraDisplay.jsx` (old) → Clean version with separation of concerns

## Development

### Code Style
- Functional components with hooks
- Custom hooks for stateful logic
- Adapter pattern for external dependencies
- Domain-driven design

### Testing
```bash
# Run tests
npm test

# Build for production
npm run build
```
