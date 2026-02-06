"""
Main FastAPI application entry point for Overwatch system
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Try to import uvloop (Unix/Linux only)
try:
    import uvloop
    UVLOOP_AVAILABLE = True
except ImportError:
    UVLOOP_AVAILABLE = False
    print("⚠️ uvloop not available (Windows system), using default asyncio event loop")

from app.config import settings
from app.core.camera_manager import CameraManager
from app.core.detection_engine import DetectionEngine
from app.core.tracking_manager import TrackingManager
from app.core.world_model import WorldModel


# Global managers (WebSocketManager imported after to avoid circular import)
camera_manager = CameraManager()
detection_engine = DetectionEngine()
tracking_manager = TrackingManager()
world_model = WorldModel()
websocket_manager = None
pipeline = None  # PerceptionPipeline (shared singleton)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global websocket_manager, pipeline
    
    # Import WebSocketManager after globals are defined
    from app.api.websocket_handler import WebSocketManager
    from app.core.perception_pipeline import PerceptionPipeline
    websocket_manager = WebSocketManager()
    
    # Set manager references
    websocket_manager.set_managers(camera_manager, detection_engine, tracking_manager, world_model)
    
    # Startup
    print("🚀 Starting Overwatch system...")
    
    # Initialize detection engine
    await detection_engine.initialize()
    print("✅ Detection engine initialized")
    
    # Initialize tracking manager
    await tracking_manager.initialize()
    print("✅ Tracking manager initialized")
    
    # Initialize world model
    await world_model.initialize()
    print("✅ World model initialized")
    
    # Start camera manager
    await camera_manager.start()
    print("✅ Camera manager started")
    
    # Start the shared perception pipeline (detect→track→fuse runs ONCE per tick)
    pipeline = PerceptionPipeline(camera_manager, detection_engine, tracking_manager, world_model)
    websocket_manager.set_pipeline(pipeline)
    await pipeline.start()
    print("✅ Perception pipeline started")
    
    print(f"🎯 Overwatch ready - listening on {settings.host}:{settings.port}")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Overwatch system...")
    await pipeline.stop()
    await camera_manager.stop()
    await detection_engine.cleanup()
    print("✅ Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Overwatch API",
    description="Connected situational awareness system with multi-camera tracking",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware — allow LAN origins for mobile camera access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (mobile devices on LAN need this)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Overwatch API is running",
        "version": "2.0.0",
        "status": "operational",
        "capabilities": [
            "perception_pipeline",
            "hungarian_tracking",
            "adaptive_kalman",
            "appearance_reid",
            "gps_imu_fusion",
            "world_update_broadcast",
        ]
    }


@app.post("/api/token")
async def create_token(node_id: str = "anonymous", role: str = "viewer"):
    """Issue a JWT for WebSocket authentication.
    
    In production, this should require credentials.
    Currently issues tokens for any node_id/role for development.
    """
    import jwt
    from datetime import datetime, timedelta
    
    payload = {
        "node_id": node_id,
        "role": role,  # "viewer" or "camera_source"
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes),
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return {"token": token, "expires_in": settings.jwt_expire_minutes * 60}


@app.get("/status")
async def get_status():
    """Get system status"""
    return {
        "cameras_active": camera_manager.get_active_camera_count(),
        "max_cameras": settings.max_cameras,
        "target_fps": settings.target_fps,
        "connected_clients": websocket_manager.get_client_count(),
        "detection_model": "YOLOv8n",
        "tracking_active": tracking_manager.is_active()
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time communication (viewer clients)"""
    await websocket_manager.connect(websocket)
    try:
        # Start processing loop for this client
        await websocket_manager.handle_client_loop(websocket)
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket)


@app.websocket("/ws/camera")
async def websocket_camera_endpoint(websocket: WebSocket):
    """WebSocket endpoint for mobile camera sources.
    
    Mobile clients connect here, send a register message, then stream
    binary JPEG frames. The frames are injected into the detection pipeline.
    """
    camera_id_str = await websocket_manager.connect_camera_source(websocket)
    
    if camera_id_str is None:
        return  # Registration failed, connection already closed
    
    camera_id = int(camera_id_str)
    
    try:
        await websocket_manager.handle_camera_source_loop(websocket, camera_id)
    except WebSocketDisconnect:
        websocket_manager.camera_manager.unregister_virtual_camera(camera_id)
    except Exception as e:
        print(f"Mobile camera WebSocket error: {e}")
        websocket_manager.camera_manager.unregister_virtual_camera(camera_id)


@app.post("/camera/{camera_id}/start")
async def start_camera(camera_id: int):
    """Start a specific camera stream"""
    if camera_id >= settings.max_cameras:
        return {"error": f"Camera ID must be less than {settings.max_cameras}"}
    
    success = await camera_manager.start_camera(camera_id)
    if success:
        return {"message": f"Camera {camera_id} started successfully"}
    else:
        return {"error": f"Failed to start camera {camera_id}"}


@app.post("/camera/{camera_id}/stop")
async def stop_camera(camera_id: int):
    """Stop a specific camera stream"""
    success = await camera_manager.stop_camera(camera_id)
    if success:
        return {"message": f"Camera {camera_id} stopped successfully"}
    else:
        return {"error": f"Failed to stop camera {camera_id}"}


@app.get("/cameras")
async def get_cameras():
    """Get information about all cameras"""
    return camera_manager.get_camera_info()


@app.get("/mobile")
async def mobile_camera_page():
    """Serve the mobile camera streaming page.
    
    This is a standalone HTML page that mobile users open to stream
    their camera to the Overwatch backend.
    """
    mobile_html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "mobile.html")
    if os.path.exists(mobile_html_path):
        with open(mobile_html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="<h1>Mobile page not found. Build frontend first.</h1>", status_code=404)


if __name__ == "__main__":
    import uvicorn
    
    # Use uvloop for better performance on Linux/macOS (if available)
    if UVLOOP_AVAILABLE and sys.platform != 'win32':
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    
    # SSL configuration
    ssl_kwargs = {}
    if settings.ssl_enabled:
        # Look for certs relative to the project root (one level up from backend/)
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(backend_dir)
        
        cert_path = os.path.join(project_root, settings.ssl_certfile)
        key_path = os.path.join(project_root, settings.ssl_keyfile)
        
        if os.path.exists(cert_path) and os.path.exists(key_path):
            ssl_kwargs["ssl_certfile"] = cert_path
            ssl_kwargs["ssl_keyfile"] = key_path
            print(f"🔒 SSL enabled: cert={cert_path}")
        else:
            print(f"⚠️ SSL cert/key not found at {cert_path}, {key_path} — running without SSL")
            print("   Mobile camera getUserMedia will NOT work over plain HTTP on LAN IPs.")
    
    # Run the server
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        ws_max_size=settings.ws_max_size,
        ws_ping_interval=settings.ws_ping_interval,
        ws_ping_timeout=settings.ws_ping_timeout,
        loop="auto",  # Let uvicorn choose the best loop
        **ssl_kwargs
    )