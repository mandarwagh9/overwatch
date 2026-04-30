"""
Main FastAPI application.
Clean entry point with proper dependency injection.
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.infrastructure.container import ApplicationContainer, create_container
from app.infrastructure.config_adapter import get_settings


# Configure logging
def setup_logging() -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Reduce uvicorn logging noise
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


setup_logging()
logger = logging.getLogger(__name__)

# Global container (initialized in lifespan)
container: Optional[ApplicationContainer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global container
    
    logger.info("🚀 Starting Overwatch system...")
    
    # Create and start container
    container = create_container()
    await container.start()
    
    host = container.config.get("host", "0.0.0.0")
    port = container.config.get_int("port", 8000)
    logger.info(f"✅ Overwatch ready - listening on {host}:{port}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Overwatch system...")
    if container:
        await container.stop()
    logger.info("✅ Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Overwatch API",
    description="Multi-camera perception and tracking system",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
_settings_for_cors = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings_for_cors.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static file serving for frontend (after container is initialized)
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "build")
static_exists = os.path.exists(static_dir)

if static_exists:
    logger.info(f"📁 Serving static files from: {static_dir}")
    app.mount("/static", StaticFiles(directory=os.path.join(static_dir, "static")), name="static")
else:
    logger.warning(f"⚠️ Frontend build directory not found: {static_dir}")


@app.get("/")
@app.get("/mobile")
async def serve_react():
    """Serve React app if available, otherwise return API status."""
    if static_exists and os.path.exists(os.path.join(static_dir, "index.html")):
        return FileResponse(os.path.join(static_dir, "index.html"))
    
    return {
        "message": "Overwatch API is running",
        "version": "2.0.0",
        "status": "operational",
        "capabilities": [
            "perception_pipeline",
            "multi_camera_tracking",
            "world_model_fusion",
            "predictive_targeting"
        ]
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    if not container:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return container.health_service.get_status()


@app.get("/status")
async def get_status():
    """Get system status."""
    if not container:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Get latest snapshot info
    snapshot = container.pipeline_service.latest_snapshot
    camera_info = []
    
    if snapshot:
        for cam_id, frame in snapshot.camera_frames.items():
            detections = snapshot.detections.get(cam_id, [])
            tracks = snapshot.tracks.get(cam_id, [])
            camera_info.append({
                "camera_id": cam_id,
                "has_frame": frame is not None,
                "frame_shape": frame.frame_data.shape if frame else None,
                "detections": len(detections),
                "tracks": len(tracks)
            })
    
    return {
        "cameras_active": container.camera_service.get_active_camera_count(),
        "connected_clients": container.communication_repo.get_client_count(),
        "detection_engine_ready": container.detection_repo.is_ready(),
        "tracking_active": container.tracking_repo.is_ready(),
        "detection_model": "YOLOv8n",
        "pipeline_metrics": {
            "frames_processed": container.pipeline_service.metrics.frames_processed,
            "average_processing_time_ms": round(
                container.pipeline_service.metrics.average_processing_time_ms, 2
            )
        },
        "snapshot": {
            "has_snapshot": snapshot is not None,
            "generation": snapshot.generation if snapshot else 0,
            "world_objects": len(snapshot.world_objects) if snapshot else 0,
            "cameras": camera_info
        }
    }


@app.post("/cameras/{camera_id}/start")
async def start_camera(camera_id: int):
    """Start a specific camera."""
    if not container:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    success = await container.camera_service.start_camera(camera_id)
    
    if success:
        return {"message": f"Camera {camera_id} started", "camera_id": camera_id}
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to start camera {camera_id}"
        )


@app.post("/cameras/{camera_id}/stop")
async def stop_camera(camera_id: int):
    """Stop a specific camera."""
    if not container:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    success = await container.camera_service.stop_camera(camera_id)
    
    if success:
        return {"message": f"Camera {camera_id} stopped", "camera_id": camera_id}
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to stop camera {camera_id}"
        )


@app.get("/cameras")
async def list_cameras():
    """List all cameras."""
    if not container:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {
        "active_count": container.camera_service.get_active_camera_count(),
        "cameras": container.camera_repo.get_camera_info() if hasattr(container.camera_repo, 'get_camera_info') else []
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time data."""
    if not container:
        await websocket.close(code=1011, reason="System not initialized")
        return

    try:
        connection_id = await container.communication_repo.connect(websocket)
    except RuntimeError as e:
        # Cap exceeded; close already issued by connect()
        logger.warning(f"WebSocket rejected: {e}")
        return

    try:
        # Keep connection alive
        while True:
            try:
                # Wait for client messages (ping/pong or commands)
                message = await websocket.receive_text()
                # Handle any client commands here
            except WebSocketDisconnect:
                break

    except Exception as e:
        logger.error(f"WebSocket error for {connection_id}: {e}")
    finally:
        container.communication_repo.disconnect(connection_id)


@app.websocket("/ws/camera")
async def camera_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for mobile camera sources."""
    if not container:
        await websocket.close(code=1011, reason="System not initialized")
        return
    
    import json
    
    await websocket.accept()

    camera_id: Optional[int] = None

    try:
        # Wait for registration message
        message = await websocket.receive_text()
        reg_data = json.loads(message)
        
        if reg_data.get("type") != "register":
            await websocket.close(code=1008, reason="Expected registration message")
            return
        
        # Register virtual camera
        requested_id = reg_data.get("camera_id")
        camera_id = container.camera_service.register_virtual_camera(requested_id)
        
        if camera_id is None:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "No camera slots available"
            }))
            await websocket.close(code=1008, reason="No slots available")
            return
        
        # Send success response
        await websocket.send_text(json.dumps({
            "type": "registered",
            "camera_id": camera_id
        }))
        
        logger.info(f"Mobile camera registered: {camera_id}")
        
        # Handle incoming frames
        frame_count = 0
        last_frame_time = time.time()
        
        while True:
            try:
                # Use timeout to prevent hanging
                message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                
                if message.get("type") == "websocket.disconnect":
                    logger.info(f"Camera {camera_id}: Client disconnected")
                    break
                
                # Handle binary JPEG frames
                if "bytes" in message:
                    jpeg_bytes = message["bytes"]
                    success = container.camera_service.inject_mobile_frame(
                        camera_id, jpeg_bytes
                    )
                    if success:
                        frame_count += 1
                        last_frame_time = time.time()
                        # Log every 30 frames
                        if frame_count % 30 == 0:
                            logger.info(f"Camera {camera_id}: {frame_count} frames received")
                
                # Handle control messages
                elif "text" in message:
                    control = json.loads(message["text"])
                    msg_type = control.get("type")
                    
                    if msg_type == "stop":
                        logger.info(f"Camera {camera_id}: Received stop command")
                        break
                    elif msg_type == "ping":
                        # Send pong to keep connection alive
                        await websocket.send_text(json.dumps({"type": "pong"}))
                    elif msg_type == "sensor_data":
                        # Handle sensor data if needed
                        pass
                
                # Check for timeout (no frames for 60 seconds)
                if time.time() - last_frame_time > 60:
                    logger.warning(f"Camera {camera_id}: No frames received for 60s, closing connection")
                    break
                    
            except asyncio.TimeoutError:
                logger.warning(f"Camera {camera_id}: Receive timeout, sending ping")
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                except:
                    break
            except WebSocketDisconnect:
                logger.info(f"Camera {camera_id}: WebSocket disconnected")
                break
            except Exception as e:
                logger.error(f"Camera {camera_id}: Error in receive loop: {e}")
                break
        
    except Exception as e:
        logger.error(f"Camera WebSocket error: {e}")
    finally:
        if camera_id is not None:
            container.camera_service.unregister_virtual_camera(camera_id)
            logger.info(f"Mobile camera unregistered: {camera_id}")


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration
    from app.infrastructure.config_adapter import get_settings
    settings = get_settings()
    
    # SSL configuration
    ssl_kwargs = {}
    if settings.ssl_enabled:
        import os
        cert_path = settings.ssl_certfile
        key_path = settings.ssl_keyfile
        
        # Resolve relative to project root
        if not os.path.isabs(cert_path):
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(backend_dir)
            cert_path = os.path.join(project_root, cert_path)
            key_path = os.path.join(project_root, key_path)
        
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
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        ws_max_size=settings.ws_max_size,
        ws_ping_interval=settings.ws_ping_interval,
        ws_ping_timeout=settings.ws_ping_timeout,
        **ssl_kwargs
    )
