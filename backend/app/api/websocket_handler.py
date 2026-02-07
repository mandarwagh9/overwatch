"""
WebSocket handler for real-time communication with msgpack serialization.

Supports two connection roles:
- "viewer": Admin dashboard clients that receive processed frames (default, existing behavior)
- "camera_source": Mobile browser clients that push camera frames via binary WebSocket messages
"""

import asyncio
import time
import json
import msgpack
import cv2
import numpy as np
from typing import Dict, List, Set, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect

from app.config import settings
from app.core.camera_manager import CameraFrame
from app.core.detection_engine import DetectionResult
from app.core.tracking_manager import TrackingResult
from app.core.world_model import PredictedTarget


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_stats: Dict[str, dict] = {}
        self.next_connection_id = 1
    
    async def connect(self, websocket: WebSocket) -> str:
        """Accept a new WebSocket connection"""
        await websocket.accept()
        connection_id = f"client_{self.next_connection_id}"
        self.next_connection_id += 1
        
        self.active_connections[connection_id] = websocket
        self.connection_stats[connection_id] = {
            'connected_at': time.time(),
            'messages_sent': 0,
            'messages_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'last_activity': time.time()
        }
        
        print(f"🔌 Client {connection_id} connected")
        return connection_id
    
    def disconnect(self, connection_id: str):
        """Remove a WebSocket connection"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            del self.connection_stats[connection_id]
            print(f"🔌 Client {connection_id} disconnected")
    
    async def send_to_client(self, connection_id: str, data: bytes):
        """Send data to a specific client"""
        if connection_id not in self.active_connections:
            return False
        
        try:
            websocket = self.active_connections[connection_id]
            await websocket.send_bytes(data)
            
            # Update statistics
            stats = self.connection_stats[connection_id]
            stats['messages_sent'] += 1
            stats['bytes_sent'] += len(data)
            stats['last_activity'] = time.time()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to send to {connection_id}: {e}")
            self.disconnect(connection_id)
            return False
    
    async def broadcast(self, data: bytes, exclude: Optional[Set[str]] = None):
        """Broadcast data to all connected clients"""
        if not self.active_connections:
            return
        
        exclude = exclude or set()
        
        # Send to all clients concurrently
        send_tasks = []
        for connection_id in list(self.active_connections.keys()):
            if connection_id not in exclude:
                task = self.send_to_client(connection_id, data)
                send_tasks.append(task)
        
        if send_tasks:
            await asyncio.gather(*send_tasks, return_exceptions=True)
    
    async def receive_from_client(self, connection_id: str, websocket: WebSocket) -> Optional[dict]:
        """Receive and parse message from client"""
        try:
            # Try to receive binary data (msgpack)
            data = await websocket.receive_bytes()
            message = msgpack.unpackb(data, raw=False)
            
            # Update statistics
            if connection_id in self.connection_stats:
                stats = self.connection_stats[connection_id]
                stats['messages_received'] += 1
                stats['bytes_received'] += len(data)
                stats['last_activity'] = time.time()
            
            return message
            
        except ValueError:
            # Try to receive text data (JSON fallback)
            try:
                text = await websocket.receive_text()
                message = json.loads(text)
                return message
            except Exception as e:
                print(f"❌ Failed to parse message from {connection_id}: {e}")
                return None
                
        except Exception as e:
            print(f"❌ Error receiving from {connection_id}: {e}")
            return None
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    def get_connection_stats(self) -> Dict[str, dict]:
        """Get statistics for all connections"""
        return self.connection_stats.copy()


class FrameEncoder:
    """Handles frame encoding for transmission"""
    
    def __init__(self, quality: int = 40):
        # Lower default JPEG quality to reduce bandwidth (more aggressive)
        self.jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        self.encode_stats = {
            'frames_encoded': 0,
            'total_encode_time': 0,
            'average_encode_time': 0,
            'total_bytes': 0,
            'average_bytes': 0
        }
    
    def encode_frame(self, frame: np.ndarray) -> Optional[bytes]:
        """Encode frame to JPEG bytes"""
        start_time = time.time()
        
        try:
            # Resize frame if too large for transmission
            max_width = 640
            if frame.shape[1] > max_width:
                scale = max_width / frame.shape[1]
                new_width = int(frame.shape[1] * scale)
                new_height = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Encode to JPEG
            success, buffer = cv2.imencode('.jpg', frame, self.jpeg_params)
            
            if success:
                encoded_bytes = buffer.tobytes()
                
                # Update statistics
                encode_time = time.time() - start_time
                self.encode_stats['frames_encoded'] += 1
                self.encode_stats['total_encode_time'] += encode_time
                self.encode_stats['average_encode_time'] = (
                    self.encode_stats['total_encode_time'] / self.encode_stats['frames_encoded']
                )
                self.encode_stats['total_bytes'] += len(encoded_bytes)
                self.encode_stats['average_bytes'] = (
                    self.encode_stats['total_bytes'] / self.encode_stats['frames_encoded']
                )
                
                return encoded_bytes
            
            return None
            
        except Exception as e:
            print(f"❌ Frame encoding error: {e}")
            return None
    
    def get_stats(self) -> dict:
        """Get encoding statistics"""
        return self.encode_stats.copy()


class MessageBuilder:
    """Builds different types of messages for WebSocket transmission"""
    
    @staticmethod
    def build_frame_message(camera_id: int, frame_data: bytes, detections: List[dict], tracks: List[dict], predictions: List[dict]) -> dict:
        """Build a frame message with all associated data"""
        return {
            'type': 'frame',
            'camera_id': camera_id,
            'timestamp': time.time(),
            'frame_data': frame_data,
            'detections': detections,
            'tracks': tracks,
            'predictions': predictions
        }
    
    @staticmethod
    def build_status_message(system_stats: dict) -> dict:
        """Build a system status message"""
        return {
            'type': 'status',
            'timestamp': time.time(),
            'stats': system_stats
        }
    
    @staticmethod
    def build_error_message(error: str, camera_id: Optional[int] = None) -> dict:
        """Build an error message"""
        return {
            'type': 'error',
            'timestamp': time.time(),
            'error': error,
            'camera_id': camera_id
        }
    
    @staticmethod
    def build_prediction_message(camera_id: int, predictions: List[dict]) -> dict:
        """Build a prediction-only message for cameras with no current detections"""
        return {
            'type': 'predictions',
            'camera_id': camera_id,
            'timestamp': time.time(),
            'predictions': predictions
        }
    
    @staticmethod
    def serialize_detection(detection) -> dict:
        """Serialize a Detection object to dict"""
        return {
            'bbox': detection.bbox,
            'confidence': detection.confidence,
            'class_id': detection.class_id,
            'class_name': detection.class_name,
            'center': detection.center
        }
    
    @staticmethod
    def serialize_track(track) -> dict:
        """Serialize a Track object to dict"""
        return {
            'track_id': track.track_id,
            'bbox': track.bbox,
            'center': track.center,
            'confidence': track.confidence,
            'class_id': track.class_id,
            'class_name': track.class_name,
            'age': track.age,
            'hits': track.hits,
            'velocity': track.velocity,
            'predicted_position': track.predicted_position
        }
    
    @staticmethod
    def serialize_prediction(prediction: PredictedTarget) -> dict:
        """Serialize a PredictedTarget object to dict"""
        return {
            'object_id': prediction.object_id,
            'bbox': prediction.predicted_bbox,
            'center': prediction.predicted_center,
            'confidence': prediction.confidence,
            'time_since_seen': prediction.time_since_seen,
            'velocity_projection': prediction.velocity_projection,
            'type': 'prediction',
            'inferred': True
        }


class WebSocketManager:
    """Main WebSocket manager coordinating all real-time communication"""
    
    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.frame_encoder = FrameEncoder()
        self.message_builder = MessageBuilder()
        self.processing_fps = 0
        self.last_fps_update = time.time()
        self.frame_counter = 0
        
        # Will be set when imported by main.py
        self.camera_manager = None
        self.detection_engine = None
        self.tracking_manager = None
        self.world_model = None
        self.pipeline = None  # PerceptionPipeline (shared singleton)
    
    def set_managers(self, camera_manager, detection_engine, tracking_manager, world_model):
        """Set manager references after initialization"""
        self.camera_manager = camera_manager
        self.detection_engine = detection_engine
        self.tracking_manager = tracking_manager
        self.world_model = world_model
    
    def set_pipeline(self, pipeline):
        """Set the shared perception pipeline."""
        self.pipeline = pipeline
    
    async def connect(self, websocket: WebSocket) -> str:
        """Handle new WebSocket connection (viewer role)"""
        connection_id = await self.connection_manager.connect(websocket)
        
        # Send initial status
        status_message = self.message_builder.build_status_message({
            'cameras_active': self.camera_manager.get_active_camera_count(),
            'detection_engine_ready': self.detection_engine.is_ready(),
            'tracking_active': self.tracking_manager.is_active()
        })
        
        await self._send_message(connection_id, status_message)
        return connection_id
    
    async def connect_camera_source(self, websocket: WebSocket) -> str:
        """Handle new WebSocket connection for a mobile camera source.
        
        Expects an initial JSON text message:
          { "type": "register", "role": "camera_source", "camera_id": <int|null> }
        
        If camera_id is null, the server auto-assigns an available slot.
        Responds with:
          { "type": "registered", "camera_id": <int>, "status": "ok" }
        or
          { "type": "error", "message": "..." }
        """
        await websocket.accept()
        
        try:
            # Wait for the registration message (timeout 10s)
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
            reg_msg = json.loads(raw)
        except (asyncio.TimeoutError, json.JSONDecodeError, Exception) as e:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Registration failed: expected JSON register message. {e}"
            }))
            await websocket.close()
            return None
        
        requested_id = reg_msg.get("camera_id")  # may be None for auto-assign
        
        # Register a virtual camera slot
        assigned_id = self.camera_manager.register_virtual_camera(
            camera_id=requested_id if requested_id is not None else None
        )
        
        if assigned_id is None:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "No camera slots available or slot already in use"
            }))
            await websocket.close()
            return None
        
        # Send success response
        await websocket.send_text(json.dumps({
            "type": "registered",
            "camera_id": assigned_id,
            "status": "ok",
            "target_fps": settings.mobile_camera_fps,
            "max_width": settings.mobile_camera_max_width
        }))
        
        print(f"📱 Mobile camera registered → slot {assigned_id}")
        return str(assigned_id)
    
    def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        # Find connection ID for this websocket
        connection_id = None
        for cid, ws in self.connection_manager.active_connections.items():
            if ws == websocket:
                connection_id = cid
                break
        
        if connection_id:
            self.connection_manager.disconnect(connection_id)
    
    async def handle_client_loop(self, websocket: WebSocket):
        """Main processing loop for a connected viewer client"""
        connection_id = None
        for cid, ws in self.connection_manager.active_connections.items():
            if ws == websocket:
                connection_id = cid
                break
        
        if not connection_id:
            return
        
        try:
            if self.pipeline:
                await self._pipeline_viewer_loop(connection_id)
            else:
                await self._legacy_viewer_loop(connection_id)
        except asyncio.CancelledError:
            pass
        except WebSocketDisconnect:
            self.disconnect(websocket)
        except Exception as e:
            print(f"❌ Client loop error for {connection_id}: {e}")
            self.disconnect(websocket)
    
    async def _pipeline_viewer_loop(self, connection_id: str):
        """Read from shared pipeline snapshot — no redundant GPU work.
        
        Each viewer polls the pipeline at the target FPS rate.  If the pipeline
        produced a new snapshot since the last send, the viewer sends it.
        Slow viewers automatically skip intermediate frames and always get
        the latest state.
        """
        last_gen = 0
        interval = 1.0 / settings.target_fps
        
        while connection_id in self.connection_manager.active_connections:
            snapshot = self.pipeline.latest
            if snapshot and snapshot.generation > last_gen:
                last_gen = snapshot.generation
                
                # Send camera frame packets
                for packed in snapshot.camera_packets.values():
                    ok = await self.connection_manager.send_to_client(connection_id, packed)
                    if not ok:
                        return
                
                # Send prediction packets for inactive cameras
                for packed in snapshot.prediction_packets.values():
                    await self.connection_manager.send_to_client(connection_id, packed)
                
                # Send world_update with all fused objects
                if snapshot.world_update_packet:
                    await self.connection_manager.send_to_client(connection_id, snapshot.world_update_packet)
            
            await asyncio.sleep(interval)
    
    async def _legacy_viewer_loop(self, connection_id: str):
        """Fallback per-viewer processing loop (original behaviour)."""
        while True:
            await self._process_frame_cycle(connection_id)
            await asyncio.sleep(1.0 / settings.target_fps)
    async def handle_camera_source_loop(self, websocket: WebSocket, camera_id: int):
        """Receive binary JPEG frames from a mobile camera source.
        
        Each incoming binary message is a raw JPEG image.
        Text messages can carry sensor data (GPS + IMU orientation).
        The loop injects each frame into the CameraManager's VirtualCamera.
        """
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                message = await websocket.receive()
                
                if message.get("type") == "websocket.disconnect":
                    break
                
                # Handle binary frames
                if "bytes" in message and message["bytes"]:
                    jpeg_bytes = message["bytes"]
                    self.camera_manager.inject_frame(camera_id, jpeg_bytes)
                    frame_count += 1
                    
                    # Log throughput every 100 frames
                    if frame_count % 100 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed if elapsed > 0 else 0
                        print(f"📱 Camera {camera_id}: {frame_count} frames, {fps:.1f} avg FPS")
                
                # Handle text control / sensor messages
                elif "text" in message and message["text"]:
                    try:
                        ctrl = json.loads(message["text"])
                        msg_type = ctrl.get("type")
                        
                        if msg_type == "stop":
                            print(f"📱 Camera {camera_id}: received stop command")
                            break
                        
                        elif msg_type == "sensor_data":
                            # GPS + IMU data from mobile device
                            self._handle_sensor_data(camera_id, ctrl)
                        
                    except json.JSONDecodeError:
                        pass
        
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"❌ Camera source loop error for camera {camera_id}: {e}")
        finally:
            # Clean up: unregister the virtual camera
            self.camera_manager.unregister_virtual_camera(camera_id)
            elapsed = time.time() - start_time
            print(f"📱 Camera {camera_id} disconnected: {frame_count} frames in {elapsed:.1f}s")
    
    def _handle_sensor_data(self, camera_id: int, data: dict):
        """Process GPS + IMU sensor data from a mobile camera.

        Expected format:
        {
            "type": "sensor_data",
            "gps": { "latitude": float, "longitude": float, "altitude": float, "accuracy": float },
            "orientation": { "alpha": float, "beta": float, "gamma": float },
            "timestamp": float
        }
        """
        try:
            import math
            import time as _time
            from app.core.world_model import CameraCalibration

            gps = data.get("gps")
            orientation = data.get("orientation")

            if gps and self.world_model:
                lat = gps.get("latitude", 0)
                lng = gps.get("longitude", 0)
                alt = gps.get("altitude", 0) or 0
                accuracy = gps.get("accuracy", 10.0) or 10.0

                # ── GPS reference origin ──────────────────────────────
                # Use config values if set, otherwise first fix becomes origin
                if not hasattr(self.world_model, '_gps_reference'):
                    ref_lat = settings.gps_reference_lat
                    ref_lng = settings.gps_reference_lng
                    if ref_lat is None or ref_lng is None:
                        ref_lat, ref_lng = lat, lng
                    self.world_model._gps_reference = (ref_lat, ref_lng)
                    print(f"📍 GPS reference set: {ref_lat:.6f}, {ref_lng:.6f}")

                ref_lat, ref_lng = self.world_model._gps_reference

                # Equirectangular projection to local metres
                meters_per_deg_lat = 111320.0
                meters_per_deg_lng = 111320.0 * math.cos(math.radians(ref_lat))
                world_x = (lng - ref_lng) * meters_per_deg_lng
                world_y = (lat - ref_lat) * meters_per_deg_lat
                world_z = max(alt, 1.5)

                # Derive rotation from device orientation
                # DeviceOrientation: alpha=compass heading, beta=tilt (90=horizontal), gamma=roll
                yaw = math.radians(orientation.get("alpha", 0)) if orientation else 0.0
                # beta=90 means phone held horizontally (normal camera position) → pitch=0
                beta_deg = orientation.get("beta", 90) if orientation else 90.0
                pitch = math.radians(beta_deg - 90.0)
                roll = math.radians(orientation.get("gamma", 0)) if orientation else 0.0

                # Compute image center from actual mobile frame dimensions
                fw = float(settings.mobile_camera_max_width)
                fh = fw * 0.75  # typical 4:3 mobile aspect
                img_cx, img_cy = fw / 2.0, fh / 2.0

                # Estimate focal length from ~60° horizontal FOV
                fov_h_rad = math.radians(60.0)
                focal = (fw / 2.0) / math.tan(fov_h_rad / 2.0)

                new_pos = (world_x, world_y, world_z)

                # ── Auto-invalidate homography on camera movement ────
                old_calib = self.world_model.coordinate_transform.camera_calibrations.get(camera_id)
                if old_calib is not None and old_calib.position_at_h_learn is not None:
                    old_p = old_calib.position_at_h_learn
                    dist_moved = math.sqrt(
                        (new_pos[0] - old_p[0])**2 +
                        (new_pos[1] - old_p[1])**2 +
                        (new_pos[2] - old_p[2])**2
                    )
                    if dist_moved > settings.homography_movement_threshold:
                        # Camera moved significantly — flush learned homographies
                        for other_cam in list(self.world_model.coordinate_transform.camera_calibrations.keys()):
                            if other_cam != camera_id:
                                self.world_model.cross_camera_homography.invalidate(camera_id, other_cam)
                        # Reset anchor so next H learns at current position
                        # (position_at_h_learn set to None → re-anchored on next H learn)

                now = _time.time()
                calib = CameraCalibration(
                    camera_id=camera_id,
                    position=new_pos,
                    rotation=(roll, pitch, yaw),
                    focal_length=focal,
                    image_center=(img_cx, img_cy),
                    gps_accuracy=accuracy,
                    last_update=now,
                    position_at_h_learn=(
                        old_calib.position_at_h_learn if old_calib else None
                    ),
                )
                self.world_model.coordinate_transform.add_camera_calibration(calib)

                # ── Update VirtualCamera with live position ──────────
                vcam = self.camera_manager.cameras.get(camera_id)
                if vcam and hasattr(vcam, 'gps_position'):
                    vcam.gps_position = new_pos
                    vcam.heading = math.degrees(yaw)
                    vcam.gps_accuracy = accuracy

        except Exception as e:
            print(f"❌ Sensor data error for camera {camera_id}: {e}")
    
    async def _process_frame_cycle(self, connection_id: str):
        """Process one cycle of frames from all cameras"""
        try:
            # Get latest frames from all cameras
            camera_frames = self.camera_manager.get_all_frames()
            
            if not camera_frames:
                return
            
            # Process detections for all frames
            detection_results = await self.detection_engine.process_frames(camera_frames)
            
            # Process tracking for each detection result
            tracking_results = []
            for detection_result in detection_results:
                # Get the original frame for DeepSORT
                frame = None
                for cam_frame in camera_frames:
                    if cam_frame.camera_id == detection_result.camera_id:
                        frame = cam_frame.frame
                        break
                
                tracking_result = await self.tracking_manager.process_detections(detection_result, frame)
                tracking_results.append(tracking_result)
            
            # Update world model with tracking results
            await self.world_model.update_with_tracking_results(tracking_results)
            
            # Send frame data to client
            for i, camera_frame in enumerate(camera_frames):
                await self._send_camera_frame(
                    connection_id,
                    camera_frame,
                    detection_results[i] if i < len(detection_results) else None,
                    tracking_results[i] if i < len(tracking_results) else None
                )
            
            # Send predictions for cameras that don't have current frames
            await self._send_predictions(connection_id, camera_frames)
            
            # Update FPS statistics
            self._update_fps_stats()
            
        except Exception as e:
            print(f"❌ Frame cycle error: {e}")
    
    async def _send_camera_frame(
        self, 
        connection_id: str, 
        camera_frame: CameraFrame, 
        detection_result: Optional[DetectionResult],
        tracking_result: Optional[TrackingResult]
    ):
        """Send processed camera frame to client"""
        # Encode frame
        frame_data = self.frame_encoder.encode_frame(camera_frame.frame)
        if frame_data is None:
            return
        
        # Serialize detections
        detections = []
        if detection_result:
            detections = [
                self.message_builder.serialize_detection(det) 
                for det in detection_result.detections
            ]
        
        # Serialize tracks
        tracks = []
        if tracking_result:
            tracks = [
                self.message_builder.serialize_track(track)
                for track in tracking_result.tracks
            ]
        
        # Get predictions for this camera
        predictions = []
        current_time = time.time()
        predicted_targets = self.world_model.generate_predictions_for_camera(
            camera_frame.camera_id, current_time
        )
        predictions = [
            self.message_builder.serialize_prediction(pred)
            for pred in predicted_targets
        ]
        
        # Build and send message
        message = self.message_builder.build_frame_message(
            camera_frame.camera_id, frame_data, detections, tracks, predictions
        )
        
        await self._send_message(connection_id, message)
    
    async def _send_predictions(self, connection_id: str, active_camera_frames: List[CameraFrame]):
        """Send predictions for cameras without current frames"""
        active_camera_ids = {frame.camera_id for frame in active_camera_frames}
        current_time = time.time()
        
        # Send predictions for inactive cameras (up to max cameras)
        for camera_id in range(settings.max_cameras):
            if camera_id not in active_camera_ids:
                predicted_targets = self.world_model.generate_predictions_for_camera(camera_id, current_time)
                
                if predicted_targets:
                    predictions = [
                        self.message_builder.serialize_prediction(pred)
                        for pred in predicted_targets
                    ]
                    
                    message = self.message_builder.build_prediction_message(camera_id, predictions)
                    await self._send_message(connection_id, message)
    
    async def _send_message(self, connection_id: str, message: dict):
        """Send message to client using msgpack serialization"""
        try:
            # Serialize with msgpack
            data = msgpack.packb(message, use_bin_type=True)
            await self.connection_manager.send_to_client(connection_id, data)
            
        except Exception as e:
            print(f"❌ Failed to send message to {connection_id}: {e}")
    
    def _update_fps_stats(self):
        """Update FPS statistics"""
        self.frame_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_update >= 1.0:
            self.processing_fps = self.frame_counter / (current_time - self.last_fps_update)
            self.frame_counter = 0
            self.last_fps_update = current_time
    
    def get_client_count(self) -> int:
        """Get number of connected clients"""
        return self.connection_manager.get_connection_count()
    
    def get_stats(self) -> dict:
        """Get WebSocket manager statistics"""
        return {
            'connected_clients': self.get_client_count(),
            'processing_fps': round(self.processing_fps, 1),
            'encoding_stats': self.frame_encoder.get_stats(),
            'connection_stats': self.connection_manager.get_connection_stats()
        }