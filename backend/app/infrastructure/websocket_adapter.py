"""
WebSocket communication adapter.
Handles real-time client connections with msgpack serialization.
"""
from __future__ import annotations
import asyncio
import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime

import msgpack
from fastapi import WebSocket, WebSocketDisconnect

from app.application.ports import CommunicationRepository
from app.domain.entities import PerceptionSnapshot, Detection, Track, WorldObject, PredictedTarget


logger = logging.getLogger(__name__)


@dataclass
class ClientConnection:
    """Represents a connected WebSocket client."""
    connection_id: str
    websocket: WebSocket
    connected_at: datetime
    last_activity: datetime
    messages_sent: int = 0
    bytes_sent: int = 0


class WebSocketCommunicationRepository(CommunicationRepository):
    """WebSocket communication repository using msgpack."""
    
    def __init__(self):
        self._clients: Dict[str, ClientConnection] = {}
        self._next_id = 1
        self._lock = asyncio.Lock()
        
        logger.info("WebSocketCommunicationRepository initialized")
    
    async def connect(self, websocket: WebSocket) -> str:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        
        connection_id = f"client_{self._next_id}"
        self._next_id += 1
        
        now = datetime.now()
        client = ClientConnection(
            connection_id=connection_id,
            websocket=websocket,
            connected_at=now,
            last_activity=now
        )
        
        async with self._lock:
            self._clients[connection_id] = client
        
        logger.info(f"Client {connection_id} connected")
        return connection_id
    
    def disconnect(self, connection_id: str) -> None:
        """Remove a WebSocket connection."""
        if connection_id in self._clients:
            del self._clients[connection_id]
            logger.info(f"Client {connection_id} disconnected")
    
    async def send_to_client(self, connection_id: str, data: bytes) -> bool:
        """Send data to a specific client."""
        if connection_id not in self._clients:
            return False
        
        client = self._clients[connection_id]
        
        try:
            await client.websocket.send_bytes(data)
            client.messages_sent += 1
            client.bytes_sent += len(data)
            client.last_activity = datetime.now()
            return True
            
        except Exception as e:
            logger.warning(f"Failed to send to {connection_id}: {e}")
            self.disconnect(connection_id)
            return False
    
    async def broadcast_snapshot(self, snapshot: PerceptionSnapshot) -> None:
        """Broadcast a perception snapshot to all connected clients."""
        if not self._clients:
            logger.warning("No clients connected, skipping broadcast")
            return
        
        # Serialize snapshot
        data = self._serialize_snapshot(snapshot)
        
        # Debug: Log snapshot being broadcast
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"📡 Broadcasting snapshot gen={snapshot.generation}, "
                    f"cameras={len(snapshot.camera_frames)}, "
                    f"clients={len(self._clients)}, "
                    f"data_size={len(data)} bytes")
        
        # Send to all clients concurrently
        tasks = [
            self.send_to_client(client_id, data)
            for client_id in list(self._clients.keys())
        ]
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            failed = sum(1 for r in results if isinstance(r, Exception))
            if failed > 0:
                logger.warning(f"Failed to send snapshot to {failed} clients")
    
    def get_client_count(self) -> int:
        """Get number of connected clients."""
        return len(self._clients)
    
    def _serialize_snapshot(self, snapshot: PerceptionSnapshot) -> bytes:
        """Serialize a perception snapshot to msgpack bytes."""
        # Serialize camera frames (convert numpy arrays to bytes)
        camera_packets = {}
        for cam_id, frame in snapshot.camera_frames.items():
            if frame and frame.frame_data is not None:
                # Encode frame to JPEG for transmission
                import cv2
                success, encoded = cv2.imencode('.jpg', frame.frame_data, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if success:
                    camera_packets[str(cam_id)] = encoded.tobytes()
        
        message = {
            "type": "snapshot",
            "timestamp": snapshot.timestamp.isoformat(),
            "generation": snapshot.generation,
            "camera_frames": camera_packets,
            "world_objects": [
                self._serialize_world_object(obj)
                for obj in snapshot.world_objects
            ],
            "detections": {
                str(cam_id): [
                    self._serialize_detection(det)
                    for det in detections
                ]
                for cam_id, detections in snapshot.detections.items()
            },
            "tracks": {
                str(cam_id): [
                    self._serialize_track(track)
                    for track in tracks
                ]
                for cam_id, tracks in snapshot.tracks.items()
            },
            "predictions": {
                str(cam_id): [
                    self._serialize_prediction(pred)
                    for pred in predictions
                ]
                for cam_id, predictions in snapshot.predictions.items()
            },
            "metrics": {
                "processing_time_ms": snapshot.processing_time_ms,
                "fps": snapshot.fps
            }
        }
        
        return msgpack.packb(message, use_bin_type=True)
    
    def _serialize_detection(self, detection: Detection) -> dict:
        """Serialize a detection to dict."""
        return {
            "detection_id": detection.detection_id,
            "camera_id": detection.camera_id,
            "bbox": [
                detection.bbox.x1,
                detection.bbox.y1,
                detection.bbox.x2,
                detection.bbox.y2
            ],
            "confidence": detection.confidence,
            "class_id": detection.class_id,
            "class_name": detection.class_name,
            "center": detection.bbox.center,
            "keypoints": [
                {"x": kp.x, "y": kp.y, "confidence": kp.confidence}
                for kp in detection.keypoints
            ] if detection.keypoints else []
        }
    
    def _serialize_track(self, track: Track) -> dict:
        """Serialize a track to dict."""
        return {
            "track_id": track.track_id,
            "camera_id": track.camera_id,
            "bbox": [
                track.bbox.x1,
                track.bbox.y1,
                track.bbox.x2,
                track.bbox.y2
            ],
            "confidence": track.confidence,
            "class_id": track.class_id,
            "class_name": track.class_name,
            "state": track.state.name,
            "age": track.age,
            "hits": track.hits,
            "velocity": track.velocity,
            "center": track.bbox.center
        }
    
    def _serialize_world_object(self, obj: WorldObject) -> dict:
        """Serialize a world object to dict."""
        return {
            "object_id": obj.object_id,
            "position": {
                "x": obj.position.x,
                "y": obj.position.y,
                "z": obj.position.z
            },
            "velocity": {
                "vx": obj.velocity.vx,
                "vy": obj.velocity.vy,
                "vz": obj.velocity.vz
            },
            "class_id": obj.class_id,
            "class_name": obj.class_name,
            "confidence": obj.confidence,
            "last_seen_camera": obj.last_seen_camera,
            "uncertainty": obj.position_uncertainty
        }
    
    def _serialize_prediction(self, pred: PredictedTarget) -> dict:
        """Serialize a prediction to dict."""
        return {
            "object_id": pred.object_id,
            "camera_id": pred.camera_id,
            "bbox": [
                pred.predicted_bbox.x1,
                pred.predicted_bbox.y1,
                pred.predicted_bbox.x2,
                pred.predicted_bbox.y2
            ],
            "confidence": pred.confidence,
            "time_since_seen": pred.time_since_seen,
            "source_camera": pred.source_camera,
            "method": pred.prediction_method.value
        }
