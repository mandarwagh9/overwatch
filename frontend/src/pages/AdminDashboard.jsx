/**
 * Admin Dashboard page
 * Main monitoring interface with camera grid and statistics
 */

import React, { useEffect, useRef } from 'react';
import { useWebSocket, useCameraData, useCameraControl, useSystemStats } from '../application/hooks';
import { WebSocketEvents } from '../infrastructure/websocketAdapter';
import { getConfig } from '../config';
import CameraDisplay from '../components/CameraDisplay';
import StatsPanel from '../components/StatsPanel';
import ConnectionStatus from '../components/ConnectionStatus';
import ErrorBanner from '../components/ErrorBanner';
import './AdminDashboard.css';

function AdminDashboard() {
  const maxCameras = getConfig('camera.maxCameras', 4);
  
  const {
    isConnected,
    error: wsError,
    stats: connectionStats,
    connect,
    disconnect,
    on,
    off
  } = useWebSocket();

  const {
    cameraData,
    worldObjects,
    pipelineStats,
    handleSnapshot,
    handleFrame,
    handlePredictions,
    handleWorldUpdate,
    setCameraActive
  } = useCameraData();

  const {
    isLoading: isCameraLoading,
    error: cameraError,
    toggleCamera
  } = useCameraControl();

  const { stats: systemStats } = useSystemStats();

  // Use refs to store stable callback references
  const handleSnapshotRef = useRef(handleSnapshot);
  const handleFrameRef = useRef(handleFrame);
  const handlePredictionsRef = useRef(handlePredictions);
  const handleWorldUpdateRef = useRef(handleWorldUpdate);
  const onRef = useRef(on);
  const offRef = useRef(off);

  // Update refs when callbacks change
  useEffect(() => {
    handleSnapshotRef.current = handleSnapshot;
    handleFrameRef.current = handleFrame;
    handlePredictionsRef.current = handlePredictions;
    handleWorldUpdateRef.current = handleWorldUpdate;
    onRef.current = on;
    offRef.current = off;
  }, [handleSnapshot, handleFrame, handlePredictions, handleWorldUpdate, on, off]);

  // Set up WebSocket event handlers
  useEffect(() => {
    const onSnap = (data) => handleSnapshotRef.current(data);
    const onFrame = (data) => handleFrameRef.current(data);
    const onPred = (data) => handlePredictionsRef.current(data);
    const onWorld = (data) => handleWorldUpdateRef.current(data);

    onRef.current('snapshot', onSnap);
    onRef.current(WebSocketEvents.FRAME, onFrame);
    onRef.current(WebSocketEvents.PREDICTIONS, onPred);
    onRef.current(WebSocketEvents.WORLD_UPDATE, onWorld);

    return () => {
      offRef.current('snapshot', onSnap);
      offRef.current(WebSocketEvents.FRAME, onFrame);
      offRef.current(WebSocketEvents.PREDICTIONS, onPred);
      offRef.current(WebSocketEvents.WORLD_UPDATE, onWorld);
    };
  }, []);

  // Connect on mount
  useEffect(() => {
    connect().catch(err => {
      console.error('Failed to connect:', err);
    });

    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  // Handle camera toggle
  const handleToggleCamera = async (cameraId) => {
    const camera = cameraData[cameraId];
    const isActive = camera?.isActive || false;
    
    try {
      await toggleCamera(cameraId, isActive);
      // Optimistically update UI
      setCameraActive(cameraId, !isActive);
    } catch (err) {
      // Error is handled by hook
    }
  };

  const error = wsError || cameraError;

  return (
    <div className="admin-dashboard">
      <header className="dashboard-header">
        <h1>🎯 OVERWATCH</h1>
        <div className="header-stats">
          <ConnectionStatus isConnected={isConnected} />
          <div className="system-info">
            <span>Cameras: {systemStats.cameras_active || 0}/{systemStats.max_cameras || maxCameras}</span>
            <span>FPS Target: {systemStats.target_fps || 24}</span>
            <span>Clients: {systemStats.connected_clients || 0}</span>
          </div>
        </div>
      </header>

      {error && <ErrorBanner message={error} />}

      <main className="dashboard-main">
        <div className="camera-grid">
          {Array.from({ length: maxCameras }, (_, i) => i).map(cameraId => {
            const camera = cameraData[cameraId] || {};
            
            return (
              <div key={cameraId} className="camera-container">
                <CameraDisplay
                  cameraId={cameraId}
                  frameData={camera.frameData}
                  detections={camera.detections || []}
                  tracks={camera.tracks || []}
                  predictions={camera.predictions || []}
                  isActive={camera.isActive || false}
                />
                
                <div className="camera-controls">
                  <button
                    onClick={() => handleToggleCamera(cameraId)}
                    disabled={isCameraLoading}
                    className={`control-btn ${camera.isActive ? 'stop' : 'start'}`}
                  >
                    {camera.isActive ? '⏹️ Stop' : '▶️ Start'}
                  </button>
                  <span className="fps-display">
                    {camera.fps ? `${camera.fps} FPS` : '0 FPS'}
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        <StatsPanel
          systemStats={systemStats}
          worldObjects={worldObjects}
          pipelineStats={pipelineStats}
          connectionStats={connectionStats}
          cameraData={cameraData}
        />
      </main>
    </div>
  );
}

export default AdminDashboard;
