/**
 * Main App component for Overwatch frontend
 */

import React, { useState, useEffect, useRef } from 'react';
import { Routes, Route, Link, useLocation } from 'react-router-dom';
import CameraDisplay from './components/CameraDisplay';
import MobileCamera from './pages/MobileCamera';
import { websocketService } from './services/websocket';
import './App.css';

function App() {
  return (
    <Routes>
      <Route path="/mobile" element={<MobileCamera />} />
      <Route path="/*" element={<AdminDashboard />} />
    </Routes>
  );
}

function AdminDashboard() {
  const [isConnected, setIsConnected] = useState(false);
  const [systemStats, setSystemStats] = useState({});
  const [cameraData, setCameraData] = useState({});
  const [connectionStats, setConnectionStats] = useState({});
  const [error, setError] = useState(null);
  
  // Refs for performance tracking
  const frameCounterRef = useRef({});
  const fpsTimerRef = useRef({});

  // Initialize WebSocket connection
  useEffect(() => {
    const connectToBackend = async () => {
      try {
        // Use env var for backend host, fallback to same-origin
        const wsHost = process.env.REACT_APP_BACKEND_HOST || window.location.hostname || 'localhost';
        const wsPort = process.env.REACT_APP_BACKEND_PORT || '8000';
        const wsUrl = `wss://${wsHost}:${wsPort}/ws`;
        
        await websocketService.connect(wsUrl);
        setIsConnected(true);
        setError(null);
      } catch (error) {
        setError(
          'Failed to connect to backend. If using a self-signed certificate, ' +
          'open https://' + (process.env.REACT_APP_BACKEND_HOST || window.location.hostname || 'localhost') + ':' +
          (process.env.REACT_APP_BACKEND_PORT || '8000') + ' in a new tab, ' +
          'accept the certificate warning, then refresh this page.'
        );
        console.error('Connection error:', error);
      }
    };

    connectToBackend();

    // Set up event listeners
    websocketService.on('connected', () => {
      setIsConnected(true);
      setError(null);
    });

    websocketService.on('disconnected', () => {
      setIsConnected(false);
    });

    websocketService.on('error', (error) => {
      setError(`WebSocket error: ${error.message || 'Unknown error'}`);
    });

    websocketService.on('maxReconnectAttemptsReached', () => {
      setError('Lost connection to backend. Please refresh the page.');
    });

    // Handle frame messages
    websocketService.on('frame', (data) => {
      handleFrameMessage(data);
    });

    // Handle prediction messages
    websocketService.on('predictions', (data) => {
      handlePredictionMessage(data);
    });

    // Handle status messages
    websocketService.on('status', (data) => {
      setSystemStats(data.stats || {});
    });

    // Cleanup on unmount
    return () => {
      websocketService.disconnect();
    };
  }, []);

  // Handle frame messages from backend
  const handleFrameMessage = (data) => {
    const cameraId = data.camera_id;
    
    // Update frame counter for FPS calculation
    if (!frameCounterRef.current[cameraId]) {
      frameCounterRef.current[cameraId] = 0;
      fpsTimerRef.current[cameraId] = Date.now();
    }
    
    frameCounterRef.current[cameraId]++;
    
    // Calculate FPS every second
    const now = Date.now();
    const elapsed = now - fpsTimerRef.current[cameraId];
    if (elapsed >= 1000) {
      const fps = (frameCounterRef.current[cameraId] / elapsed) * 1000;
      frameCounterRef.current[cameraId] = 0;
      fpsTimerRef.current[cameraId] = now;
      
      // Update FPS in camera data
      setCameraData(prev => ({
        ...prev,
        [cameraId]: {
          ...prev[cameraId],
          fps: Math.round(fps * 10) / 10
        }
      }));
    }

    // Update camera data
    setCameraData(prev => ({
      ...prev,
      [cameraId]: {
        frameData: data.frame_data,
        detections: data.detections || [],
        tracks: data.tracks || [],
        predictions: data.predictions || [],
        isActive: true,
        lastUpdate: Date.now(),
        fps: prev[cameraId]?.fps || 0
      }
    }));
  };

  // Handle prediction-only messages (for inactive cameras)
  const handlePredictionMessage = (data) => {
    const cameraId = data.camera_id;
    
    setCameraData(prev => ({
      ...prev,
      [cameraId]: {
        ...prev[cameraId],
        predictions: data.predictions || [],
        lastPredictionUpdate: Date.now()
      }
    }));
  };

  // Update connection statistics
  useEffect(() => {
    const updateStats = () => {
      setConnectionStats(websocketService.getStats());
    };

    const interval = setInterval(updateStats, 1000);
    return () => clearInterval(interval);
  }, []);

  // Mark cameras as inactive if no recent updates
  useEffect(() => {
    const checkCameraActivity = () => {
      const now = Date.now();
      const inactivityTimeout = 3000; // 3 seconds

      setCameraData(prev => {
        const updated = { ...prev };
        
        Object.keys(updated).forEach(cameraId => {
          if (updated[cameraId].lastUpdate && 
              now - updated[cameraId].lastUpdate > inactivityTimeout) {
            updated[cameraId] = {
              ...updated[cameraId],
              isActive: false
            };
          }
        });
        
        return updated;
      });
    };

    const interval = setInterval(checkCameraActivity, 1000);
    return () => clearInterval(interval);
  }, []);

  // Request camera start/stop
  const toggleCamera = async (cameraId) => {
    try {
      const camera = cameraData[cameraId];
      const action = camera?.isActive ? 'stop' : 'start';
      
      const apiBase = `https://${window.location.hostname}:8000`;
      const response = await fetch(`${apiBase}/camera/${cameraId}/${action}`, {
        method: 'POST'
      });
      
      if (!response.ok) {
        throw new Error(`Failed to ${action} camera ${cameraId}`);
      }
      
      const result = await response.json();
      console.log(result.message);
      
    } catch (error) {
      setError(`Camera control error: ${error.message}`);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🎯 OVERWATCH</h1>
        <div className="header-stats">
          <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
            {isConnected ? '🟢 CONNECTED' : '🔴 DISCONNECTED'}
          </div>
          <div className="system-info">
            <span>Cameras: {systemStats.cameras_active || 0}/{systemStats.max_cameras || 4}</span>
            <span>FPS Target: {systemStats.target_fps || 24}</span>
            <span>Clients: {systemStats.connected_clients || 0}</span>
          </div>
        </div>
      </header>

      {error && (
        <div className="error-banner">
          ⚠️ {error}
        </div>
      )}

      <main className="app-main">
        <div className="camera-grid">
          {[0, 1, 2, 3].map(cameraId => {
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
                    onClick={() => toggleCamera(cameraId)}
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

        <aside className="stats-panel">
          <h3>📊 System Statistics</h3>
          
          <div className="stat-section">
            <h4>Detection Engine</h4>
            <p>Model: {systemStats.detection_model || 'YOLOv8n'}</p>
            <p>Status: {systemStats.detection_engine_ready ? '✅ Ready' : '❌ Loading'}</p>
          </div>

          <div className="stat-section">
            <h4>Tracking</h4>
            <p>Active: {systemStats.tracking_active ? '✅ Yes' : '❌ No'}</p>
            <p>Mode: CPU Optimized</p>
          </div>

          <div className="stat-section">
            <h4>Connection</h4>
            <p>Messages: {connectionStats.messagesReceived || 0}</p>
            <p>Data: {((connectionStats.bytesReceived || 0) / 1024 / 1024).toFixed(1)} MB</p>
            <p>Uptime: {
              connectionStats.connectionTime 
                ? Math.round((Date.now() - connectionStats.connectionTime) / 1000) + 's'
                : '0s'
            }</p>
          </div>

          <div className="stat-section">
            <h4>Performance</h4>
            {Object.entries(cameraData).map(([cameraId, data]) => (
              <p key={cameraId}>
                Cam {cameraId}: {data.isActive ? `${data.fps || 0} FPS` : 'Offline'}
              </p>
            ))}
          </div>
        </aside>
      </main>
    </div>
  );
}

export default App;