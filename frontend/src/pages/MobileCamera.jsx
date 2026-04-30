/**
 * Mobile Camera Page
 * Standalone page for mobile users to stream their camera
 */

import React, { useRef, useState, useEffect, useCallback } from 'react';
import cameraStreamAdapter, { StreamState, StreamEvents } from '../infrastructure/cameraStreamAdapter';
import './MobileCamera.css';

function MobileCamera() {
  const videoRef = useRef(null);
  const [streamState, setStreamState] = useState(StreamState.IDLE);
  const [cameraId, setCameraId] = useState(null);
  const [stats, setStats] = useState({
    framesSent: 0,
    bytesSent: 0,
    mbSent: '0.0',
    fps: 0,
    elapsed: 0,
    facingMode: 'environment'
  });
  const [error, setError] = useState(null);

  // Update stream state
  useEffect(() => {
    const handleStateChange = (state) => {
      setStreamState(state);
    };

    const handleCameraRegistered = (id) => {
      setCameraId(id);
    };

    const handleStatsUpdate = (newStats) => {
      setStats(newStats);
    };

    const handleError = (message) => {
      setError(message);
    };

    cameraStreamAdapter.on(StreamEvents.STATE_CHANGE, handleStateChange);
    cameraStreamAdapter.on(StreamEvents.CAMERA_REGISTERED, handleCameraRegistered);
    cameraStreamAdapter.on(StreamEvents.STATS_UPDATE, handleStatsUpdate);
    cameraStreamAdapter.on(StreamEvents.ERROR, handleError);

    return () => {
      cameraStreamAdapter.off(StreamEvents.STATE_CHANGE, handleStateChange);
      cameraStreamAdapter.off(StreamEvents.CAMERA_REGISTERED, handleCameraRegistered);
      cameraStreamAdapter.off(StreamEvents.STATS_UPDATE, handleStatsUpdate);
      cameraStreamAdapter.off(StreamEvents.ERROR, handleError);
      try {
        cameraStreamAdapter.stop();
      } catch (e) {
        // adapter may not be running; safe to ignore
      }
    };
  }, []);

  // Start streaming
  const handleStart = useCallback(async () => {
    setError(null);
    
    try {
      await cameraStreamAdapter.start(videoRef.current);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  // Stop streaming
  const handleStop = useCallback(() => {
    cameraStreamAdapter.stop();
    setCameraId(null);
  }, []);

  // Switch camera
  const handleSwitchCamera = useCallback(async () => {
    try {
      await cameraStreamAdapter.switchCamera();
    } catch (err) {
      setError(err.message);
    }
  }, []);

  // Toggle start/stop
  const handleToggle = useCallback(() => {
    if (cameraStreamAdapter.isStreaming) {
      handleStop();
    } else {
      handleStart();
    }
  }, [handleStart, handleStop]);

  const isStreaming = cameraStreamAdapter.isStreaming;

  return (
    <div className="mobile-camera-page">
      <header className="mobile-header">
        <h1>📱 OVERWATCH</h1>
        <span className="mobile-subtitle">Mobile Camera</span>
      </header>

      {/* Error Banner */}
      {error && (
        <div className="mobile-error">
          ⚠️ {error}
          <button onClick={() => setError(null)} className="dismiss-btn">✕</button>
        </div>
      )}

      {/* Camera Preview */}
      <div className="mobile-preview-container">
        <video
          ref={videoRef}
          className="mobile-video-preview"
          autoPlay
          playsInline
          muted
          style={{ display: isStreaming ? 'block' : 'none' }}
        />
        
        {!isStreaming && (
          <div className="mobile-placeholder">
            <div className="placeholder-icon">📷</div>
            <p>Tap "Start Streaming" to begin</p>
            <p className="placeholder-hint">Your camera feed will be sent to the Overwatch system</p>
          </div>
        )}

        {/* Status overlay */}
        {streamState === StreamState.STREAMING && (
          <div className="streaming-indicator">
            <span className="recording-dot"></span>
            LIVE — Camera {cameraId}
          </div>
        )}

        {streamState === StreamState.CONNECTING && (
          <div className="streaming-indicator initializing">
            ⏳ Connecting...
          </div>
        )}
      </div>

      {/* Stats */}
      {isStreaming && (
        <div className="mobile-stats">
          <div className="stat-item">
            <span className="stat-value">{stats.fps}</span>
            <span className="stat-label">FPS</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">{stats.framesSent}</span>
            <span className="stat-label">Frames</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">{stats.mbSent} MB</span>
            <span className="stat-label">Sent</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">{stats.elapsed}s</span>
            <span className="stat-label">Duration</span>
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="mobile-controls">
        {!isStreaming ? (
          <button onClick={handleStart} className="mobile-btn start-btn">
            ▶️ Start Streaming
          </button>
        ) : (
          <button onClick={handleStop} className="mobile-btn stop-btn">
            ⏹️ Stop Streaming
          </button>
        )}

        <button
          onClick={handleSwitchCamera}
          className="mobile-btn switch-btn"
          disabled={!isStreaming}
        >
          🔄 {stats.facingMode === 'environment' ? 'Front' : 'Rear'} Camera
        </button>
      </div>

      {/* Connection Info */}
      <div className="mobile-info">
        <p>
          Status: <strong>{streamState}</strong>
          {cameraId !== null && ` • Slot: ${cameraId}`}
        </p>
        <p className="mobile-info-hint">
          Make sure you're connected to the same network as the Overwatch server.
          {window.location.protocol !== 'https:' && (
            <span className="ssl-warning"> ⚠️ Camera access requires HTTPS.</span>
          )}
        </p>
      </div>
    </div>
  );
}

export default MobileCamera;
