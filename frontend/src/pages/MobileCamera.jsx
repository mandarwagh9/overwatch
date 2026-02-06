/**
 * MobileCamera — standalone page for mobile users to stream their camera
 * 
 * Usage: User opens https://<server-ip>:3000/mobile on their phone,
 *        taps "Start Streaming", and their camera feed is sent to the backend.
 */

import React, { useRef, useState, useCallback, useEffect } from 'react';
import { CameraStreamService } from '../services/cameraStream';
import './MobileCamera.css';

const MobileCamera = () => {
  const videoRef = useRef(null);
  const streamServiceRef = useRef(null);

  const [status, setStatus] = useState('idle');       // idle | initializing | camera_ready | streaming | stopped | disconnected
  const [cameraId, setCameraId] = useState(null);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);
  const [facingMode, setFacingMode] = useState('environment');

  // Create the stream service once
  useEffect(() => {
    streamServiceRef.current = new CameraStreamService();
    const svc = streamServiceRef.current;

    svc.onStatusChange = (newStatus) => setStatus(newStatus);
    svc.onError = (msg) => setError(msg);
    svc.onStatsUpdate = (newStats) => setStats(newStats);

    return () => {
      // Cleanup on unmount
      if (streamServiceRef.current) {
        streamServiceRef.current.stop();
      }
    };
  }, []);

  const handleStart = useCallback(async () => {
    setError(null);
    try {
      const id = await streamServiceRef.current.start(videoRef.current, {
        facingMode,
        targetFps: 15,
        jpegQuality: 0.5,
        maxWidth: 640
      });
      setCameraId(id);
    } catch (err) {
      setError(err.message);
      setStatus('idle');
    }
  }, [facingMode]);

  const handleStop = useCallback(() => {
    if (streamServiceRef.current) {
      streamServiceRef.current.stop();
    }
    setCameraId(null);
    setStats(null);
    setStatus('stopped');
  }, []);

  const handleSwitchCamera = useCallback(async () => {
    if (streamServiceRef.current) {
      await streamServiceRef.current.switchCamera();
      setFacingMode(streamServiceRef.current.facingMode);
    }
  }, []);

  const isActive = status === 'streaming' || status === 'camera_ready' || status === 'initializing';

  return (
    <div className="mobile-camera-page">
      {/* Header */}
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
          style={{ display: status !== 'idle' && status !== 'stopped' ? 'block' : 'none' }}
        />
        
        {(status === 'idle' || status === 'stopped') && (
          <div className="mobile-placeholder">
            <div className="placeholder-icon">📷</div>
            <p>Tap "Start Streaming" to begin</p>
            <p className="placeholder-hint">Your camera feed will be sent to the Overwatch system</p>
          </div>
        )}

        {/* Status overlay */}
        {status === 'streaming' && (
          <div className="streaming-indicator">
            <span className="recording-dot"></span>
            LIVE — Camera {cameraId}
          </div>
        )}

        {status === 'initializing' && (
          <div className="streaming-indicator initializing">
            ⏳ Connecting...
          </div>
        )}
      </div>

      {/* Stats */}
      {stats && status === 'streaming' && (
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
        {!isActive ? (
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
          disabled={status === 'idle' || status === 'stopped'}
        >
          🔄 {facingMode === 'environment' ? 'Front' : 'Rear'} Camera
        </button>
      </div>

      {/* Connection Info */}
      <div className="mobile-info">
        <p>
          Status: <strong>{status}</strong>
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
};

export default MobileCamera;
