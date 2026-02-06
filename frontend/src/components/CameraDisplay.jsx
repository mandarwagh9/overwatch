/**
 * Camera display component with AR overlays
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';

const CameraDisplay = ({ 
  cameraId, 
  frameData, 
  detections = [], 
  tracks = [], 
  predictions = [],
  isActive = false
}) => {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 640, height: 480 });
  const [stats, setStats] = useState({
    fps: 0,
    detectionsCount: 0,
    tracksCount: 0,
    predictionsCount: 0
  });

  // Update statistics
  useEffect(() => {
    setStats({
      fps: 0, // Will be calculated elsewhere
      detectionsCount: detections.length,
      tracksCount: tracks.length,
      predictionsCount: predictions.length
    });
  }, [detections, tracks, predictions]);

  // Draw overlays on canvas
  const drawOverlays = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw detections — clean tactical-style PERSON boxes
    detections.forEach(detection => {
      const [x1, y1, x2, y2] = detection.bbox;
      const w = x2 - x1;
      const h = y2 - y1;
      const cornerLen = Math.min(22, w * 0.18, h * 0.18);

      // Subtle full bounding box
      ctx.strokeStyle = 'rgba(0, 255, 200, 0.35)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x1, y1, w, h);

      // Corner accent brackets (bright cyan)
      ctx.strokeStyle = '#00ffc8';
      ctx.lineWidth = 3;
      ctx.lineCap = 'square';

      // Top-left
      ctx.beginPath();
      ctx.moveTo(x1, y1 + cornerLen);
      ctx.lineTo(x1, y1);
      ctx.lineTo(x1 + cornerLen, y1);
      ctx.stroke();
      // Top-right
      ctx.beginPath();
      ctx.moveTo(x2 - cornerLen, y1);
      ctx.lineTo(x2, y1);
      ctx.lineTo(x2, y1 + cornerLen);
      ctx.stroke();
      // Bottom-left
      ctx.beginPath();
      ctx.moveTo(x1, y2 - cornerLen);
      ctx.lineTo(x1, y2);
      ctx.lineTo(x1 + cornerLen, y2);
      ctx.stroke();
      // Bottom-right
      ctx.beginPath();
      ctx.moveTo(x2 - cornerLen, y2);
      ctx.lineTo(x2, y2);
      ctx.lineTo(x2, y2 - cornerLen);
      ctx.stroke();

      // Center crosshair marker
      const cx = (x1 + x2) / 2;
      const cy = (y1 + y2) / 2;
      const crossSize = 7;
      ctx.strokeStyle = 'rgba(0, 255, 200, 0.6)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(cx - crossSize, cy);
      ctx.lineTo(cx + crossSize, cy);
      ctx.moveTo(cx, cy - crossSize);
      ctx.lineTo(cx, cy + crossSize);
      ctx.stroke();

      // PERSON label pill
      const conf = (detection.confidence * 100).toFixed(0);
      const label = `PERSON ${conf}%`;
      ctx.font = 'bold 11px "Courier New", monospace';
      const textW = ctx.measureText(label).width;
      const pillPad = 8;
      const pillH = 20;
      const pillW = textW + pillPad * 2;
      const pillX = x1;
      const pillY = y1 - pillH - 5;
      const pillR = 3;

      // Pill background (rounded rect)
      ctx.fillStyle = 'rgba(0, 255, 200, 0.9)';
      ctx.beginPath();
      ctx.moveTo(pillX + pillR, pillY);
      ctx.lineTo(pillX + pillW - pillR, pillY);
      ctx.quadraticCurveTo(pillX + pillW, pillY, pillX + pillW, pillY + pillR);
      ctx.lineTo(pillX + pillW, pillY + pillH - pillR);
      ctx.quadraticCurveTo(pillX + pillW, pillY + pillH, pillX + pillW - pillR, pillY + pillH);
      ctx.lineTo(pillX + pillR, pillY + pillH);
      ctx.quadraticCurveTo(pillX, pillY + pillH, pillX, pillY + pillH - pillR);
      ctx.lineTo(pillX, pillY + pillR);
      ctx.quadraticCurveTo(pillX, pillY, pillX + pillR, pillY);
      ctx.closePath();
      ctx.fill();

      // Small triangle pointer from pill to box
      ctx.beginPath();
      ctx.moveTo(x1 + 8, pillY + pillH);
      ctx.lineTo(x1 + 14, y1);
      ctx.lineTo(x1 + 20, pillY + pillH);
      ctx.closePath();
      ctx.fill();

      // Pill text
      ctx.fillStyle = '#000000';
      ctx.fillText(label, pillX + pillPad, pillY + 14);
    });

    // Draw tracks (yellow boxes with trail)
    ctx.strokeStyle = '#ffff00';
    ctx.lineWidth = 3;
    ctx.fillStyle = 'rgba(255, 255, 0, 0.15)';

    tracks.forEach(track => {
      const [x1, y1, x2, y2] = track.bbox;
      const width = x2 - x1;
      const height = y2 - y1;

      // Draw bounding box
      ctx.strokeRect(x1, y1, width, height);
      
      // Draw center point
      const centerX = (x1 + x2) / 2;
      const centerY = (y1 + y2) / 2;
      ctx.fillStyle = '#ffff00';
      ctx.beginPath();
      ctx.arc(centerX, centerY, 4, 0, 2 * Math.PI);
      ctx.fill();

      // Draw velocity vector
      if (track.velocity) {
        const [vx, vy] = track.velocity;
        const scale = 10; // Scale factor for visualization
        ctx.strokeStyle = '#ffff00';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(centerX + vx * scale, centerY + vy * scale);
        ctx.stroke();
        
        // Arrow head
        const angle = Math.atan2(vy, vx);
        const arrowLength = 8;
        ctx.beginPath();
        ctx.moveTo(centerX + vx * scale, centerY + vy * scale);
        ctx.lineTo(
          centerX + vx * scale - arrowLength * Math.cos(angle - Math.PI / 6),
          centerY + vy * scale - arrowLength * Math.sin(angle - Math.PI / 6)
        );
        ctx.moveTo(centerX + vx * scale, centerY + vy * scale);
        ctx.lineTo(
          centerX + vx * scale - arrowLength * Math.cos(angle + Math.PI / 6),
          centerY + vy * scale - arrowLength * Math.sin(angle + Math.PI / 6)
        );
        ctx.stroke();
      }

      // Draw track ID
      ctx.fillStyle = '#ffff00';
      ctx.font = 'bold 12px Arial';
      ctx.fillText(`Track ${track.track_id}`, x1, y1 - 5);
    });

    // Draw predictions (red dashed boxes with ghost effect). Style predicted vs observed differently.
    ctx.setLineDash([5, 5]);
    ctx.lineWidth = 2;

    predictions.forEach(prediction => {
      const [x1, y1, x2, y2] = prediction.bbox;
      const width = x2 - x1;
      const height = y2 - y1;

      const isInferred = !!prediction.inferred;

      if (isInferred) {
        // Ghosted inferred overlay
        ctx.strokeStyle = 'rgba(255, 100, 100, 0.9)';
        ctx.fillStyle = 'rgba(255, 100, 100, 0.08)';
      } else {
        // Stronger visible prediction (should be rare)
        ctx.strokeStyle = 'rgba(255, 0, 0, 0.95)';
        ctx.fillStyle = 'rgba(255, 0, 0, 0.12)';
      }

      // Draw dashed bounding box and fill
      ctx.strokeRect(x1, y1, width, height);
      ctx.fillRect(x1, y1, width, height);

      // Draw prediction center with pulsing effect for inferred
      const centerX = (x1 + x2) / 2;
      const centerY = (y1 + y2) / 2;
      const pulse = Math.sin(Date.now() / 200) * 0.3 + 0.7;

      ctx.beginPath();
      if (isInferred) {
        ctx.fillStyle = `rgba(255, 100, 100, ${pulse * 0.9})`;
        ctx.arc(centerX, centerY, 6, 0, 2 * Math.PI);
        ctx.fill();
      } else {
        ctx.fillStyle = `rgba(255, 0, 0, ${pulse})`;
        ctx.arc(centerX, centerY, 6, 0, 2 * Math.PI);
        ctx.fill();
      }

      // Draw confidence and time info
      ctx.setLineDash([]); // Reset line dash
      ctx.fillStyle = isInferred ? 'rgba(255, 100, 100, 0.9)' : 'rgba(255, 0, 0, 0.9)';
      ctx.font = '12px Arial';
      const predLabel = `${isInferred ? 'Predicted' : 'Pred'} ${(prediction.confidence * 100).toFixed(0)}%`;
      const timeLabel = `${prediction.time_since_seen.toFixed(1)}s ago`;

      ctx.fillRect(x1, y1 - 35, 140, 30);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(predLabel, x1 + 4, y1 - 20);
      ctx.fillText(timeLabel, x1 + 4, y1 - 8);

      // Restore dashed setting for next prediction
      ctx.setLineDash([5, 5]);
    });

    // Reset line dash
    ctx.setLineDash([]);

  }, [detections, tracks, predictions]);

  // Handle frame data update
  useEffect(() => {
    if (!frameData || !imageRef.current) return;

    try {
      // Convert frame data to blob and create object URL
      const blob = new Blob([frameData], { type: 'image/jpeg' });
      const imageUrl = URL.createObjectURL(blob);
      
      const img = imageRef.current;
      img.onload = () => {
        // Update canvas dimensions to match image
        if (canvasRef.current) {
          canvasRef.current.width = img.naturalWidth;
          canvasRef.current.height = img.naturalHeight;
          setDimensions({ width: img.naturalWidth, height: img.naturalHeight });
        }
        
        // Draw overlays after image loads
        setTimeout(drawOverlays, 10);
        
        // Clean up object URL
        URL.revokeObjectURL(imageUrl);
      };
      
      img.src = imageUrl;

    } catch (error) {
      console.error('Error loading frame data:', error);
    }
  }, [frameData, drawOverlays]);

  // Redraw overlays when detection data changes
  useEffect(() => {
    drawOverlays();
  }, [drawOverlays]);

  return (
    <div className={`camera-display ${isActive ? 'active' : 'inactive'}`}>
      <div className="camera-header">
        <h3>Camera {cameraId}</h3>
        <div className="camera-stats">
          <span className={`status ${isActive ? 'online' : 'offline'}`}>
            {isActive ? '🟢 LIVE' : '🔴 OFFLINE'}
          </span>
          <span>Det: {stats.detectionsCount}</span>
          <span>Track: {stats.tracksCount}</span>
          <span>Pred: {stats.predictionsCount}</span>
        </div>
      </div>
      
      <div className="camera-viewport" style={{ position: 'relative' }}>
        {/* Background image */}
        <img
          ref={imageRef}
          alt={`Camera ${cameraId}`}
          style={{
            width: '100%',
            height: 'auto',
            display: frameData ? 'block' : 'none'
          }}
        />
        
        {/* Overlay canvas */}
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: 'auto',
            pointerEvents: 'none'
          }}
          width={dimensions.width}
          height={dimensions.height}
        />
        
        {/* Placeholder when no frame data */}
        {!frameData && (
          <div className="no-signal">
            <div className="no-signal-content">
              <span>📷</span>
              <p>Camera {cameraId}</p>
              <p>{isActive ? 'Waiting for signal...' : 'Camera offline'}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default CameraDisplay;