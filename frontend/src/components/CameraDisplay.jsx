/**
 * Camera display component with AR overlays + COCO skeleton stick-figure renderer
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';

// ── COCO Skeleton Topology ──────────────────────────────────────────────
// 17 keypoints: nose(0), left_eye(1), right_eye(2), left_ear(3), right_ear(4),
// left_shoulder(5), right_shoulder(6), left_elbow(7), right_elbow(8),
// left_wrist(9), right_wrist(10), left_hip(11), right_hip(12),
// left_knee(13), right_knee(14), left_ankle(15), right_ankle(16)
const COCO_SKELETON = [
  [0, 1], [0, 2], [1, 3], [2, 4],           // head
  [5, 6],                                     // shoulders
  [5, 7], [7, 9],                             // left arm
  [6, 8], [8, 10],                            // right arm
  [5, 11], [6, 12],                           // torso
  [11, 12],                                   // hips
  [11, 13], [13, 15],                         // left leg
  [12, 14], [14, 16],                         // right leg
  [0, 5], [0, 6],                             // nose → shoulders (neck approx)
  [3, 5], [4, 6],                             // ears → shoulders (optional)
];

const KP_CONF_THRESHOLD = 0.25; // minimum joint confidence to draw

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

  // ── Skeleton stick-figure drawer ─────────────────────────────────────
  const drawSkeleton = useCallback((ctx, keypoints, color, lineWidth, ghost = false) => {
    if (!keypoints || keypoints.length < 17) return;

    ctx.save();

    if (ghost) {
      ctx.shadowColor = color;
      ctx.shadowBlur = 14;
      ctx.setLineDash([6, 4]);
    }

    // Draw limb bones
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    COCO_SKELETON.forEach(([i, j]) => {
      const a = keypoints[i];
      const b = keypoints[j];
      if (!a || !b) return;
      if (a[2] < KP_CONF_THRESHOLD || b[2] < KP_CONF_THRESHOLD) return;
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
    });

    // Draw joint dots
    ctx.setLineDash([]);
    ctx.shadowBlur = 0;
    keypoints.forEach(kp => {
      if (!kp || kp[2] < KP_CONF_THRESHOLD) return;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(kp[0], kp[1], ghost ? 4 : 3, 0, 2 * Math.PI);
      ctx.fill();
    });

    ctx.restore();
  }, []);

  // Draw overlays on canvas
  const drawOverlays = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // ── Phase 1: Detections — cyan brackets + optional skeleton ──────
    detections.forEach(detection => {
      const [x1, y1, x2, y2] = detection.bbox;
      const w = x2 - x1;
      const h = y2 - y1;
      const cornerLen = Math.min(22, w * 0.18, h * 0.18);

      // Subtle full bounding box
      ctx.strokeStyle = 'rgba(0, 255, 200, 0.35)';
      ctx.lineWidth = 1;
      ctx.setLineDash([]);
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

      // Skeleton overlay (cyan) if keypoints available
      if (detection.keypoints && detection.keypoints.length >= 17) {
        drawSkeleton(ctx, detection.keypoints, '#00ffc8', 2, false);
      }

      // Center crosshair marker
      const cx = (x1 + x2) / 2;
      const cy = (y1 + y2) / 2;
      const crossSize = 7;
      ctx.strokeStyle = 'rgba(0, 255, 200, 0.6)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([]);
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

    // ── Phase 2: Tracks — yellow box + optional skeleton ─────────────
    ctx.strokeStyle = '#ffff00';
    ctx.lineWidth = 3;
    ctx.fillStyle = 'rgba(255, 255, 0, 0.15)';
    ctx.setLineDash([]);

    tracks.forEach(track => {
      const [x1, y1, x2, y2] = track.bbox;
      const width = x2 - x1;
      const height = y2 - y1;

      // Draw bounding box
      ctx.strokeStyle = '#ffff00';
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, width, height);

      // Skeleton overlay (yellow) if keypoints available
      if (track.keypoints && track.keypoints.length >= 17) {
        drawSkeleton(ctx, track.keypoints, '#ffff00', 2, false);
      }

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
        const scale = 10;
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

    // ── Phase 3: Predictions / Ghost overlays ─────────────────────────
    predictions.forEach(prediction => {
      const [x1, y1, x2, y2] = prediction.bbox;
      const width = x2 - x1;
      const height = y2 - y1;
      const centerX = (x1 + x2) / 2;
      const centerY = (y1 + y2) / 2;
      const hasKeypoints = prediction.keypoints && prediction.keypoints.length >= 17;
      const srcCam = prediction.source_camera ?? -1;
      const isHomography = prediction.homography_source === true;

      // Color scheme: green for homography (accurate), red for extrapolation (guess)
      const ghostColor = isHomography ? 'rgba(0, 255, 120, 0.85)' : 'rgba(255, 100, 100, 0.9)';
      const bgColor = isHomography ? 'rgba(0, 255, 120, 0.75)' : 'rgba(255, 100, 100, 0.75)';
      const methodTag = isHomography ? 'H-PROJ' : 'EXTRAP';

      if (hasKeypoints) {
        // ─── GHOST SKELETON MODE ───────────────────────────────
        drawSkeleton(ctx, prediction.keypoints, ghostColor, 3, true);

        ctx.save();
        ctx.setLineDash([]);
        ctx.shadowBlur = 0;
        const ghostLabel = srcCam >= 0
          ? `GHOST [${methodTag}] — Cam ${srcCam}`
          : `GHOST [${methodTag}]`;
        ctx.font = 'bold 11px "Courier New", monospace';
        const tw = ctx.measureText(ghostLabel).width;
        const lx = centerX - tw / 2 - 6;
        const ly = y1 - 26;
        const r = 3;
        const lw = tw + 12;
        const lh = 20;

        ctx.fillStyle = bgColor;
        ctx.beginPath();
        ctx.moveTo(lx + r, ly);
        ctx.lineTo(lx + lw - r, ly);
        ctx.quadraticCurveTo(lx + lw, ly, lx + lw, ly + r);
        ctx.lineTo(lx + lw, ly + lh - r);
        ctx.quadraticCurveTo(lx + lw, ly + lh, lx + lw - r, ly + lh);
        ctx.lineTo(lx + r, ly + lh);
        ctx.quadraticCurveTo(lx, ly + lh, lx, ly + lh - r);
        ctx.lineTo(lx, ly + r);
        ctx.quadraticCurveTo(lx, ly, lx + r, ly);
        ctx.closePath();
        ctx.fill();

        ctx.fillStyle = '#000';
        ctx.fillText(ghostLabel, lx + 6, ly + 14);

        const infoLabel = `${(prediction.confidence * 100).toFixed(0)}% · ${prediction.time_since_seen.toFixed(1)}s ago`;
        ctx.fillStyle = ghostColor;
        ctx.font = '10px "Courier New", monospace';
        ctx.fillText(infoLabel, centerX - ctx.measureText(infoLabel).width / 2, y2 + 14);
        ctx.restore();
      } else {
        // ─── BOX MODE (no keypoints) ───────────────────────────
        ctx.save();

        if (isHomography) {
          // Homography: solid green box with glow
          ctx.shadowColor = '#00ff78';
          ctx.shadowBlur = 10;
          ctx.setLineDash([]);
          ctx.lineWidth = 2;
          ctx.strokeStyle = ghostColor;
          ctx.fillStyle = 'rgba(0, 255, 120, 0.06)';
        } else {
          // Extrapolation: red dashed box
          ctx.setLineDash([5, 5]);
          ctx.lineWidth = 2;
          ctx.strokeStyle = ghostColor;
          ctx.fillStyle = 'rgba(255, 100, 100, 0.08)';
        }
        ctx.strokeRect(x1, y1, width, height);
        ctx.fillRect(x1, y1, width, height);

        // Pulsing center dot
        ctx.shadowBlur = 0;
        const pulse = Math.sin(Date.now() / 200) * 0.3 + 0.7;
        ctx.fillStyle = isHomography
          ? `rgba(0, 255, 120, ${pulse * 0.9})`
          : `rgba(255, 100, 100, ${pulse * 0.9})`;
        ctx.beginPath();
        ctx.arc(centerX, centerY, 6, 0, 2 * Math.PI);
        ctx.fill();

        // Info label
        ctx.setLineDash([]);
        ctx.fillStyle = bgColor;
        ctx.fillRect(x1, y1 - 40, 180, 35);
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 11px "Courier New", monospace';
        const predLabel = `${methodTag} ${(prediction.confidence * 100).toFixed(0)}% — Cam ${srcCam}`;
        const timeLabel = `${prediction.time_since_seen.toFixed(1)}s ago`;
        ctx.fillText(predLabel, x1 + 4, y1 - 24);
        ctx.fillText(timeLabel, x1 + 4, y1 - 10);
        ctx.restore();
      }
    });

    // Reset line dash
    ctx.setLineDash([]);

  }, [detections, tracks, predictions, drawSkeleton]);

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
        drawOverlays();
        
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