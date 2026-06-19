/**
 * Camera Display component with tactical overlay
 * EagleEye-style HUD for detection visualization
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { getConfig } from '../config';
import { IFF_COLORS, COCO_SKELETON, KEYPOINT_CONFIDENCE_THRESHOLD } from '../domain/entities';
import './CameraDisplay.css';

/**
 * Get IFF color based on detection/track/prediction type
 */
export function getIFFColor(item, isPrediction = false) {
  if (isPrediction) {
    const method = item.prediction_method || (item.homography_source ? 'HOMOGRAPHY' : 'EXTRAP');
    if (method === 'HOMOGRAPHY') return IFF_COLORS.PROJECTED;
    if (method === 'WORLD') return IFF_COLORS.WORLD;
    return IFF_COLORS.HOSTILE;
  }
  return IFF_COLORS.FRIENDLY;
}

/**
 * Estimate distance from bbox height
 */
export function estimateDistance(bboxHeight) {
  const personHeight = getConfig('display.personHeightMeters', 1.7);
  const refPixels = getConfig('display.referencePixelsAt1M', 520);
  
  if (bboxHeight <= 0) return null;
  return Math.round((personHeight * refPixels) / bboxHeight * 10) / 10;
}

/**
 * Camera Display Component
 */
function CameraDisplay({ 
  cameraId, 
  frameData, 
  detections = [], 
  tracks = [], 
  predictions = [],
  isActive = false 
}) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const prevBlobUrlRef = useRef(null);
  const [stats, setStats] = useState({
    detectionsCount: 0, 
    tracksCount: 0, 
    predictionsCount: 0 
  });

  // Update stats
  useEffect(() => {
    setStats({
      detectionsCount: detections.length,
      tracksCount: tracks.length,
      predictionsCount: predictions.length
    });
  }, [detections, tracks, predictions]);

  /**
   * Draw diamond marker
   */
  const drawDiamond = useCallback((ctx, cx, cy, size, iff, filled) => {
    ctx.save();
    
    // Dark backdrop
    ctx.fillStyle = iff.dim;
    ctx.beginPath();
    ctx.arc(cx, cy, size * 0.9, 0, 2 * Math.PI);
    ctx.fill();
    
    ctx.translate(cx, cy);
    ctx.rotate(Math.PI / 4);
    ctx.strokeStyle = iff.stroke;
    ctx.lineWidth = 3;
    ctx.shadowColor = iff.glow;
    ctx.shadowBlur = 14;
    
    if (filled) {
      ctx.fillStyle = iff.bg;
      ctx.fillRect(-size / 2, -size / 2, size, size);
    }
    
    ctx.strokeRect(-size / 2, -size / 2, size, size);
    ctx.restore();
  }, []);

  /**
   * Draw corner brackets
   */
  const drawCornerBrackets = useCallback((ctx, x1, y1, x2, y2, iff) => {
    ctx.save();
    ctx.strokeStyle = iff.stroke;
    ctx.lineWidth = 3;
    ctx.lineCap = 'square';
    ctx.shadowColor = iff.glow;
    ctx.shadowBlur = 10;
    
    const len = Math.max(20, Math.min(30, (x2 - x1) * 0.2, (y2 - y1) * 0.2));
    
    // Top-left
    ctx.beginPath();
    ctx.moveTo(x1, y1 + len);
    ctx.lineTo(x1, y1);
    ctx.lineTo(x1 + len, y1);
    ctx.stroke();
    
    // Top-right
    ctx.beginPath();
    ctx.moveTo(x2 - len, y1);
    ctx.lineTo(x2, y1);
    ctx.lineTo(x2, y1 + len);
    ctx.stroke();
    
    // Bottom-left
    ctx.beginPath();
    ctx.moveTo(x1, y2 - len);
    ctx.lineTo(x1, y2);
    ctx.lineTo(x1 + len, y2);
    ctx.stroke();
    
    // Bottom-right
    ctx.beginPath();
    ctx.moveTo(x2 - len, y2);
    ctx.lineTo(x2, y2);
    ctx.lineTo(x2, y2 - len);
    ctx.stroke();
    
    ctx.restore();
  }, []);

  /**
   * Draw tactical pill with label
   */
  const drawTacticalPill = useCallback((ctx, cx, topY, label, sublabel, iff) => {
    ctx.save();
    ctx.font = 'bold 12px "Consolas", "Courier New", monospace';
    const tw = ctx.measureText(label).width;
    const stw = sublabel ? ctx.measureText(sublabel).width : 0;
    
    const pad = 8;
    const h = sublabel ? 32 : 20;
    const w = Math.max(tw, stw) + pad * 2;
    const px = cx - w / 2;
    const py = topY - h - 10;
    
    // Background
    ctx.fillStyle = 'rgba(0,0,0,0.75)';
    ctx.beginPath();
    ctx.roundRect(px, py, w, h, 4);
    ctx.fill();
    
    // Border
    ctx.strokeStyle = iff.stroke;
    ctx.lineWidth = 2;
    ctx.shadowColor = iff.glow;
    ctx.shadowBlur = 8;
    ctx.stroke();
    ctx.shadowBlur = 0;
    
    // Label
    ctx.fillStyle = iff.stroke;
    ctx.font = 'bold 12px "Consolas", "Courier New", monospace';
    ctx.fillText(label, px + pad, py + 15);
    
    // Sublabel
    if (sublabel) {
      ctx.font = '10px "Consolas", "Courier New", monospace';
      ctx.fillStyle = 'rgba(255,255,255,0.8)';
      ctx.fillText(sublabel, px + pad, py + 27);
    }
    
    // Connector line
    ctx.strokeStyle = iff.stroke;
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.6;
    ctx.beginPath();
    ctx.moveTo(cx, py + h);
    ctx.lineTo(cx, topY);
    ctx.stroke();
    ctx.globalAlpha = 1;
    
    ctx.restore();
  }, []);

  /**
   * Draw COCO skeleton
   */
  const drawSkeleton = useCallback((ctx, keypoints, color, lineWidth) => {
    if (!keypoints || keypoints.length < 17) return;
    
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    
    COCO_SKELETON.forEach(([i, j]) => {
      const a = keypoints[i];
      const b = keypoints[j];
      
      if (!a || !b || a[2] < KEYPOINT_CONFIDENCE_THRESHOLD || b[2] < KEYPOINT_CONFIDENCE_THRESHOLD) {
        return;
      }
      
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
    });
    
    // Draw keypoints
    keypoints.forEach(kp => {
      if (!kp || kp[2] < KEYPOINT_CONFIDENCE_THRESHOLD) return;
      
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(kp[0], kp[1], 3, 0, 2 * Math.PI);
      ctx.fill();
    });
    
    ctx.restore();
  }, []);

  /**
   * Draw velocity arrow
   */
  const drawVelocityArrow = useCallback((ctx, cx, cy, vx, vy, iff) => {
    const scale = 18;
    const ex = cx + vx * scale;
    const ey = cy + vy * scale;
    
    if (Math.sqrt(vx * vx + vy * vy) <= 0.3) return;
    
    const angle = Math.atan2(vy, vx);
    const arrowLen = 9;
    
    ctx.save();
    ctx.strokeStyle = iff.stroke;
    ctx.lineWidth = 2.5;
    ctx.shadowColor = iff.glow;
    ctx.shadowBlur = 6;
    
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(ex, ey);
    ctx.stroke();
    
    // Arrowhead
    ctx.beginPath();
    ctx.moveTo(ex, ey);
    ctx.lineTo(
      ex - arrowLen * Math.cos(angle - Math.PI / 5),
      ey - arrowLen * Math.sin(angle - Math.PI / 5)
    );
    ctx.moveTo(ex, ey);
    ctx.lineTo(
      ex - arrowLen * Math.cos(angle + Math.PI / 5),
      ey - arrowLen * Math.sin(angle + Math.PI / 5)
    );
    ctx.stroke();
    
    ctx.restore();
  }, []);

  /**
   * Main draw function
   */
  const drawOverlays = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    
    ctx.clearRect(0, 0, W, H);

    // Draw detections
    detections.forEach(det => {
      const [x1, y1, x2, y2] = det.bbox;
      const cx = (x1 + x2) / 2;
      const cy = (y1 + y2) / 2;
      const height = y2 - y1;
      const distance = estimateDistance(height);
      const iff = IFF_COLORS.UNKNOWN;

      // Dark tint
      ctx.fillStyle = 'rgba(0,0,0,0.2)';
      ctx.fillRect(x1, y1, x2 - x1, height);

      // Skeleton
      if (det.keypoints) {
        drawSkeleton(ctx, det.keypoints, 'rgba(255,191,0,0.7)', 2);
      }

      // Brackets and diamond
      drawCornerBrackets(ctx, x1, y1, x2, y2, iff);
      drawDiamond(ctx, cx, cy, 18, iff, false);

      // Label
      const conf = (det.confidence * 100).toFixed(0);
      drawTacticalPill(ctx, cx, y1, `PERSON · ${conf}%`, distance ? `~${distance}m` : null, iff);
    });

    // Draw tracks
    tracks.forEach(track => {
      const [x1, y1, x2, y2] = track.bbox;
      const cx = (x1 + x2) / 2;
      const cy = (y1 + y2) / 2;
      const height = y2 - y1;
      const distance = estimateDistance(height);
      const iff = IFF_COLORS.FRIENDLY;

      // Dark tint
      ctx.fillStyle = 'rgba(0,0,0,0.15)';
      ctx.fillRect(x1, y1, x2 - x1, height);

      // Skeleton
      if (track.keypoints) {
        drawSkeleton(ctx, track.keypoints, 'rgba(0,160,255,0.7)', 2);
      }

      // Brackets and diamond
      drawCornerBrackets(ctx, x1, y1, x2, y2, iff);
      drawDiamond(ctx, cx, cy, 20, iff, true);

      // Velocity arrow
      if (track.velocity) {
        drawVelocityArrow(ctx, cx, cy, track.velocity[0], track.velocity[1], iff);
      }

      // Label
      drawTacticalPill(ctx, cx, y1, `T-${track.track_id}`, distance ? `~${distance}m · ${track.age || 0} hits` : `${track.age || 0} hits`, iff);
    });

    // Draw predictions
    predictions.forEach(pred => {
      const [x1, y1, x2, y2] = pred.bbox;
      const cx = (x1 + x2) / 2;
      const cy = (y1 + y2) / 2;
      const width = x2 - x1;
      const height = y2 - y1;
      
      const iff = getIFFColor(pred, true);
      const method = pred.prediction_method || (pred.homography_source ? 'HOMOGRAPHY' : 'EXTRAP');
      const tag = method === 'HOMOGRAPHY' ? 'H-PROJ' : method === 'WORLD' ? 'WORLD' : 'EXTRAP';

      // Dark tint
      ctx.save();
      ctx.fillStyle = 'rgba(0,0,0,0.3)';
      ctx.strokeStyle = iff.stroke;
      ctx.lineWidth = 2;
      ctx.shadowColor = iff.glow;
      ctx.shadowBlur = 12;
      
      if (method !== 'HOMOGRAPHY') {
        ctx.setLineDash([6, 4]);
      }
      
      ctx.fillRect(x1, y1, width, height);
      ctx.strokeRect(x1, y1, width, height);
      ctx.restore();

      // Skeleton
      if (pred.keypoints) {
        drawSkeleton(ctx, pred.keypoints, iff.stroke, 2.5);
      }

      // Brackets
      ctx.save();
      if (method !== 'HOMOGRAPHY') {
        ctx.setLineDash([5, 4]);
      }
      drawCornerBrackets(ctx, x1, y1, x2, y2, iff);
      ctx.restore();

      // Chevron
      ctx.save();
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.beginPath();
      ctx.arc(cx, cy, 22, 0, 2 * Math.PI);
      ctx.fill();
      
      ctx.translate(cx, cy);
      ctx.strokeStyle = iff.stroke;
      ctx.fillStyle = iff.bg;
      ctx.lineWidth = 3;
      ctx.shadowColor = iff.glow;
      ctx.shadowBlur = 16;
      
      const cs = 16;
      ctx.beginPath();
      ctx.moveTo(0, -cs);
      ctx.lineTo(cs * 0.8, cs * 0.4);
      ctx.lineTo(-cs * 0.8, cs * 0.4);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.restore();

      // Label
      const conf = (pred.confidence * 100).toFixed(0);
      const timeAgo = pred.time_since_seen?.toFixed(1) || '?';
      drawTacticalPill(ctx, cx, y1, `${tag} ${conf}%`, `Cam ${pred.source_camera} · ${timeAgo}s ago`, iff);
    });

    // HUD corners
    ctx.save();
    ctx.strokeStyle = IFF_COLORS.HUD.stroke;
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.5;
    const hc = 24;
    
    ctx.beginPath();
    ctx.moveTo(3, hc);
    ctx.lineTo(3, 3);
    ctx.lineTo(hc, 3);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(W - hc, 3);
    ctx.lineTo(W - 3, 3);
    ctx.lineTo(W - 3, hc);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(3, H - hc);
    ctx.lineTo(3, H - 3);
    ctx.lineTo(hc, H - 3);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(W - hc, H - 3);
    ctx.lineTo(W - 3, H - 3);
    ctx.lineTo(W - 3, H - hc);
    ctx.stroke();
    
    ctx.restore();
  }, [detections, tracks, predictions, drawDiamond, drawCornerBrackets, drawTacticalPill, drawSkeleton, drawVelocityArrow]);

  // Handle frame data changes
  useEffect(() => {
    if (!frameData || !imageRef.current) return;

    try {
      const blob = new Blob([frameData], { type: 'image/jpeg' });
      const url = URL.createObjectURL(blob);
      if (prevBlobUrlRef.current) {
        URL.revokeObjectURL(prevBlobUrlRef.current);
      }
      prevBlobUrlRef.current = url;

      const img = imageRef.current;

      img.onload = () => {
        if (canvasRef.current) {
          canvasRef.current.width = img.naturalWidth;
          canvasRef.current.height = img.naturalHeight;
        }
        drawOverlays();
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        if (prevBlobUrlRef.current === url) prevBlobUrlRef.current = null;
      };

      img.src = url;
    } catch (e) {
      console.error('Frame load error:', e);
    }
  }, [frameData, drawOverlays]);

  // Revoke any outstanding blob URL on unmount
  useEffect(() => () => {
    if (prevBlobUrlRef.current) {
      URL.revokeObjectURL(prevBlobUrlRef.current);
      prevBlobUrlRef.current = null;
    }
  }, []);

  // Redraw overlays when detections/tracks/predictions change
  useEffect(() => {
    drawOverlays();
  }, [drawOverlays]);

  return (
    <div className={`camera-display ${isActive ? 'active' : 'inactive'}`}>
      <div className="camera-header">
        <div className="camera-header-left">
          <span className="cam-id">CAM-{cameraId}</span>
          <span className={`status-badge ${isActive ? 'live' : 'off'}`}>
            {isActive ? 'LIVE' : 'OFFLINE'}
          </span>
        </div>
        <div className="camera-header-right">
          <span className="hud-stat"><em>{stats.detectionsCount}</em> DET</span>
          <span className="hud-stat"><em>{stats.tracksCount}</em> TRK</span>
          <span className="hud-stat"><em>{stats.predictionsCount}</em> PRED</span>
        </div>
      </div>
      
      <div className="camera-viewport">
        <img 
          ref={imageRef} 
          alt={`Camera ${cameraId}`}
          style={{ width: '100%', height: 'auto', display: frameData ? 'block' : 'none' }} 
        />
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
        />
        {frameData && <div className="scanline-overlay" />}
        
        {!frameData && (
          <div className="no-signal">
            <div className="no-signal-content">
              <div className="no-signal-icon">◆</div>
              <p className="no-signal-cam">CAM-{cameraId}</p>
              <p className="no-signal-msg">{isActive ? 'AWAITING SIGNAL' : 'SENSOR OFFLINE'}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default CameraDisplay;
