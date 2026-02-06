/**
 * CameraDisplay — EagleEye-inspired Tactical Perception Overlay
 * 
 * Replaces developer bounding boxes with military-grade ambient AR markers:
 *  ◆ Diamond/chevron entity markers with IFF color coding
 *  ◆ Distance estimation from bbox height (~1.7m assumed person height)
 *  ◆ BLOS (Beyond Line Of Sight) edge-clamped directional indicators
 *  ◆ Compass bearing ribbon (when orientation data available)
 *  ◆ Threat awareness arc on viewport edge
 *  ◆ Minimal COCO skeleton in ghost-mode for predictions
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';

// ── IFF Color System ────────────────────────────────────────────────────
// Inspired by Anduril EagleEye IFF classification
const IFF = {
  // Detection — unclassified, first-seen
  UNKNOWN:     { fill: 'rgba(255,191,0,0.9)',   stroke: '#ffbf00', bg: 'rgba(255,191,0,0.18)', glow: 'rgba(255,191,0,0.5)',  text: '#000' },
  // Tracked — identified & being followed  
  FRIENDLY:    { fill: 'rgba(0,160,255,0.9)',    stroke: '#00a0ff', bg: 'rgba(0,160,255,0.12)', glow: 'rgba(0,160,255,0.45)', text: '#fff' },
  // Prediction via homography (high confidence cross-cam)
  PROJECTED:   { fill: 'rgba(0,255,130,0.9)',    stroke: '#00ff82', bg: 'rgba(0,255,130,0.10)', glow: 'rgba(0,255,130,0.5)',  text: '#000' },
  // Prediction via extrapolation (lower confidence)
  HOSTILE:     { fill: 'rgba(255,80,80,0.9)',     stroke: '#ff5050', bg: 'rgba(255,80,80,0.12)', glow: 'rgba(255,80,80,0.5)', text: '#fff' },
  // UI accent
  HUD:         { stroke: 'rgba(0,200,220,0.6)',  fill: 'rgba(0,200,220,0.15)', text: '#00c8dc' },
};

// ── COCO Skeleton Topology ──────────────────────────────────────────────
const COCO_SKELETON = [
  [0, 1], [0, 2], [1, 3], [2, 4],           // head
  [5, 6],                                     // shoulders
  [5, 7], [7, 9],                             // left arm
  [6, 8], [8, 10],                            // right arm
  [5, 11], [6, 12],                           // torso
  [11, 12],                                   // hips
  [11, 13], [13, 15],                         // left leg
  [12, 14], [14, 16],                         // right leg
  [0, 5], [0, 6],                             // neck approx
  [3, 5], [4, 6],                             // ears → shoulders
];
const KP_CONF = 0.25;

// ── Assumed person height for distance estimation (meters) ──────────────
const PERSON_HEIGHT_M = 1.7;
// Approximate vertical FOV reference height at 1m (calibrate per camera)
const REF_PX_AT_1M = 520;

// ─────────────────────────────────────────────────────────────────────────
const CameraDisplay = ({
  cameraId,
  frameData,
  detections = [],
  tracks = [],
  predictions = [],
  isActive = false,
  sensorData = null,   // { orientation: { alpha, beta, gamma } }
}) => {
  const canvasRef = useRef(null);
  const imageRef  = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 640, height: 480 });
  const [stats, setStats] = useState({ detectionsCount: 0, tracksCount: 0, predictionsCount: 0 });

  useEffect(() => {
    setStats({
      detectionsCount: detections.length,
      tracksCount: tracks.length,
      predictionsCount: predictions.length,
    });
  }, [detections, tracks, predictions]);

  // ── Helper: estimate distance from bounding-box height ───────────────
  const estimateDistance = useCallback((bboxHeight) => {
    if (bboxHeight <= 0) return null;
    const dist = (PERSON_HEIGHT_M * REF_PX_AT_1M) / bboxHeight;
    return Math.round(dist * 10) / 10;   // one decimal
  }, []);

  // ── Helper: draw a diamond (rotated square) ──────────────────────────
  const drawDiamond = useCallback((ctx, cx, cy, size, iff, filled = false) => {
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(Math.PI / 4);
    ctx.strokeStyle = iff.stroke;
    ctx.lineWidth = 2;
    ctx.shadowColor = iff.glow;
    ctx.shadowBlur = 8;

    if (filled) {
      ctx.fillStyle = iff.bg;
      ctx.fillRect(-size / 2, -size / 2, size, size);
    }
    ctx.strokeRect(-size / 2, -size / 2, size, size);

    ctx.restore();
  }, []);

  // ── Helper: draw tactical corner brackets ────────────────────────────
  const drawCornerBrackets = useCallback((ctx, x1, y1, x2, y2, iff, lineLen) => {
    ctx.save();
    ctx.strokeStyle = iff.stroke;
    ctx.lineWidth = 2;
    ctx.lineCap = 'square';
    ctx.shadowColor = iff.glow;
    ctx.shadowBlur = 4;

    const len = lineLen || Math.min(18, (x2 - x1) * 0.15, (y2 - y1) * 0.15);

    // Top-left
    ctx.beginPath();
    ctx.moveTo(x1, y1 + len); ctx.lineTo(x1, y1); ctx.lineTo(x1 + len, y1);
    ctx.stroke();
    // Top-right
    ctx.beginPath();
    ctx.moveTo(x2 - len, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + len);
    ctx.stroke();
    // Bottom-left
    ctx.beginPath();
    ctx.moveTo(x1, y2 - len); ctx.lineTo(x1, y2); ctx.lineTo(x1 + len, y2);
    ctx.stroke();
    // Bottom-right
    ctx.beginPath();
    ctx.moveTo(x2 - len, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - len);
    ctx.stroke();

    ctx.restore();
  }, []);

  // ── Helper: draw a compact tactical info pill ────────────────────────
  const drawTacticalPill = useCallback((ctx, cx, topY, label, sublabel, iff) => {
    ctx.save();
    ctx.font = 'bold 10px "Consolas", "Courier New", monospace';
    const tw = ctx.measureText(label).width;
    const stw = sublabel ? ctx.measureText(sublabel).width : 0;
    const pad = 6;
    const h = sublabel ? 28 : 16;
    const w = Math.max(tw, stw) + pad * 2;
    const px = cx - w / 2;
    const py = topY - h - 6;

    // Pill bg
    ctx.fillStyle = iff.bg;
    ctx.strokeStyle = iff.stroke;
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.92;

    // roundRect with fallback
    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(px, py, w, h, 3);
    } else {
      // Fallback for browsers without roundRect
      const r = 3;
      ctx.moveTo(px + r, py);
      ctx.lineTo(px + w - r, py);
      ctx.quadraticCurveTo(px + w, py, px + w, py + r);
      ctx.lineTo(px + w, py + h - r);
      ctx.quadraticCurveTo(px + w, py + h, px + w - r, py + h);
      ctx.lineTo(px + r, py + h);
      ctx.quadraticCurveTo(px, py + h, px, py + h - r);
      ctx.lineTo(px, py + r);
      ctx.quadraticCurveTo(px, py, px + r, py);
      ctx.closePath();
    }
    ctx.fill();
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Main label
    ctx.fillStyle = iff.stroke;
    ctx.fillText(label, px + pad, py + 12);

    // Sub-label
    if (sublabel) {
      ctx.font = '9px "Consolas", "Courier New", monospace';
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.fillText(sublabel, px + pad, py + 23);
    }

    // Thin connecting line to entity
    ctx.strokeStyle = iff.stroke;
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.4;
    ctx.beginPath();
    ctx.moveTo(cx, py + h);
    ctx.lineTo(cx, topY);
    ctx.stroke();
    ctx.globalAlpha = 1;

    ctx.restore();
  }, []);

  // ── Helper: draw chevron (directional wedge for BLOS indicators) ─────
  const drawEdgeChevron = useCallback((ctx, x, y, angle, size, iff) => {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);

    ctx.strokeStyle = iff.stroke;
    ctx.fillStyle = iff.bg;
    ctx.lineWidth = 2;
    ctx.shadowColor = iff.glow;
    ctx.shadowBlur = 10;

    ctx.beginPath();
    ctx.moveTo(0, -size);
    ctx.lineTo(size * 0.7, 0);
    ctx.lineTo(0, size);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    ctx.restore();
  }, []);

  // ── Helper: skeleton ghost renderer ──────────────────────────────────
  const drawSkeleton = useCallback((ctx, keypoints, color, lineWidth, ghost = false) => {
    if (!keypoints || keypoints.length < 17) return;
    ctx.save();
    if (ghost) {
      ctx.shadowColor = color;
      ctx.shadowBlur = 12;
      ctx.setLineDash([5, 4]);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    COCO_SKELETON.forEach(([i, j]) => {
      const a = keypoints[i], b = keypoints[j];
      if (!a || !b || a[2] < KP_CONF || b[2] < KP_CONF) return;
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    });
    ctx.setLineDash([]);
    ctx.shadowBlur = 0;
    keypoints.forEach(kp => {
      if (!kp || kp[2] < KP_CONF) return;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(kp[0], kp[1], ghost ? 3 : 2.5, 0, 2 * Math.PI);
      ctx.fill();
    });
    ctx.restore();
  }, []);

  // ── Helper: compass bearing ribbon ───────────────────────────────────
  const drawCompassRibbon = useCallback((ctx, canvasW, heading) => {
    if (heading == null) return;
    const ribbonH = 22;
    ctx.save();

    // Semi-transparent bar
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.fillRect(0, 0, canvasW, ribbonH);
    ctx.strokeStyle = IFF.HUD.stroke;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, ribbonH); ctx.lineTo(canvasW, ribbonH); ctx.stroke();

    // Cardinal and ordinal directions
    const cardinals = [
      { deg: 0, label: 'N' }, { deg: 45, label: 'NE' },
      { deg: 90, label: 'E' }, { deg: 135, label: 'SE' },
      { deg: 180, label: 'S' }, { deg: 225, label: 'SW' },
      { deg: 270, label: 'W' }, { deg: 315, label: 'NW' },
    ];

    ctx.font = 'bold 10px "Consolas", monospace';
    ctx.textAlign = 'center';
    const pixPerDeg = canvasW / 120;   // show ~120° of compass

    cardinals.forEach(({ deg, label }) => {
      let diff = deg - heading;
      while (diff > 180) diff -= 360;
      while (diff < -180) diff += 360;
      if (Math.abs(diff) > 60) return;

      const px = canvasW / 2 + diff * pixPerDeg;
      ctx.strokeStyle = label.length === 1 ? IFF.HUD.text : 'rgba(0,200,220,0.35)';
      ctx.lineWidth = label.length === 1 ? 2 : 1;
      ctx.beginPath(); ctx.moveTo(px, ribbonH - 6); ctx.lineTo(px, ribbonH); ctx.stroke();
      ctx.fillStyle = label.length === 1 ? IFF.HUD.text : 'rgba(0,200,220,0.5)';
      ctx.fillText(label, px, 13);
    });

    // Center bearing indicator
    ctx.fillStyle = IFF.HUD.text;
    ctx.beginPath();
    ctx.moveTo(canvasW / 2 - 5, ribbonH);
    ctx.lineTo(canvasW / 2, ribbonH - 5);
    ctx.lineTo(canvasW / 2 + 5, ribbonH);
    ctx.closePath();
    ctx.fill();

    // Heading readout
    ctx.font = 'bold 11px "Consolas", monospace';
    ctx.fillStyle = IFF.HUD.text;
    ctx.textAlign = 'right';
    ctx.fillText(`${Math.round(heading)}\u00B0`, canvasW - 8, 14);

    ctx.textAlign = 'start';
    ctx.restore();
  }, []);

  // ── Helper: threat awareness arc ─────────────────────────────────────
  const drawThreatRing = useCallback((ctx, canvasW, canvasH, preds) => {
    if (!preds || preds.length === 0) return;
    const cx = canvasW / 2;
    const cy = canvasH / 2;
    const radius = Math.min(canvasW, canvasH) / 2 - 4;

    ctx.save();
    preds.forEach(pred => {
      const [px1, py1, px2, py2] = pred.bbox;
      const pcx = (px1 + px2) / 2;
      const pcy = (py1 + py2) / 2;
      const angle = Math.atan2(pcy - cy, pcx - cx);
      const isHomography = pred.homography_source === true;
      const iff = isHomography ? IFF.PROJECTED : IFF.HOSTILE;

      ctx.strokeStyle = iff.stroke;
      ctx.lineWidth = 3;
      ctx.globalAlpha = 0.35;
      ctx.shadowColor = iff.glow;
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, angle - 0.15, angle + 0.15);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.shadowBlur = 0;
    });
    ctx.restore();
  }, []);

  // ══════════════════════════════════════════════════════════════════════
  //  MAIN DRAW LOOP
  // ══════════════════════════════════════════════════════════════════════
  const drawOverlays = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const now = Date.now();

    // ── Compass ribbon (top) ─────────────────────────────────────────
    const heading = sensorData?.orientation?.alpha ?? null;
    drawCompassRibbon(ctx, W, heading);

    // ── Threat awareness ring ────────────────────────────────────────
    drawThreatRing(ctx, W, H, predictions);

    // ── PHASE 1: DETECTIONS — Amber diamond + corner brackets ────────
    detections.forEach(det => {
      const [x1, y1, x2, y2] = det.bbox;
      const cx = (x1 + x2) / 2;
      const cy = (y1 + y2) / 2;
      const bboxH = y2 - y1;
      const dist = estimateDistance(bboxH);
      const iff = IFF.UNKNOWN;

      // Diamond marker at center
      drawDiamond(ctx, cx, cy, 14, iff, false);

      // Corner brackets
      drawCornerBrackets(ctx, x1, y1, x2, y2, iff);

      // Skeleton (subtle amber)
      if (det.keypoints && det.keypoints.length >= 17) {
        drawSkeleton(ctx, det.keypoints, 'rgba(255,191,0,0.55)', 1.5, false);
      }

      // Info pill
      const conf = (det.confidence * 100).toFixed(0);
      const mainLabel = `PERSON \u00B7 ${conf}%`;
      const subLabel = dist ? `~${dist}m` : null;
      drawTacticalPill(ctx, cx, y1, mainLabel, subLabel, iff);
    });

    // ── PHASE 2: TRACKS — Blue diamond + velocity bearing ────────────
    tracks.forEach(track => {
      const [x1, y1, x2, y2] = track.bbox;
      const cx = (x1 + x2) / 2;
      const cy = (y1 + y2) / 2;
      const bboxH = y2 - y1;
      const dist = estimateDistance(bboxH);
      const iff = IFF.FRIENDLY;

      // Diamond marker (filled)
      drawDiamond(ctx, cx, cy, 16, iff, true);

      // Corner brackets
      drawCornerBrackets(ctx, x1, y1, x2, y2, iff);

      // Skeleton (blue, solid)
      if (track.keypoints && track.keypoints.length >= 17) {
        drawSkeleton(ctx, track.keypoints, 'rgba(0,160,255,0.6)', 1.5, false);
      }

      // Velocity bearing arrow
      if (track.velocity) {
        const [vx, vy] = track.velocity;
        const mag = Math.sqrt(vx * vx + vy * vy);
        if (mag > 0.5) {
          const scale = 15;
          const endX = cx + vx * scale;
          const endY = cy + vy * scale;
          const angle = Math.atan2(vy, vx);
          const arrowLen = 7;

          ctx.save();
          ctx.strokeStyle = iff.stroke;
          ctx.lineWidth = 2;
          ctx.shadowColor = iff.glow;
          ctx.shadowBlur = 4;
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(endX, endY);
          ctx.stroke();

          // Arrowhead
          ctx.beginPath();
          ctx.moveTo(endX, endY);
          ctx.lineTo(endX - arrowLen * Math.cos(angle - Math.PI / 5), endY - arrowLen * Math.sin(angle - Math.PI / 5));
          ctx.moveTo(endX, endY);
          ctx.lineTo(endX - arrowLen * Math.cos(angle + Math.PI / 5), endY - arrowLen * Math.sin(angle + Math.PI / 5));
          ctx.stroke();
          ctx.restore();
        }
      }

      // Tactical pill
      const mainLabel = `T-${track.track_id}`;
      const subLabel = dist ? `~${dist}m \u00B7 ${track.age || 0} hits` : `${track.age || 0} hits`;
      drawTacticalPill(ctx, cx, y1, mainLabel, subLabel, iff);
    });

    // ── PHASE 3: PREDICTIONS / GHOSTS — Green chevron or Red chevron ─
    predictions.forEach(pred => {
      const [x1, y1, x2, y2] = pred.bbox;
      const cx = (x1 + x2) / 2;
      const cy = (y1 + y2) / 2;
      const isHomography = pred.homography_source === true;
      const iff = isHomography ? IFF.PROJECTED : IFF.HOSTILE;
      const methodTag = isHomography ? 'H-PROJ' : 'EXTRAP';
      const srcCam = pred.source_camera ?? -1;

      // ── BLOS: if center is off-screen, draw edge-clamped chevron ───
      const margin = 30;
      const isOffScreen = cx < -margin || cx > W + margin || cy < -margin || cy > H + margin;

      if (isOffScreen) {
        // Clamp to viewport edge
        const angle = Math.atan2(cy - H / 2, cx - W / 2);
        let edgeX = Math.max(margin, Math.min(W - margin, cx));
        let edgeY = Math.max(margin, Math.min(H - margin, cy));
        if (cx < -margin)        edgeX = margin;
        else if (cx > W + margin) edgeX = W - margin;
        if (cy < -margin)        edgeY = margin;
        else if (cy > H + margin) edgeY = H - margin;

        // Draw BLOS chevron
        drawEdgeChevron(ctx, edgeX, edgeY, angle, 14, iff);

        // Small label next to chevron
        ctx.save();
        ctx.font = 'bold 9px "Consolas", monospace';
        ctx.fillStyle = iff.stroke;
        ctx.shadowColor = iff.glow;
        ctx.shadowBlur = 6;
        const blosLabel = `${methodTag} \u00B7 Cam ${srcCam}`;
        ctx.fillText(blosLabel, edgeX + 18, edgeY + 4);
        ctx.restore();
        return; // skip full overlay for off-screen predictions
      }

      // ── On-screen prediction ─────────────────────────────────────
      const hasKeypoints = pred.keypoints && pred.keypoints.length >= 17;

      // Ghost skeleton
      if (hasKeypoints) {
        drawSkeleton(ctx, pred.keypoints, iff.stroke, 2, true);
      }

      // Chevron marker (open triangle pointing up)
      ctx.save();
      ctx.translate(cx, cy);
      ctx.strokeStyle = iff.stroke;
      ctx.fillStyle = iff.bg;
      ctx.lineWidth = 2;
      ctx.shadowColor = iff.glow;
      ctx.shadowBlur = 10;
      const chevSize = 12;
      ctx.beginPath();
      ctx.moveTo(0, -chevSize);
      ctx.lineTo(chevSize * 0.7, chevSize * 0.4);
      ctx.lineTo(-chevSize * 0.7, chevSize * 0.4);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.restore();

      // Pulsing ring
      const pulse = Math.sin(now / 300) * 0.3 + 0.7;
      ctx.save();
      ctx.strokeStyle = iff.stroke;
      ctx.globalAlpha = pulse * 0.5;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.arc(cx, cy, 22, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      ctx.restore();

      // Corner brackets (dashed for extrapolation, solid for homography)
      if (!isHomography) {
        ctx.save();
        ctx.setLineDash([4, 4]);
        drawCornerBrackets(ctx, x1, y1, x2, y2, iff);
        ctx.setLineDash([]);
        ctx.restore();
      } else {
        drawCornerBrackets(ctx, x1, y1, x2, y2, iff);
      }

      // Tactical pill
      const conf = (pred.confidence * 100).toFixed(0);
      const timeAgo = pred.time_since_seen?.toFixed(1) || '?';
      const mainLabel = `${methodTag} ${conf}%`;
      const subLabel = srcCam >= 0 ? `Cam ${srcCam} \u00B7 ${timeAgo}s ago` : `${timeAgo}s ago`;
      drawTacticalPill(ctx, cx, y1, mainLabel, subLabel, iff);
    });

    // ── HUD frame decoration (thin rule lines at edges) ──────────────
    ctx.save();
    ctx.strokeStyle = IFF.HUD.stroke;
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.25;
    ctx.beginPath();
    ctx.moveTo(0, 1);     ctx.lineTo(W, 1);
    ctx.moveTo(0, H - 1); ctx.lineTo(W, H - 1);
    ctx.stroke();
    // Tiny corner accents
    const hc = 20;
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.4;
    ctx.beginPath(); ctx.moveTo(2, hc); ctx.lineTo(2, 2); ctx.lineTo(hc, 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(W - hc, 2); ctx.lineTo(W - 2, 2); ctx.lineTo(W - 2, hc); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(2, H - hc); ctx.lineTo(2, H - 2); ctx.lineTo(hc, H - 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(W - hc, H - 2); ctx.lineTo(W - 2, H - 2); ctx.lineTo(W - 2, H - hc); ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.restore();

    ctx.setLineDash([]);

  }, [detections, tracks, predictions, sensorData, drawSkeleton, drawDiamond,
      drawCornerBrackets, drawTacticalPill, drawEdgeChevron, drawCompassRibbon,
      drawThreatRing, estimateDistance]);

  // ── Frame data handler ──────────────────────────────────────────────
  useEffect(() => {
    if (!frameData || !imageRef.current) return;
    try {
      const blob = new Blob([frameData], { type: 'image/jpeg' });
      const imageUrl = URL.createObjectURL(blob);
      const img = imageRef.current;

      img.onload = () => {
        if (canvasRef.current) {
          canvasRef.current.width  = img.naturalWidth;
          canvasRef.current.height = img.naturalHeight;
          setDimensions({ width: img.naturalWidth, height: img.naturalHeight });
        }
        drawOverlays();
        URL.revokeObjectURL(imageUrl);
      };
      img.src = imageUrl;
    } catch (err) {
      console.error('Frame load error:', err);
    }
  }, [frameData, drawOverlays]);

  // Redraw when overlay data changes
  useEffect(() => { drawOverlays(); }, [drawOverlays]);

  // ── Render ──────────────────────────────────────────────────────────
  return (
    <div className={`camera-display ${isActive ? 'active' : 'inactive'}`}>
      {/* Header bar */}
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

      {/* Viewport */}
      <div className="camera-viewport" style={{ position: 'relative' }}>
        <img
          ref={imageRef}
          alt={`Camera ${cameraId}`}
          style={{ width: '100%', height: 'auto', display: frameData ? 'block' : 'none' }}
        />
        <canvas
          ref={canvasRef}
          style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: 'auto', pointerEvents: 'none' }}
          width={dimensions.width}
          height={dimensions.height}
        />
        {/* Scan-line overlay (CSS) */}
        {frameData && <div className="scanline-overlay" />}

        {!frameData && (
          <div className="no-signal">
            <div className="no-signal-content">
              <div className="no-signal-icon">{'\u25C6'}</div>
              <p className="no-signal-cam">CAM-{cameraId}</p>
              <p className="no-signal-msg">{isActive ? 'AWAITING SIGNAL' : 'SENSOR OFFLINE'}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default CameraDisplay;
