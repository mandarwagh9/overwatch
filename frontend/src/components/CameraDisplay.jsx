/**
 * CameraDisplay — EagleEye Tactical Perception Overlay v2
 * 
 * BOLD, high-contrast military HUD that works on ANY background:
 *  ◆ Large diamond/chevron markers with dark contrast backdrops
 *  ◆ IFF color coding: Amber=detect, Blue=track, Green=H-PROJ, Red=EXTRAP
 *  ◆ Distance estimation from bbox height (~1.7m person)
 *  ◆ BLOS edge-clamped indicators for off-screen predictions
 *  ◆ Compass ribbon + threat ring
 *  ◆ COCO skeleton overlay
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';

// ── IFF Color System — BOLD opacities ───────────────────────────────────
const IFF = {
  UNKNOWN:   { stroke: '#ffbf00', fill: 'rgba(255,191,0,0.95)',  bg: 'rgba(255,191,0,0.30)', glow: 'rgba(255,191,0,0.7)',  dim: 'rgba(0,0,0,0.55)' },
  FRIENDLY:  { stroke: '#00a0ff', fill: 'rgba(0,160,255,0.95)',  bg: 'rgba(0,160,255,0.25)', glow: 'rgba(0,160,255,0.65)', dim: 'rgba(0,0,0,0.55)' },
  PROJECTED: { stroke: '#00ff82', fill: 'rgba(0,255,130,0.95)',  bg: 'rgba(0,255,130,0.25)', glow: 'rgba(0,255,130,0.7)',  dim: 'rgba(0,0,0,0.55)' },
  HOSTILE:   { stroke: '#ff5050', fill: 'rgba(255,80,80,0.95)',   bg: 'rgba(255,80,80,0.25)', glow: 'rgba(255,80,80,0.7)', dim: 'rgba(0,0,0,0.55)' },
  WORLD:     { stroke: '#ff9800', fill: 'rgba(255,152,0,0.95)',    bg: 'rgba(255,152,0,0.25)',  glow: 'rgba(255,152,0,0.7)', dim: 'rgba(0,0,0,0.55)' },
  HUD:       { stroke: 'rgba(0,200,220,0.6)', fill: 'rgba(0,200,220,0.15)', text: '#00c8dc' },
};

// ── COCO Skeleton ───────────────────────────────────────────────────────
const COCO_SKELETON = [
  [0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],
  [5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[0,5],[0,6],[3,5],[4,6],
];
const KP_CONF = 0.25;

const PERSON_HEIGHT_M = 1.7;
const REF_PX_AT_1M = 520;

// ─────────────────────────────────────────────────────────────────────────
const CameraDisplay = ({
  cameraId, frameData, detections = [], tracks = [], predictions = [],
  isActive = false, sensorData = null,
}) => {
  const canvasRef = useRef(null);
  const imageRef  = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 640, height: 480 });
  const [stats, setStats] = useState({ detectionsCount: 0, tracksCount: 0, predictionsCount: 0 });

  useEffect(() => {
    setStats({ detectionsCount: detections.length, tracksCount: tracks.length, predictionsCount: predictions.length });
  }, [detections, tracks, predictions]);

  // ── Distance from bbox height ────────────────────────────────────────
  const estimateDistance = useCallback((bboxH) => {
    if (bboxH <= 0) return null;
    return Math.round((PERSON_HEIGHT_M * REF_PX_AT_1M) / bboxH * 10) / 10;
  }, []);

  // ── DIAMOND — large, bold, with dark backdrop ────────────────────────
  const drawDiamond = useCallback((ctx, cx, cy, size, iff, filled) => {
    ctx.save();

    // Dark backdrop circle for contrast on ANY background
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

  // ── CORNER BRACKETS — thick, long, glowing ───────────────────────────
  const drawCornerBrackets = useCallback((ctx, x1, y1, x2, y2, iff, overrideLen) => {
    ctx.save();
    ctx.strokeStyle = iff.stroke;
    ctx.lineWidth = 3;
    ctx.lineCap = 'square';
    ctx.shadowColor = iff.glow;
    ctx.shadowBlur = 10;
    const len = overrideLen || Math.max(20, Math.min(30, (x2 - x1) * 0.2, (y2 - y1) * 0.2));

    // TL
    ctx.beginPath(); ctx.moveTo(x1, y1 + len); ctx.lineTo(x1, y1); ctx.lineTo(x1 + len, y1); ctx.stroke();
    // TR
    ctx.beginPath(); ctx.moveTo(x2 - len, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + len); ctx.stroke();
    // BL
    ctx.beginPath(); ctx.moveTo(x1, y2 - len); ctx.lineTo(x1, y2); ctx.lineTo(x1 + len, y2); ctx.stroke();
    // BR
    ctx.beginPath(); ctx.moveTo(x2 - len, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - len); ctx.stroke();
    ctx.restore();
  }, []);

  // ── TACTICAL PILL — solid dark bg, readable on any image ─────────────
  const drawTacticalPill = useCallback((ctx, cx, topY, label, sublabel, iff) => {
    ctx.save();
    ctx.font = 'bold 12px "Consolas", "Courier New", monospace';
    const tw = ctx.measureText(label).width;
    const stw = sublabel ? (() => { ctx.font = '10px "Consolas", monospace'; return ctx.measureText(sublabel).width; })() : 0;
    ctx.font = 'bold 12px "Consolas", "Courier New", monospace';
    const pad = 8;
    const h = sublabel ? 32 : 20;
    const w = Math.max(tw, stw) + pad * 2;
    const px = cx - w / 2;
    const py = topY - h - 10;

    // Solid dark background for readability
    ctx.fillStyle = 'rgba(0,0,0,0.75)';
    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(px, py, w, h, 4);
    } else {
      const r = 4;
      ctx.moveTo(px + r, py);
      ctx.lineTo(px + w - r, py); ctx.quadraticCurveTo(px + w, py, px + w, py + r);
      ctx.lineTo(px + w, py + h - r); ctx.quadraticCurveTo(px + w, py + h, px + w - r, py + h);
      ctx.lineTo(px + r, py + h); ctx.quadraticCurveTo(px, py + h, px, py + h - r);
      ctx.lineTo(px, py + r); ctx.quadraticCurveTo(px, py, px + r, py);
      ctx.closePath();
    }
    ctx.fill();

    // Colored border
    ctx.strokeStyle = iff.stroke;
    ctx.lineWidth = 2;
    ctx.shadowColor = iff.glow;
    ctx.shadowBlur = 8;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Main label
    ctx.fillStyle = iff.stroke;
    ctx.font = 'bold 12px "Consolas", "Courier New", monospace';
    ctx.fillText(label, px + pad, py + 15);

    // Sub-label
    if (sublabel) {
      ctx.font = '10px "Consolas", "Courier New", monospace';
      ctx.fillStyle = 'rgba(255,255,255,0.8)';
      ctx.fillText(sublabel, px + pad, py + 27);
    }

    // Connecting line
    ctx.strokeStyle = iff.stroke;
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.6;
    ctx.beginPath(); ctx.moveTo(cx, py + h); ctx.lineTo(cx, topY); ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.restore();
  }, []);

  // ── BLOS CHEVRON — large directional indicator ───────────────────────
  const drawEdgeChevron = useCallback((ctx, x, y, angle, size, iff) => {
    ctx.save();

    // Dark backdrop circle
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.beginPath();
    ctx.arc(x, y, size + 4, 0, 2 * Math.PI);
    ctx.fill();

    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.strokeStyle = iff.stroke;
    ctx.fillStyle = iff.bg;
    ctx.lineWidth = 3;
    ctx.shadowColor = iff.glow;
    ctx.shadowBlur = 14;
    ctx.beginPath();
    ctx.moveTo(0, -size);
    ctx.lineTo(size * 0.8, 0);
    ctx.lineTo(0, size);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }, []);

  // ── SKELETON ─────────────────────────────────────────────────────────
  const drawSkeleton = useCallback((ctx, keypoints, color, lineWidth, ghost) => {
    if (!keypoints || keypoints.length < 17) return;
    ctx.save();
    if (ghost) { ctx.shadowColor = color; ctx.shadowBlur = 16; ctx.setLineDash([6, 4]); }
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
      ctx.arc(kp[0], kp[1], ghost ? 4 : 3, 0, 2 * Math.PI);
      ctx.fill();
    });
    ctx.restore();
  }, []);

  // ── COMPASS RIBBON ───────────────────────────────────────────────────
  const drawCompassRibbon = useCallback((ctx, W, heading) => {
    if (heading == null) return;
    const rH = 24;
    ctx.save();
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(0, 0, W, rH);
    ctx.strokeStyle = IFF.HUD.stroke;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, rH); ctx.lineTo(W, rH); ctx.stroke();
    const cards = [{d:0,l:'N'},{d:45,l:'NE'},{d:90,l:'E'},{d:135,l:'SE'},{d:180,l:'S'},{d:225,l:'SW'},{d:270,l:'W'},{d:315,l:'NW'}];
    ctx.font = 'bold 11px "Consolas", monospace';
    ctx.textAlign = 'center';
    const ppd = W / 120;
    cards.forEach(({ d, l }) => {
      let diff = d - heading;
      while (diff > 180) diff -= 360;
      while (diff < -180) diff += 360;
      if (Math.abs(diff) > 60) return;
      const px = W / 2 + diff * ppd;
      ctx.strokeStyle = l.length === 1 ? IFF.HUD.text : 'rgba(0,200,220,0.4)';
      ctx.lineWidth = l.length === 1 ? 2 : 1;
      ctx.beginPath(); ctx.moveTo(px, rH - 7); ctx.lineTo(px, rH); ctx.stroke();
      ctx.fillStyle = l.length === 1 ? IFF.HUD.text : 'rgba(0,200,220,0.5)';
      ctx.fillText(l, px, 14);
    });
    ctx.fillStyle = IFF.HUD.text;
    ctx.beginPath(); ctx.moveTo(W/2-6, rH); ctx.lineTo(W/2, rH-6); ctx.lineTo(W/2+6, rH); ctx.closePath(); ctx.fill();
    ctx.font = 'bold 12px "Consolas", monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`${Math.round(heading)}\u00B0`, W - 10, 15);
    ctx.textAlign = 'start';
    ctx.restore();
  }, []);

  // ── THREAT RING ──────────────────────────────────────────────────────
  const drawThreatRing = useCallback((ctx, W, H, preds) => {
    if (!preds || preds.length === 0) return;
    const cx = W / 2, cy = H / 2;
    const r = Math.min(W, H) / 2 - 6;
    ctx.save();
    preds.forEach(p => {
      const pcx = (p.bbox[0] + p.bbox[2]) / 2;
      const pcy = (p.bbox[1] + p.bbox[3]) / 2;
      const ang = Math.atan2(pcy - cy, pcx - cx);
      const pm = p.prediction_method || (p.homography_source ? 'HOMOGRAPHY' : 'EXTRAP');
      const iff = pm === 'HOMOGRAPHY' ? IFF.PROJECTED : pm === 'WORLD' ? IFF.WORLD : IFF.HOSTILE;
      ctx.strokeStyle = iff.stroke;
      ctx.lineWidth = 4;
      ctx.globalAlpha = 0.5;
      ctx.shadowColor = iff.glow;
      ctx.shadowBlur = 12;
      ctx.beginPath(); ctx.arc(cx, cy, r, ang - 0.2, ang + 0.2); ctx.stroke();
      ctx.globalAlpha = 1; ctx.shadowBlur = 0;
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
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const now = Date.now();

    // Compass + threat ring
    drawCompassRibbon(ctx, W, sensorData?.orientation?.alpha ?? null);
    drawThreatRing(ctx, W, H, predictions);

    // ── PHASE 1: DETECTIONS — Amber ──────────────────────────────────
    detections.forEach(det => {
      const [x1, y1, x2, y2] = det.bbox;
      const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
      const bH = y2 - y1, dist = estimateDistance(bH);
      const iff = IFF.UNKNOWN;

      // Dark tint inside bbox for contrast
      ctx.save();
      ctx.fillStyle = 'rgba(0,0,0,0.2)';
      ctx.fillRect(x1, y1, x2 - x1, bH);
      ctx.restore();

      // Skeleton
      if (det.keypoints && det.keypoints.length >= 17)
        drawSkeleton(ctx, det.keypoints, 'rgba(255,191,0,0.7)', 2, false);

      // Corner brackets
      drawCornerBrackets(ctx, x1, y1, x2, y2, iff);

      // Diamond at center
      drawDiamond(ctx, cx, cy, 18, iff, false);

      // Pill
      const conf = (det.confidence * 100).toFixed(0);
      drawTacticalPill(ctx, cx, y1, `PERSON \u00B7 ${conf}%`, dist ? `~${dist}m` : null, iff);
    });

    // ── PHASE 2: TRACKS — Blue ───────────────────────────────────────
    tracks.forEach(track => {
      const [x1, y1, x2, y2] = track.bbox;
      const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
      const bH = y2 - y1, dist = estimateDistance(bH);
      const iff = IFF.FRIENDLY;

      // Dark tint
      ctx.save();
      ctx.fillStyle = 'rgba(0,0,0,0.15)';
      ctx.fillRect(x1, y1, x2 - x1, bH);
      ctx.restore();

      // Skeleton
      if (track.keypoints && track.keypoints.length >= 17)
        drawSkeleton(ctx, track.keypoints, 'rgba(0,160,255,0.7)', 2, false);

      // Brackets
      drawCornerBrackets(ctx, x1, y1, x2, y2, iff);

      // Filled diamond
      drawDiamond(ctx, cx, cy, 20, iff, true);

      // Velocity arrow
      if (track.velocity) {
        const [vx, vy] = track.velocity;
        if (Math.sqrt(vx*vx + vy*vy) > 0.3) {
          const sc = 18, ex = cx + vx*sc, ey = cy + vy*sc;
          const ang = Math.atan2(vy, vx), al = 9;
          ctx.save();
          ctx.strokeStyle = iff.stroke;
          ctx.lineWidth = 2.5;
          ctx.shadowColor = iff.glow;
          ctx.shadowBlur = 6;
          ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ex, ey); ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(ex, ey);
          ctx.lineTo(ex - al*Math.cos(ang - Math.PI/5), ey - al*Math.sin(ang - Math.PI/5));
          ctx.moveTo(ex, ey);
          ctx.lineTo(ex - al*Math.cos(ang + Math.PI/5), ey - al*Math.sin(ang + Math.PI/5));
          ctx.stroke();
          ctx.restore();
        }
      }

      // Pill
      drawTacticalPill(ctx, cx, y1, `T-${track.track_id}`, dist ? `~${dist}m \u00B7 ${track.age||0} hits` : `${track.age||0} hits`, iff);
    });

    // ── PHASE 3: PREDICTIONS / GHOSTS ────────────────────────────────
    predictions.forEach(pred => {
      const [x1, y1, x2, y2] = pred.bbox;
      const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
      const bW = x2 - x1, bH = y2 - y1;
      const pm = pred.prediction_method || (pred.homography_source ? 'HOMOGRAPHY' : 'EXTRAP');
      const isH = pm === 'HOMOGRAPHY';
      const isW = pm === 'WORLD';
      const iff = isH ? IFF.PROJECTED : isW ? IFF.WORLD : IFF.HOSTILE;
      const tag = isH ? 'H-PROJ' : isW ? 'WORLD' : 'EXTRAP';
      const src = pred.source_camera ?? -1;

      // ── BLOS off-screen check ──────────────────────────────────
      const isOff = cx < 0 || cx > W || cy < 0 || cy > H;
      if (isOff) {
        const ang = Math.atan2(cy - H/2, cx - W/2);
        const m = 40;
        const ex = Math.max(m, Math.min(W - m, cx));
        const ey = Math.max(m, Math.min(H - m, cy));
        drawEdgeChevron(ctx, ex, ey, ang, 18, iff);
        // Label
        ctx.save();
        ctx.font = 'bold 11px "Consolas", monospace';
        ctx.fillStyle = 'rgba(0,0,0,0.7)';
        const bl = `${tag} \u00B7 Cam ${src}`;
        const btw = ctx.measureText(bl).width;
        ctx.fillRect(ex + 22, ey - 9, btw + 10, 20);
        ctx.fillStyle = iff.stroke;
        ctx.shadowColor = iff.glow;
        ctx.shadowBlur = 8;
        ctx.fillText(bl, ex + 27, ey + 5);
        ctx.restore();
        return;
      }

      // ── On-screen prediction — VERY VISIBLE ───────────────────
      const hasKP = pred.keypoints && pred.keypoints.length >= 17;

      // Large dark tinted bbox area
      ctx.save();
      ctx.fillStyle = 'rgba(0,0,0,0.3)';
      ctx.strokeStyle = iff.stroke;
      ctx.lineWidth = 2;
      ctx.shadowColor = iff.glow;
      ctx.shadowBlur = 12;
      if (!isH) ctx.setLineDash([6, 4]);  // dashed for EXTRAP and WORLD
      ctx.fillRect(x1, y1, bW, bH);
      ctx.strokeRect(x1, y1, bW, bH);
      ctx.setLineDash([]);
      ctx.shadowBlur = 0;
      ctx.restore();

      // Ghost skeleton
      if (hasKP) drawSkeleton(ctx, pred.keypoints, iff.stroke, 2.5, true);

      // Corner brackets
      ctx.save();
      if (!isH) ctx.setLineDash([5, 4]);  // dashed for EXTRAP and WORLD
      drawCornerBrackets(ctx, x1, y1, x2, y2, iff);
      ctx.setLineDash([]);
      ctx.restore();

      // Large chevron marker with dark backdrop
      ctx.save();
      // Dark circle behind
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.beginPath();
      ctx.arc(cx, cy, 22, 0, 2 * Math.PI);
      ctx.fill();
      // Chevron
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

      // Pulsing ring — thick, visible
      const pulse = Math.sin(now / 250) * 0.3 + 0.7;
      ctx.save();
      ctx.strokeStyle = iff.stroke;
      ctx.globalAlpha = pulse * 0.7;
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.arc(cx, cy, 30, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      ctx.restore();

      // Pill
      const conf = (pred.confidence * 100).toFixed(0);
      const tAgo = pred.time_since_seen?.toFixed(1) || '?';
      drawTacticalPill(ctx, cx, y1, `${tag} ${conf}%`, src >= 0 ? `Cam ${src} \u00B7 ${tAgo}s ago` : `${tAgo}s ago`, iff);
    });

    // ── HUD frame corners ────────────────────────────────────────────
    ctx.save();
    ctx.strokeStyle = IFF.HUD.stroke;
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.5;
    const hc = 24;
    ctx.beginPath(); ctx.moveTo(3, hc); ctx.lineTo(3, 3); ctx.lineTo(hc, 3); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(W-hc, 3); ctx.lineTo(W-3, 3); ctx.lineTo(W-3, hc); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(3, H-hc); ctx.lineTo(3, H-3); ctx.lineTo(hc, H-3); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(W-hc, H-3); ctx.lineTo(W-3, H-3); ctx.lineTo(W-3, H-hc); ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.restore();
    ctx.setLineDash([]);

  }, [detections, tracks, predictions, sensorData, drawSkeleton, drawDiamond,
      drawCornerBrackets, drawTacticalPill, drawEdgeChevron, drawCompassRibbon,
      drawThreatRing, estimateDistance]);

  // ── Frame handler ───────────────────────────────────────────────────
  useEffect(() => {
    if (!frameData || !imageRef.current) return;
    try {
      const blob = new Blob([frameData], { type: 'image/jpeg' });
      const url = URL.createObjectURL(blob);
      const img = imageRef.current;
      img.onload = () => {
        if (canvasRef.current) {
          canvasRef.current.width = img.naturalWidth;
          canvasRef.current.height = img.naturalHeight;
          setDimensions({ width: img.naturalWidth, height: img.naturalHeight });
        }
        drawOverlays();
        URL.revokeObjectURL(url);
      };
      img.src = url;
    } catch (e) { console.error('Frame load error:', e); }
  }, [frameData, drawOverlays]);

  useEffect(() => { drawOverlays(); }, [drawOverlays]);

  // ── Render ──────────────────────────────────────────────────────────
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
      <div className="camera-viewport" style={{ position: 'relative' }}>
        <img ref={imageRef} alt={`Camera ${cameraId}`}
          style={{ width: '100%', height: 'auto', display: frameData ? 'block' : 'none' }} />
        <canvas ref={canvasRef}
          style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: 'auto', pointerEvents: 'none' }}
          width={dimensions.width} height={dimensions.height} />
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
