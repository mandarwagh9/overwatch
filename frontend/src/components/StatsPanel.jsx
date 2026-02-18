/**
 * Stats Panel component
 * Displays system statistics and world objects
 */

import React from 'react';
import './StatsPanel.css';

function StatsPanel({ 
  systemStats, 
  worldObjects, 
  pipelineStats, 
  connectionStats, 
  cameraData 
}) {
  const formatBytes = (bytes) => {
    if (!bytes) return '0 MB';
    return (bytes / 1024 / 1024).toFixed(1) + ' MB';
  };

  const formatUptime = (connectionTime) => {
    if (!connectionTime) return '0s';
    return Math.round((Date.now() - connectionTime) / 1000) + 's';
  };

  return (
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
        <h4>World Model</h4>
        <p>Fused Objects: {worldObjects.length}</p>
        <p>Pipeline: {pipelineStats.avg_tick_ms ? `${pipelineStats.avg_tick_ms}ms/tick` : 'N/A'}</p>
        {worldObjects.map(obj => (
          <p 
            key={obj.object_id} 
            className="world-object-item"
          >
            ◆ [{obj.class_name}] T-{obj.object_id} {Math.round(obj.confidence * 100)}%
            · CAM-{obj.last_seen_camera}
          </p>
        ))}
      </div>

      <div className="stat-section">
        <h4>Connection</h4>
        <p>Messages: {connectionStats.messagesReceived || 0}</p>
        <p>Data: {formatBytes(connectionStats.bytesReceived)}</p>
        <p>Uptime: {formatUptime(connectionStats.connectionTime)}</p>
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
  );
}

export default StatsPanel;
