/**
 * Hook for managing camera data and state.
 * Handles frame data, detections, tracks, and predictions.
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { getConfig } from '../../config';
import { WebSocketEvents } from '../../infrastructure/websocketAdapter';

/**
 * Initial camera data structure
 * @param {number} cameraId
 * @returns {Object}
 */
function createInitialCameraData(cameraId) {
  return {
    cameraId,
    frameData: null,
    detections: [],
    tracks: [],
    predictions: [],
    isActive: false,
    fps: 0,
    lastUpdate: null,
    lastPredictionUpdate: null
  };
}

/**
 * Use camera data hook
 * @param {Object} options
 * @param {import('react').MutableRefObject} webSocketRef - Ref to WebSocket adapter
 * @returns {{
 *   cameraData: Object<number, Object>,
 *   worldObjects: Array,
 *   pipelineStats: Object,
 *   handleFrame: (data: Object) => void,
 *   handlePredictions: (data: Object) => void,
 *   handleWorldUpdate: (data: Object) => void,
 *   setCameraActive: (cameraId: number, active: boolean) => void
 * }}
 */
export function useCameraData(webSocketRef) {
  const [cameraData, setCameraData] = useState({});
  const [worldObjects, setWorldObjects] = useState([]);
  const [pipelineStats, setPipelineStats] = useState({});

  const frameCounterRef = useRef({});
  const fpsTimerRef = useRef({});

  const inactivityTimeout = getConfig('camera.inactivityTimeout', 3000);

  // Initialize camera data for all possible cameras
  useEffect(() => {
    const maxCameras = getConfig('camera.maxCameras', 4);
    const initialData = {};
    for (let i = 0; i < maxCameras; i++) {
      initialData[i] = createInitialCameraData(i);
    }
    setCameraData(initialData);
  }, []);

  /**
   * Handle snapshot message with camera frames
   */
  const handleSnapshot = useCallback((data) => {
    console.log('[CameraData] Received snapshot:', {
      generation: data.generation,
      camera_count: Object.keys(data.camera_frames || {}).length,
      world_objects: data.world_objects?.length
    });
    
    // Update world objects
    if (data.world_objects) {
      setWorldObjects(data.world_objects);
    }
    
    // Process camera frames
    const cameraFrames = data.camera_frames || {};
    const detections = data.detections || {};
    const tracks = data.tracks || {};
    const predictions = data.predictions || {};
    
    Object.keys(cameraFrames).forEach(cameraIdStr => {
      const cameraId = parseInt(cameraIdStr, 10);
      const frameData = cameraFrames[cameraIdStr];
      
      // Update frame counter for FPS calculation
      if (!frameCounterRef.current[cameraId]) {
        frameCounterRef.current[cameraId] = 0;
        fpsTimerRef.current[cameraId] = Date.now();
      }
      
      frameCounterRef.current[cameraId]++;
      
      // Calculate FPS every second
      const now = Date.now();
      const elapsed = now - fpsTimerRef.current[cameraId];
      
      setCameraData(prev => {
        const camera = prev[cameraId] || createInitialCameraData(cameraId);
        let fps = camera.fps;

        if (elapsed >= 1000) {
          fps = Math.round((frameCounterRef.current[cameraId] / elapsed) * 10000) / 10;
          frameCounterRef.current[cameraId] = 0;
          fpsTimerRef.current[cameraId] = now;
        }

        return {
          ...prev,
          [cameraId]: {
            ...camera,
            frameData: frameData,
            detections: detections[cameraIdStr] || [],
            tracks: tracks[cameraIdStr] || [],
            predictions: predictions[cameraIdStr] || [],
            isActive: true,
            fps,
            lastUpdate: now
          }
        };
      });
    });
  }, []);

  /**
   * Handle frame message (legacy)
   */
  const handleFrame = useCallback((data) => {
    console.log('[CameraData] Received frame message:', {
      camera_id: data.camera_id,
      has_frame_data: !!data.frame_data,
      detections_count: data.detections?.length,
      tracks_count: data.tracks?.length
    });
    
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
    
    setCameraData(prev => {
      const camera = prev[cameraId] || createInitialCameraData(cameraId);
      let fps = camera.fps;

      if (elapsed >= 1000) {
        fps = Math.round((frameCounterRef.current[cameraId] / elapsed) * 10000) / 10;
        frameCounterRef.current[cameraId] = 0;
        fpsTimerRef.current[cameraId] = now;
      }

      return {
        ...prev,
        [cameraId]: {
          ...camera,
          frameData: data.frame_data,
          detections: data.detections || [],
          tracks: data.tracks || [],
          predictions: data.predictions || [],
          isActive: true,
          fps,
          lastUpdate: now
        }
      };
    });
  }, []);

  /**
   * Handle prediction message
   */
  const handlePredictions = useCallback((data) => {
    const cameraId = data.camera_id;
    
    setCameraData(prev => {
      const camera = prev[cameraId] || createInitialCameraData(cameraId);
      return {
        ...prev,
        [cameraId]: {
          ...camera,
          predictions: data.predictions || [],
          lastPredictionUpdate: Date.now()
        }
      };
    });
  }, []);

  /**
   * Handle world update message
   */
  const handleWorldUpdate = useCallback((data) => {
    if (data.objects) {
      setWorldObjects(data.objects);
    }
    if (data.pipeline_stats) {
      setPipelineStats(data.pipeline_stats);
    }
  }, []);

  /**
   * Set camera active state
   */
  const setCameraActive = useCallback((cameraId, active) => {
    setCameraData(prev => {
      const camera = prev[cameraId] || createInitialCameraData(cameraId);
      return {
        ...prev,
        [cameraId]: {
          ...camera,
          isActive: active
        }
      };
    });
  }, []);

  // Check camera activity periodically
  useEffect(() => {
    const interval = setInterval(() => {
      const now = Date.now();
      
      setCameraData(prev => {
        const updated = { ...prev };
        let hasChanges = false;
        
        Object.keys(updated).forEach(cameraId => {
          const camera = updated[cameraId];
          if (camera.isActive && camera.lastUpdate && 
              now - camera.lastUpdate > inactivityTimeout) {
            updated[cameraId] = {
              ...camera,
              isActive: false
            };
            hasChanges = true;
          }
        });
        
        return hasChanges ? updated : prev;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [inactivityTimeout]);

  return {
    cameraData,
    worldObjects,
    pipelineStats,
    handleSnapshot,
    handleFrame,
    handlePredictions,
    handleWorldUpdate,
    setCameraActive
  };
}

export default useCameraData;
