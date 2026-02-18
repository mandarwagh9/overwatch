/**
 * Domain entities and value objects.
 * Pure types with no external dependencies.
 */

/**
 * @typedef {Object} BoundingBox
 * @property {number} x1 - Left coordinate
 * @property {number} y1 - Top coordinate
 * @property {number} x2 - Right coordinate
 * @property {number} y2 - Bottom coordinate
 */

/**
 * @typedef {Object} Detection
 * @property {string} detection_id
 * @property {number} camera_id
 * @property {BoundingBox} bbox
 * @property {number} confidence
 * @property {number} class_id
 * @property {string} class_name
 * @property {[number, number]} center
 * @property {Array<[number, number, number]>} [keypoints]
 */

/**
 * @typedef {Object} Track
 * @property {number} track_id
 * @property {number} camera_id
 * @property {BoundingBox} bbox
 * @property {number} confidence
 * @property {number} class_id
 * @property {string} class_name
 * @property {string} state
 * @property {number} age
 * @property {number} hits
 * @property {[number, number]} velocity
 * @property {Array<[number, number, number]>} [keypoints]
 */

/**
 * @typedef {Object} PredictedTarget
 * @property {number} object_id
 * @property {number} camera_id
 * @property {BoundingBox} predicted_bbox
 * @property {number} confidence
 * @property {number} time_since_seen
 * @property {[number, number]} velocity_projection
 * @property {number} source_camera
 * @property {string} method
 * @property {Array<[number, number, number]>} [keypoints]
 */

/**
 * @typedef {Object} WorldObject
 * @property {number} object_id
 * @property {{x: number, y: number, z: number}} position
 * @property {{vx: number, vy: number, vz: number}} velocity
 * @property {number} class_id
 * @property {string} class_name
 * @property {number} confidence
 * @property {number} last_seen_camera
 * @property {[number, number, number]} [uncertainty]
 */

/**
 * @typedef {Object} CameraFrame
 * @property {number} camera_id
 * @property {Uint8Array} frame_data
 * @property {Array<Detection>} detections
 * @property {Array<Track>} tracks
 * @property {Array<PredictedTarget>} predictions
 */

/**
 * @typedef {Object} PerceptionSnapshot
 * @property {string} timestamp
 * @property {number} generation
 * @property {Array<WorldObject>} world_objects
 * @property {Object<number, CameraFrame>} camera_frames
 * @property {Object} metrics
 */

/**
 * @typedef {Object} SystemStats
 * @property {number} cameras_active
 * @property {number} max_cameras
 * @property {number} target_fps
 * @property {number} connected_clients
 * @property {boolean} detection_engine_ready
 * @property {boolean} tracking_active
 * @property {string} detection_model
 */

/**
 * @typedef {Object} ConnectionStats
 * @property {number} messagesReceived
 * @property {number} bytesReceived
 * @property {number} connectionTime
 * @property {number} lastMessage
 * @property {boolean} isConnected
 * @property {number} reconnectAttempts
 */

/**
 * @typedef {Object} CameraStats
 * @property {number} cameraId
 * @property {boolean} isActive
 * @property {number} fps
 * @property {number} lastUpdate
 */

/**
 * @typedef {Object} StreamStats
 * @property {number} cameraId
 * @property {boolean} isStreaming
 * @property {number} framesSent
 * @property {number} bytesSent
 * @property {string} mbSent
 * @property {number} fps
 * @property {number} elapsed
 * @property {string} facingMode
 */

export const EntityTypes = {
  DETECTION: 'detection',
  TRACK: 'track',
  PREDICTION: 'prediction',
  WORLD_OBJECT: 'world_object'
};

/**
 * IFF (Identification Friend or Foe) color system
 */
export const IFF_COLORS = {
  UNKNOWN: {
    stroke: '#ffbf00',
    fill: 'rgba(255,191,0,0.95)',
    bg: 'rgba(255,191,0,0.30)',
    glow: 'rgba(255,191,0,0.7)',
    dim: 'rgba(0,0,0,0.55)'
  },
  FRIENDLY: {
    stroke: '#00a0ff',
    fill: 'rgba(0,160,255,0.95)',
    bg: 'rgba(0,160,255,0.25)',
    glow: 'rgba(0,160,255,0.65)',
    dim: 'rgba(0,0,0,0.55)'
  },
  PROJECTED: {
    stroke: '#00ff82',
    fill: 'rgba(0,255,130,0.95)',
    bg: 'rgba(0,255,130,0.25)',
    glow: 'rgba(0,255,130,0.7)',
    dim: 'rgba(0,0,0,0.55)'
  },
  HOSTILE: {
    stroke: '#ff5050',
    fill: 'rgba(255,80,80,0.95)',
    bg: 'rgba(255,80,80,0.25)',
    glow: 'rgba(255,80,80,0.7)',
    dim: 'rgba(0,0,0,0.55)'
  },
  WORLD: {
    stroke: '#ff9800',
    fill: 'rgba(255,152,0,0.95)',
    bg: 'rgba(255,152,0,0.25)',
    glow: 'rgba(255,152,0,0.7)',
    dim: 'rgba(0,0,0,0.55)'
  },
  HUD: {
    stroke: 'rgba(0,200,220,0.6)',
    fill: 'rgba(0,200,220,0.15)',
    text: '#00c8dc'
  }
};

/**
 * COCO skeleton connections for pose visualization
 */
export const COCO_SKELETON = [
  [0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],
  [5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],
  [0,5],[0,6],[3,5],[4,6]
];

/**
 * COCO keypoint names
 */
export const COCO_KEYPOINTS = [
  'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
  'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
];

/**
 * Keypoint confidence threshold
 */
export const KEYPOINT_CONFIDENCE_THRESHOLD = 0.25;

/**
 * Calculate center point of a bounding box
 * @param {BoundingBox} bbox
 * @returns {[number, number]}
 */
export function getBoundingBoxCenter(bbox) {
  return [
    (bbox.x1 + bbox.x2) / 2,
    (bbox.y1 + bbox.y2) / 2
  ];
}

/**
 * Calculate bounding box dimensions
 * @param {BoundingBox} bbox
 * @returns {{width: number, height: number}}
 */
export function getBoundingBoxDimensions(bbox) {
  return {
    width: bbox.x2 - bbox.x1,
    height: bbox.y2 - bbox.y1
  };
}

/**
 * Calculate IoU between two bounding boxes
 * @param {BoundingBox} box1
 * @param {BoundingBox} box2
 * @returns {number}
 */
export function calculateIoU(box1, box2) {
  const x1 = Math.max(box1.x1, box2.x1);
  const y1 = Math.max(box1.y1, box2.y1);
  const x2 = Math.min(box1.x2, box2.x2);
  const y2 = Math.min(box1.y2, box2.y2);
  
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
  const area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
  const union = area1 + area2 - intersection;
  
  return union > 0 ? intersection / union : 0;
}
