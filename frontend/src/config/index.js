/**
 * Configuration management.
 * Centralized configuration with validation and defaults.
 */

/**
 * Configuration schema with validation
 */
const CONFIG_SCHEMA = {
  backend: {
    host: {
      env: 'REACT_APP_BACKEND_HOST',
      default: () => window.location.hostname || 'localhost',
      validate: (v) => typeof v === 'string' && v.length > 0
    },
    port: {
      env: 'REACT_APP_BACKEND_PORT',
      default: '8000',
      validate: (v) => !isNaN(parseInt(v, 10)) && parseInt(v, 10) > 0 && parseInt(v, 10) < 65536,
      transform: (v) => parseInt(v, 10)
    },
    protocol: {
      env: 'REACT_APP_BACKEND_PROTOCOL',
      default: () => window.location.protocol === 'https:' ? 'wss' : 'ws',
      validate: (v) => ['ws', 'wss', 'http', 'https'].includes(v)
    },
    apiProtocol: {
      env: 'REACT_APP_API_PROTOCOL',
      default: () => window.location.protocol === 'https:' ? 'https' : 'http',
      validate: (v) => ['http', 'https'].includes(v)
    }
  },
  websocket: {
    maxReconnectAttempts: {
      env: 'REACT_APP_WS_MAX_RECONNECT',
      default: 5,
      validate: (v) => parseInt(v, 10) > 0,
      transform: (v) => parseInt(v, 10)
    },
    reconnectDelay: {
      env: 'REACT_APP_WS_RECONNECT_DELAY',
      default: 1000,
      validate: (v) => parseInt(v, 10) >= 100,
      transform: (v) => parseInt(v, 10)
    },
    pingInterval: {
      env: 'REACT_APP_WS_PING_INTERVAL',
      default: 20000,
      validate: (v) => parseInt(v, 10) >= 5000,
      transform: (v) => parseInt(v, 10)
    }
  },
  camera: {
    maxCameras: {
      env: 'REACT_APP_MAX_CAMERAS',
      default: 4,
      validate: (v) => parseInt(v, 10) >= 1 && parseInt(v, 10) <= 16,
      transform: (v) => parseInt(v, 10)
    },
    inactivityTimeout: {
      env: 'REACT_APP_CAMERA_INACTIVITY_TIMEOUT',
      default: 3000,
      validate: (v) => parseInt(v, 10) >= 1000,
      transform: (v) => parseInt(v, 10)
    },
    fpsUpdateInterval: {
      env: 'REACT_APP_FPS_UPDATE_INTERVAL',
      default: 1000,
      validate: (v) => parseInt(v, 10) >= 100,
      transform: (v) => parseInt(v, 10)
    }
  },
  mobile: {
    targetFps: {
      env: 'REACT_APP_MOBILE_TARGET_FPS',
      default: 15,
      validate: (v) => parseInt(v, 10) >= 1 && parseInt(v, 10) <= 60,
      transform: (v) => parseInt(v, 10)
    },
    jpegQuality: {
      env: 'REACT_APP_MOBILE_JPEG_QUALITY',
      default: 0.5,
      validate: (v) => parseFloat(v) > 0 && parseFloat(v) <= 1,
      transform: (v) => parseFloat(v)
    },
    maxWidth: {
      env: 'REACT_APP_MOBILE_MAX_WIDTH',
      default: 640,
      validate: (v) => parseInt(v, 10) >= 160 && parseInt(v, 10) <= 1920,
      transform: (v) => parseInt(v, 10)
    },
    sensorInterval: {
      env: 'REACT_APP_MOBILE_SENSOR_INTERVAL',
      default: 500,
      validate: (v) => parseInt(v, 10) >= 100,
      transform: (v) => parseInt(v, 10)
    }
  },
  display: {
    personHeightMeters: {
      env: 'REACT_APP_PERSON_HEIGHT_METERS',
      default: 1.7,
      validate: (v) => parseFloat(v) > 0.5 && parseFloat(v) < 3.0,
      transform: (v) => parseFloat(v)
    },
    referencePixelsAt1M: {
      env: 'REACT_APP_REF_PX_AT_1M',
      default: 520,
      validate: (v) => parseInt(v, 10) > 0,
      transform: (v) => parseInt(v, 10)
    }
  }
};

/**
 * Get environment variable value
 * @param {string} name
 * @returns {string|undefined}
 */
function getEnvVar(name) {
  try {
    return process.env[name];
  } catch (e) {
    return undefined;
  }
}

/**
 * Parse configuration value
 * @param {Object} schema
 * @returns {any}
 */
function parseConfigValue(schema) {
  const envValue = getEnvVar(schema.env);
  
  if (envValue !== undefined) {
    const transformed = schema.transform ? schema.transform(envValue) : envValue;
    if (schema.validate(transformed)) {
      return transformed;
    }
    console.warn(`Invalid value for ${schema.env}: ${envValue}, using default`);
  }
  
  const defaultValue = typeof schema.default === 'function' ? schema.default() : schema.default;
  return schema.transform && typeof defaultValue === 'string' 
    ? schema.transform(defaultValue) 
    : defaultValue;
}

/**
 * Build configuration object
 * @returns {Object}
 */
function buildConfig() {
  const config = {};
  
  for (const [category, schemas] of Object.entries(CONFIG_SCHEMA)) {
    config[category] = {};
    for (const [key, schema] of Object.entries(schemas)) {
      config[category][key] = parseConfigValue(schema);
    }
  }
  
  return config;
}

// Build and cache configuration
const config = buildConfig();

/**
 * Get configuration value by path
 * @param {string} path - Dot-notation path (e.g., 'backend.host')
 * @param {any} defaultValue - Fallback value
 * @returns {any}
 */
export function getConfig(path, defaultValue) {
  const parts = path.split('.');
  let value = config;
  
  for (const part of parts) {
    if (value === undefined || value === null) {
      return defaultValue;
    }
    value = value[part];
  }
  
  return value !== undefined ? value : defaultValue;
}

/**
 * Get WebSocket URL
 * @returns {string}
 */
export function getWebSocketUrl() {
  const protocol = getConfig('backend.protocol', 'wss');
  const host = getConfig('backend.host', window.location.hostname);
  const port = getConfig('backend.port', 8000);
  return `${protocol}://${host}:${port}/ws`;
}

/**
 * Get camera WebSocket URL
 * @returns {string}
 */
export function getCameraWebSocketUrl() {
  const protocol = getConfig('backend.protocol', 'wss');
  const host = getConfig('backend.host', window.location.hostname);
  const port = getConfig('backend.port', 8000);
  return `${protocol}://${host}:${port}/ws/camera`;
}

/**
 * Get API base URL
 * @returns {string}
 */
export function getApiBaseUrl() {
  const protocol = getConfig('backend.apiProtocol', 'https');
  const host = getConfig('backend.host', window.location.hostname);
  const port = getConfig('backend.port', 8000);
  return `${protocol}://${host}:${port}`;
}

/**
 * Check if configuration is valid
 * @returns {{valid: boolean, errors: string[]}}
 */
export function validateConfig() {
  const errors = [];
  
  const host = getConfig('backend.host');
  if (!host) {
    errors.push('Backend host is not configured');
  }
  
  const port = getConfig('backend.port');
  if (!port || port <= 0 || port > 65535) {
    errors.push('Backend port must be between 1 and 65535');
  }
  
  const maxCameras = getConfig('camera.maxCameras');
  if (maxCameras < 1 || maxCameras > 16) {
    errors.push('Max cameras must be between 1 and 16');
  }
  
  return {
    valid: errors.length === 0,
    errors
  };
}

/**
 * Get full configuration object (for debugging)
 * @returns {Object}
 */
export function getFullConfig() {
  return { ...config };
}

export default config;
