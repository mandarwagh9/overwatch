/**
 * Camera stream adapter for mobile camera streaming.
 * Handles video capture, encoding, and WebSocket transmission.
 */

import { getCameraWebSocketUrl, getConfig } from '../config';

/**
 * Camera stream states
 */
export const StreamState = {
  IDLE: 'idle',
  INITIALIZING: 'initializing',
  CAMERA_READY: 'camera_ready',
  CONNECTING: 'connecting',
  STREAMING: 'streaming',
  DISCONNECTED: 'disconnected',
  ERROR: 'error',
  STOPPED: 'stopped'
};

/**
 * Camera stream events
 */
export const StreamEvents = {
  STATE_CHANGE: 'stateChange',
  ERROR: 'error',
  STATS_UPDATE: 'statsUpdate',
  CAMERA_REGISTERED: 'cameraRegistered'
};

/**
 * Camera stream adapter
 */
class CameraStreamAdapter {
  constructor() {
    this.socket = null;
    this.stream = null;
    this.videoElement = null;
    this.canvas = null;
    this.ctx = null;
    this.isStreaming = false;
    this.cameraId = null;
    this.facingMode = 'environment';
    
    // Configuration
    this.targetFps = getConfig('mobile.targetFps', 15);
    this.jpegQuality = getConfig('mobile.jpegQuality', 0.5);
    this.maxWidth = getConfig('mobile.maxWidth', 640);
    this.sensorInterval = getConfig('mobile.sensorInterval', 500);
    
    // State
    this.currentState = StreamState.IDLE;
    this.listeners = new Map();
    
    // Stats
    this.stats = {
      framesSent: 0,
      bytesSent: 0,
      startTime: null,
      fps: 0,
      lastFpsUpdate: 0,
      fpsCounter: 0
    };
    
    // Sensor data
    this._geoWatchId = null;
    this._lastGps = null;
    this._lastOrientation = null;
    this._sensorIntervalId = null;
    this._orientationHandler = null;
    this._captureIntervalId = null;
    this._keepAliveIntervalId = null;
  }

  /**
   * Add event listener
   * @param {string} event
   * @param {Function} callback
   */
  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
  }

  /**
   * Remove event listener
   * @param {string} event
   * @param {Function} callback
   */
  off(event, callback) {
    if (this.listeners.has(event)) {
      const callbacks = this.listeners.get(event);
      const index = callbacks.indexOf(callback);
      if (index !== -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  /**
   * Emit event
   * @param {string} event
   * @param {any} data
   */
  emit(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`[CameraStream] Error in ${event} listener:`, error);
        }
      });
    }
  }

  /**
   * Set state
   * @param {string} state
   */
  setState(state) {
    if (this.currentState !== state) {
      this.currentState = state;
      this.emit(StreamEvents.STATE_CHANGE, state);
    }
  }

  /**
   * Start streaming
   * @param {HTMLVideoElement} videoElement
   * @param {Object} [options]
   * @returns {Promise<number>} Camera ID
   */
  async start(videoElement, options = {}) {
    if (this.isStreaming) {
      throw new Error('Already streaming');
    }

    // Apply options
    if (options.facingMode) this.facingMode = options.facingMode;
    if (options.targetFps) this.targetFps = options.targetFps;
    if (options.jpegQuality) this.jpegQuality = options.jpegQuality;
    if (options.maxWidth) this.maxWidth = options.maxWidth;

    this.videoElement = videoElement;
    this.setState(StreamState.INITIALIZING);

    try {
      // Get camera access
      await this._initCamera();
      this.setState(StreamState.CAMERA_READY);

      // Connect WebSocket
      await this._connectWebSocket();

      // Start streaming
      this.isStreaming = true;
      this.stats.startTime = Date.now();
      this.stats.lastFpsUpdate = Date.now();
      this._startCaptureLoop();
      this._startSensorCapture();
      this._startKeepAlive();
      this.setState(StreamState.STREAMING);

      return this.cameraId;

    } catch (error) {
      this.setState(StreamState.ERROR);
      this.emit(StreamEvents.ERROR, error.message);
      throw error;
    }
  }

  /**
   * Initialize camera
   * @private
   */
  async _initCamera() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: this.facingMode,
          width: { ideal: this.maxWidth },
          height: { ideal: Math.round(this.maxWidth * 3 / 4) }
        },
        audio: false
      });

      this.videoElement.srcObject = this.stream;
      await this.videoElement.play();

      // Initialize canvas for frame capture
      this.canvas = document.createElement('canvas');
      this.ctx = this.canvas.getContext('2d');

    } catch (error) {
      throw new Error(`Camera access denied: ${error.message}`);
    }
  }

  /**
   * Connect to WebSocket
   * @private
   */
  _connectWebSocket() {
    return new Promise((resolve, reject) => {
      const url = getCameraWebSocketUrl();
      console.log(`[CameraStream] Connecting to ${url}`);

      this.socket = new WebSocket(url);
      this.socket.binaryType = 'arraybuffer';

      const timeout = setTimeout(() => {
        reject(new Error('WebSocket connection timeout'));
        this.socket.close();
      }, 10000);

      this.socket.onopen = () => {
        console.log('[CameraStream] WebSocket connected');
        this.setState(StreamState.CONNECTING);

        // Send registration
        this.socket.send(JSON.stringify({
          type: 'register',
          role: 'camera_source',
          camera_id: null
        }));
      };

      this.socket.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);

          if (msg.type === 'registered') {
            clearTimeout(timeout);
            this.cameraId = msg.camera_id;
            
            // Apply server-suggested settings
            if (msg.target_fps) this.targetFps = msg.target_fps;
            if (msg.max_width) this.maxWidth = msg.max_width;
            
            console.log(`[CameraStream] Registered as camera ${this.cameraId}`);
            this.emit(StreamEvents.CAMERA_REGISTERED, this.cameraId);
            resolve(this.cameraId);

          } else if (msg.type === 'error') {
            clearTimeout(timeout);
            reject(new Error(msg.message || 'Registration failed'));
          } else if (msg.type === 'pong') {
            // Server responded to ping, connection is alive
            console.log('[CameraStream] Received pong from server');
          }
        } catch (e) {
          // Ignore non-JSON messages after registration
        }
      };

      this.socket.onerror = (err) => {
        clearTimeout(timeout);
        console.error('[CameraStream] WebSocket error:', err);
        reject(new Error('WebSocket connection failed'));
      };

      this.socket.onclose = (event) => {
        console.log(`[CameraStream] WebSocket closed: code=${event.code}, reason=${event.reason}`);
        if (this.isStreaming) {
          this.setState(StreamState.DISCONNECTED);
          this.stop();
        }
      };
    });
  }

  /**
   * Start frame capture loop
   * @private
   */
  _startCaptureLoop() {
    const interval = 1000 / this.targetFps;

    this._captureIntervalId = setInterval(() => {
      if (!this.isStreaming || !this.socket || this.socket.readyState !== WebSocket.OPEN) {
        return;
      }

      this._captureAndSendFrame();
    }, interval);
  }

  /**
   * Capture and send frame
   * @private
   */
  _captureAndSendFrame() {
    if (!this.videoElement || this.videoElement.readyState < 2) return;

    const vw = this.videoElement.videoWidth;
    const vh = this.videoElement.videoHeight;
    if (vw === 0 || vh === 0) return;

    // Scale down if needed
    let cw = vw;
    let ch = vh;
    if (cw > this.maxWidth) {
      const scale = this.maxWidth / cw;
      cw = Math.round(cw * scale);
      ch = Math.round(ch * scale);
    }

    this.canvas.width = cw;
    this.canvas.height = ch;
    this.ctx.drawImage(this.videoElement, 0, 0, cw, ch);

    // Encode and send
    this.canvas.toBlob(
      (blob) => {
        if (!blob || !this.socket || this.socket.readyState !== WebSocket.OPEN) return;

        blob.arrayBuffer().then((buffer) => {
          this.socket.send(buffer);

          // Update stats
          this.stats.framesSent++;
          this.stats.bytesSent += buffer.byteLength;
          this.stats.fpsCounter++;

          const now = Date.now();
          if (now - this.stats.lastFpsUpdate >= 1000) {
            this.stats.fps = (this.stats.fpsCounter / (now - this.stats.lastFpsUpdate)) * 1000;
            this.stats.fpsCounter = 0;
            this.stats.lastFpsUpdate = now;
            this.emit(StreamEvents.STATS_UPDATE, this.getStats());
          }
        });
      },
      'image/jpeg',
      this.jpegQuality
    );
  }

  /**
   * Start keepalive ping
   * @private
   */
  _startKeepAlive() {
    // Send ping every 20 seconds to keep connection alive
    this._keepAliveIntervalId = setInterval(() => {
      if (this.socket && this.socket.readyState === WebSocket.OPEN) {
        try {
          this.socket.send(JSON.stringify({ type: 'ping' }));
        } catch (e) {
          console.error('[CameraStream] Failed to send ping:', e);
        }
      }
    }, 20000);
  }

  /**
   * Start sensor capture
   * @private
   */
  _startSensorCapture() {
    // GPS
    if ('geolocation' in navigator) {
      try {
        this._geoWatchId = navigator.geolocation.watchPosition(
          (pos) => {
            this._lastGps = {
              latitude: pos.coords.latitude,
              longitude: pos.coords.longitude,
              altitude: pos.coords.altitude,
              accuracy: pos.coords.accuracy
            };
          },
          (err) => console.warn('[CameraStream] GPS error:', err.message),
          { enableHighAccuracy: true, maximumAge: 1000, timeout: 5000 }
        );
      } catch (e) {
        console.warn('[CameraStream] Geolocation not available:', e);
      }
    }

    // Device orientation
    const handleOrientation = (event) => {
      this._lastOrientation = {
        alpha: event.alpha,
        beta: event.beta,
        gamma: event.gamma
      };
    };

    if (typeof DeviceOrientationEvent !== 'undefined') {
      if (typeof DeviceOrientationEvent.requestPermission === 'function') {
        DeviceOrientationEvent.requestPermission()
          .then(state => {
            if (state === 'granted') {
              window.addEventListener('deviceorientation', handleOrientation, true);
            }
          })
          .catch(console.warn);
      } else {
        window.addEventListener('deviceorientation', handleOrientation, true);
      }
    }
    this._orientationHandler = handleOrientation;

    // Send sensor data periodically
    this._sensorIntervalId = setInterval(() => {
      if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return;
      if (!this._lastGps && !this._lastOrientation) return;

      const msg = {
        type: 'sensor_data',
        timestamp: Date.now()
      };
      if (this._lastGps) msg.gps = this._lastGps;
      if (this._lastOrientation) msg.orientation = this._lastOrientation;

      try {
        this.socket.send(JSON.stringify(msg));
      } catch (e) {
        // Ignore send errors
      }
    }, this.sensorInterval);
  }

  /**
   * Stop sensor capture
   * @private
   */
  _stopSensorCapture() {
    if (this._geoWatchId !== null) {
      navigator.geolocation.clearWatch(this._geoWatchId);
      this._geoWatchId = null;
    }
    if (this._sensorIntervalId) {
      clearInterval(this._sensorIntervalId);
      this._sensorIntervalId = null;
    }
    if (this._orientationHandler) {
      window.removeEventListener('deviceorientation', this._orientationHandler, true);
      this._orientationHandler = null;
    }
    this._lastGps = null;
    this._lastOrientation = null;
  }

  /**
   * Stop streaming
   */
  stop() {
    this.isStreaming = false;

    // Stop keepalive
    if (this._keepAliveIntervalId) {
      clearInterval(this._keepAliveIntervalId);
      this._keepAliveIntervalId = null;
    }

    // Stop sensor capture
    this._stopSensorCapture();

    // Stop capture loop
    if (this._captureIntervalId) {
      clearInterval(this._captureIntervalId);
      this._captureIntervalId = null;
    }

    // Close WebSocket
    if (this.socket) {
      try {
        if (this.socket.readyState === WebSocket.OPEN) {
          this.socket.send(JSON.stringify({ type: 'stop' }));
        }
        this.socket.close();
      } catch (e) {
        // Ignore
      }
      this.socket = null;
    }

    // Stop camera tracks
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }

    // Clear video element
    if (this.videoElement) {
      this.videoElement.srcObject = null;
    }

    this.cameraId = null;
    this.setState(StreamState.STOPPED);
  }

  /**
   * Switch camera
   */
  async switchCamera() {
    this.facingMode = this.facingMode === 'environment' ? 'user' : 'environment';

    if (this.isStreaming && this.stream) {
      this.stream.getTracks().forEach(track => track.stop());

      try {
        this.stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: this.facingMode,
            width: { ideal: this.maxWidth },
            height: { ideal: Math.round(this.maxWidth * 3 / 4) }
          },
          audio: false
        });

        if (this.videoElement) {
          this.videoElement.srcObject = this.stream;
          await this.videoElement.play();
        }
      } catch (err) {
        this.emit(StreamEvents.ERROR, `Failed to switch camera: ${err.message}`);
      }
    }
  }

  /**
   * Get stats
   * @returns {Object}
   */
  getStats() {
    const elapsed = this.stats.startTime ? (Date.now() - this.stats.startTime) / 1000 : 0;
    return {
      cameraId: this.cameraId,
      isStreaming: this.isStreaming,
      framesSent: this.stats.framesSent,
      bytesSent: this.stats.bytesSent,
      mbSent: (this.stats.bytesSent / 1024 / 1024).toFixed(1),
      fps: Math.round(this.stats.fps * 10) / 10,
      elapsed: Math.round(elapsed),
      facingMode: this.facingMode
    };
  }

  /**
   * Get current state
   * @returns {string}
   */
  getState() {
    return this.currentState;
  }
}

// Singleton instance
const cameraStreamAdapter = new CameraStreamAdapter();

export { CameraStreamAdapter };
export default cameraStreamAdapter;
