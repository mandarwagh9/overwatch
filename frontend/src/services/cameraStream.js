/**
 * Camera Stream Service
 * 
 * Captures video from the device camera using getUserMedia,
 * encodes frames as JPEG via an offscreen canvas, and sends
 * them as binary WebSocket messages to the Overwatch backend.
 */

export class CameraStreamService {
  constructor() {
    this.socket = null;
    this.stream = null;
    this.videoElement = null;
    this.canvas = null;
    this.ctx = null;
    this.isStreaming = false;
    this.captureIntervalId = null;
    this.cameraId = null;

    // Configurable settings (may be overridden by server response)
    this.targetFps = 15;
    this.jpegQuality = 0.5;
    this.maxWidth = 640;
    this.facingMode = 'environment'; // rear camera by default

    // Stats
    this.stats = {
      framesSent: 0,
      bytesSent: 0,
      startTime: null,
      fps: 0,
      lastFpsUpdate: 0,
      fpsCounter: 0
    };

    // Sensor fusion: GPS + IMU
    this._geoWatchId = null;
    this._lastGps = null;
    this._lastOrientation = null;
    this._sensorIntervalId = null;

    // Callbacks
    this.onStatusChange = null;
    this.onError = null;
    this.onStatsUpdate = null;
  }

  /**
   * Get the WebSocket URL for the camera endpoint.
   * Automatically determines ws:// vs wss:// based on page protocol.
   */
  _getWsUrl() {
    // Use env var for backend host, fallback to same-origin
    const host = process.env.REACT_APP_BACKEND_HOST || window.location.hostname || 'localhost';
    const port = process.env.REACT_APP_BACKEND_PORT || '8000';
    return `wss://${host}:${port}/ws/camera`;
  }

  /**
   * Start capturing and streaming from the device camera.
   * 
   * @param {HTMLVideoElement} videoElement - The <video> element for local preview
   * @param {Object} options - Optional overrides: { facingMode, targetFps, jpegQuality, maxWidth }
   * @returns {Promise<number>} The assigned camera ID
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
    this._emitStatus('initializing');

    // 1. Get camera access
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: this.facingMode,
          width: { ideal: this.maxWidth },
          height: { ideal: Math.round(this.maxWidth * 3 / 4) }
        },
        audio: false
      });
    } catch (err) {
      this._emitError(`Camera access denied: ${err.message}`);
      throw err;
    }

    // Attach stream to video element for local preview
    this.videoElement.srcObject = this.stream;
    await this.videoElement.play();
    this._emitStatus('camera_ready');

    // 2. Set up offscreen canvas for frame capture
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d');

    // 3. Connect WebSocket to /ws/camera
    try {
      await this._connectWebSocket();
    } catch (err) {
      this.stop();
      throw err;
    }

    // 4. Start the capture loop
    this.isStreaming = true;
    this.stats.startTime = Date.now();
    this.stats.lastFpsUpdate = Date.now();
    this._startCaptureLoop();
    this._startSensorCapture();
    this._emitStatus('streaming');

    return this.cameraId;
  }

  /**
   * Connect to the backend WebSocket and register as a camera source.
   */
  _connectWebSocket() {
    return new Promise((resolve, reject) => {
      const url = this._getWsUrl();
      console.log(`📱 Connecting to ${url}`);

      this.socket = new WebSocket(url);
      this.socket.binaryType = 'arraybuffer';

      const timeout = setTimeout(() => {
        reject(new Error('WebSocket connection timeout'));
        this.socket.close();
      }, 10000);

      this.socket.onopen = () => {
        clearTimeout(timeout);
        console.log('📱 WebSocket connected, sending registration...');

        // Send registration message
        this.socket.send(JSON.stringify({
          type: 'register',
          role: 'camera_source',
          camera_id: null  // auto-assign
        }));
      };

      this.socket.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);

          if (msg.type === 'registered') {
            this.cameraId = msg.camera_id;
            // Apply server-suggested settings
            if (msg.target_fps) this.targetFps = msg.target_fps;
            if (msg.max_width) this.maxWidth = msg.max_width;
            console.log(`📱 Registered as camera ${this.cameraId} (${this.targetFps} FPS)`);
            resolve(this.cameraId);
          } else if (msg.type === 'error') {
            reject(new Error(msg.message || 'Registration failed'));
          }
        } catch (e) {
          // Ignore non-JSON messages after registration
        }
      };

      this.socket.onerror = (err) => {
        clearTimeout(timeout);
        console.error('📱 WebSocket error:', err);
        this._emitError('WebSocket connection failed');
        reject(new Error('WebSocket connection failed'));
      };

      this.socket.onclose = () => {
        console.log('📱 WebSocket closed');
        if (this.isStreaming) {
          this._emitStatus('disconnected');
          this.stop();
        }
      };
    });
  }

  /**
   * Start the frame capture loop at the target FPS.
   */
  _startCaptureLoop() {
    const interval = 1000 / this.targetFps;

    this.captureIntervalId = setInterval(() => {
      if (!this.isStreaming || !this.socket || this.socket.readyState !== WebSocket.OPEN) {
        return;
      }

      this._captureAndSendFrame();
    }, interval);
  }

  /**
   * Capture one frame from the video, encode as JPEG, and send over WebSocket.
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

    // Encode as JPEG blob and send as binary
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

            if (this.onStatsUpdate) {
              this.onStatsUpdate(this.getStats());
            }
          }
        });
      },
      'image/jpeg',
      this.jpegQuality
    );
  }

  /**
   * Start capturing GPS + device orientation and sending as sensor_data messages.
   */
  _startSensorCapture() {
    // GPS via Geolocation API
    if ('geolocation' in navigator) {
      try {
        this._geoWatchId = navigator.geolocation.watchPosition(
          (pos) => {
            this._lastGps = {
              latitude: pos.coords.latitude,
              longitude: pos.coords.longitude,
              altitude: pos.coords.altitude,
              accuracy: pos.coords.accuracy,
            };
          },
          (err) => console.warn('GPS error:', err.message),
          { enableHighAccuracy: true, maximumAge: 1000, timeout: 5000 }
        );
      } catch (e) {
        console.warn('Geolocation not available:', e);
      }
    }

    // IMU via DeviceOrientation API
    const handleOrientation = (event) => {
      this._lastOrientation = {
        alpha: event.alpha,  // compass heading (0-360)
        beta: event.beta,    // front-back tilt (-180 to 180)
        gamma: event.gamma,  // left-right tilt (-90 to 90)
      };
    };

    if (typeof DeviceOrientationEvent !== 'undefined') {
      // iOS 13+ requires permission
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

    // Send sensor data at 2 Hz (GPS updates are slow, no need to spam)
    this._sensorIntervalId = setInterval(() => {
      if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return;
      if (!this._lastGps && !this._lastOrientation) return;

      const msg = {
        type: 'sensor_data',
        timestamp: Date.now(),
      };
      if (this._lastGps) msg.gps = this._lastGps;
      if (this._lastOrientation) msg.orientation = this._lastOrientation;

      try {
        this.socket.send(JSON.stringify(msg));
      } catch (e) { /* ignore */ }
    }, 500);
  }

  /**
   * Stop sensor capture and clean up listeners.
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
   * Stop streaming and clean up all resources.
   */
  stop() {
    this.isStreaming = false;

    // Stop sensor capture
    this._stopSensorCapture();

    // Stop capture loop
    if (this.captureIntervalId) {
      clearInterval(this.captureIntervalId);
      this.captureIntervalId = null;
    }

    // Send stop command and close WebSocket
    if (this.socket) {
      try {
        if (this.socket.readyState === WebSocket.OPEN) {
          this.socket.send(JSON.stringify({ type: 'stop' }));
        }
        this.socket.close();
      } catch (e) { /* ignore */ }
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
    this._emitStatus('stopped');
  }

  /**
   * Switch between front and rear cameras.
   */
  async switchCamera() {
    this.facingMode = this.facingMode === 'environment' ? 'user' : 'environment';

    if (this.isStreaming) {
      // Stop existing stream tracks
      if (this.stream) {
        this.stream.getTracks().forEach(track => track.stop());
      }

      // Restart with new facing mode
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
        this._emitError(`Failed to switch camera: ${err.message}`);
      }
    }
  }

  /**
   * Get current streaming statistics.
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

  // Internal helpers
  _emitStatus(status) {
    if (this.onStatusChange) this.onStatusChange(status);
  }

  _emitError(message) {
    console.error(`📱 CameraStream error: ${message}`);
    if (this.onError) this.onError(message);
  }
}

// Export singleton
export const cameraStreamService = new CameraStreamService();
