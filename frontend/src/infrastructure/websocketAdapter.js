/**
 * WebSocket adapter for real-time communication.
 * Handles connection, reconnection, and message parsing.
 */

import { decode } from '@msgpack/msgpack';
import { getWebSocketUrl, getConfig } from '../config';

/**
 * Event types
 */
export const WebSocketEvents = {
  CONNECTED: 'connected',
  DISCONNECTED: 'disconnected',
  ERROR: 'error',
  MESSAGE: 'message',
  FRAME: 'frame',
  PREDICTIONS: 'predictions',
  WORLD_UPDATE: 'world_update',
  STATUS: 'status',
  MAX_RECONNECT_REACHED: 'maxReconnectAttemptsReached'
};

/**
 * WebSocket connection manager
 */
class WebSocketAdapter {
  constructor() {
    this.socket = null;
    this.isConnected = false;
    this.url = null;
    this.listeners = new Map();
    this.reconnectAttempts = 0;
    
    this.maxReconnectAttempts = getConfig('websocket.maxReconnectAttempts', 5);
    this.reconnectDelay = getConfig('websocket.reconnectDelay', 1000);
    
    this.stats = {
      messagesReceived: 0,
      bytesReceived: 0,
      connectionTime: null,
      lastMessage: null
    };
  }

  /**
   * Connect to WebSocket server
   * @param {string} [url] - Optional URL override
   * @returns {Promise<void>}
   */
  connect(url) {
    this.url = url || getWebSocketUrl();
    
    return new Promise((resolve, reject) => {
      try {
        this.socket = new WebSocket(this.url);
        this.socket.binaryType = 'arraybuffer';

        this.socket.onopen = () => {
          console.log('[WebSocket] Connected');
          this.isConnected = true;
          this.reconnectAttempts = 0;
          this.stats.connectionTime = Date.now();
          this.emit(WebSocketEvents.CONNECTED);
          resolve();
        };

        this.socket.onmessage = (event) => {
          this.handleMessage(event);
        };

        this.socket.onclose = (event) => {
          console.log(`[WebSocket] Disconnected (code: ${event.code})`);
          this.isConnected = false;
          this.emit(WebSocketEvents.DISCONNECTED);
          
          // Attempt reconnection if not intentionally closed
          if (!event.wasClean) {
            this.attemptReconnect();
          }
        };

        this.socket.onerror = (error) => {
          console.error('[WebSocket] Error:', error);
          this.emit(WebSocketEvents.ERROR, error);
          reject(error);
        };

      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Handle incoming message
   * @param {MessageEvent} event
   */
  handleMessage(event) {
    try {
      this.stats.messagesReceived++;
      this.stats.bytesReceived += event.data.byteLength;
      this.stats.lastMessage = Date.now();

      // Decode msgpack
      const data = decode(new Uint8Array(event.data));
      
      // Debug logging
      console.log('[WebSocket] Received message:', {
        type: data.type,
        timestamp: data.timestamp,
        generation: data.generation,
        camera_count: data.camera_frames ? Object.keys(data.camera_frames).length : 0,
        world_objects: data.world_objects?.length
      });
      
      this.emit(WebSocketEvents.MESSAGE, data);
      
      // Emit specific event types
      if (data.type) {
        this.emit(data.type, data);
      }

    } catch (error) {
      console.error('[WebSocket] Failed to parse message:', error);
      this.emit(WebSocketEvents.ERROR, {
        message: 'Failed to parse message',
        originalError: error
      });
    }
  }

  /**
   * Attempt to reconnect
   */
  attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[WebSocket] Max reconnection attempts reached');
      this.emit(WebSocketEvents.MAX_RECONNECT_REACHED);
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    
    console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
    
    setTimeout(() => {
      this.connect(this.url).catch(() => {
        // Reconnection failed, will try again
      });
    }, delay);
  }

  /**
   * Send message to server
   * @param {Object} message
   * @returns {boolean}
   */
  send(message) {
    if (!this.isConnected || !this.socket) {
      console.warn('[WebSocket] Cannot send: not connected');
      return false;
    }

    try {
      this.socket.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('[WebSocket] Failed to send:', error);
      return false;
    }
  }

  /**
   * Send binary data
   * @param {ArrayBuffer} data
   * @returns {boolean}
   */
  sendBinary(data) {
    if (!this.isConnected || !this.socket) {
      console.warn('[WebSocket] Cannot send binary: not connected');
      return false;
    }

    try {
      this.socket.send(data);
      return true;
    } catch (error) {
      console.error('[WebSocket] Failed to send binary:', error);
      return false;
    }
  }

  /**
   * Disconnect from server
   */
  disconnect() {
    if (this.socket) {
      // Prevent reconnection on intentional close
      this.reconnectAttempts = this.maxReconnectAttempts;
      this.socket.close();
      this.socket = null;
    }
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
   * Emit event to listeners
   * @param {string} event
   * @param {any} data
   */
  emit(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`[WebSocket] Error in ${event} listener:`, error);
        }
      });
    }
  }

  /**
   * Get connection statistics
   * @returns {Object}
   */
  getStats() {
    return {
      ...this.stats,
      isConnected: this.isConnected,
      reconnectAttempts: this.reconnectAttempts
    };
  }
}

// Singleton instance
const webSocketAdapter = new WebSocketAdapter();

export { WebSocketAdapter };
export default webSocketAdapter;
