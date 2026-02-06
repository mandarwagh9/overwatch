/**
 * WebSocket service for real-time communication with Overwatch backend
 */

import { decode } from '@msgpack/msgpack';

export class WebSocketService {
  constructor() {
    this.socket = null;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000;
    this.listeners = new Map();
    this.stats = {
      messagesReceived: 0,
      bytesReceived: 0,
      connectionTime: null,
      lastMessage: null
    };
  }

  /**
   * Connect to WebSocket server
   */
  connect(url) {
    // Use env var for backend host, fallback to same-origin
    if (!url) {
      const host = process.env.REACT_APP_BACKEND_HOST || window.location.hostname || 'localhost';
      const port = process.env.REACT_APP_BACKEND_PORT || '8000';
      url = `wss://${host}:${port}/ws`;
    }
    this._url = url;  // Save for reconnect
    
    return new Promise((resolve, reject) => {
      try {
        this.socket = new WebSocket(url);
        this.socket.binaryType = 'arraybuffer';

        this.socket.onopen = () => {
          console.log('🔌 Connected to Overwatch backend');
          this.isConnected = true;
          this.reconnectAttempts = 0;
          this.stats.connectionTime = Date.now();
          
          this.emit('connected');
          resolve();
        };

        this.socket.onmessage = (event) => {
          this.handleMessage(event);
        };

        this.socket.onclose = () => {
          console.log('🔌 Disconnected from Overwatch backend');
          this.isConnected = false;
          this.emit('disconnected');
          
          // Attempt reconnection
          this.attemptReconnect(this._url || url);
        };

        this.socket.onerror = (error) => {
          console.error('❌ WebSocket error:', error);
          this.emit('error', error);
          reject(error);
        };

      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Handle incoming messages
   */
  handleMessage(event) {
    try {
      this.stats.messagesReceived++;
      this.stats.bytesReceived += event.data.byteLength;
      this.stats.lastMessage = Date.now();

      // Decode msgpack message
      const data = decode(new Uint8Array(event.data));
      
      this.emit('message', data);
      
      // Emit specific message types
      if (data.type) {
        this.emit(data.type, data);
      }

    } catch (error) {
      console.error('❌ Failed to parse message:', error);
    }
  }

  /**
   * Attempt to reconnect
   */
  attemptReconnect(url) {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('❌ Max reconnection attempts reached');
      this.emit('maxReconnectAttemptsReached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    
    console.log(`🔄 Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    setTimeout(() => {
      this.connect(url).catch(() => {
        // Failed to reconnect, will try again
      });
    }, delay);
  }

  /**
   * Send message to server
   */
  send(message) {
    if (!this.isConnected || !this.socket) {
      console.warn('⚠️ Cannot send message: not connected');
      return false;
    }

    try {
      this.socket.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('❌ Failed to send message:', error);
      return false;
    }
  }

  /**
   * Disconnect from server
   */
  disconnect() {
    if (this.socket) {
      this.socket.close();
    }
  }

  /**
   * Add event listener
   */
  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
  }

  /**
   * Remove event listener
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
   */
  emit(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`❌ Error in ${event} listener:`, error);
        }
      });
    }
  }

  /**
   * Get connection statistics
   */
  getStats() {
    return {
      ...this.stats,
      isConnected: this.isConnected,
      reconnectAttempts: this.reconnectAttempts
    };
  }
}

// Create singleton instance
export const websocketService = new WebSocketService();