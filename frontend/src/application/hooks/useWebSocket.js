/**
 * Hook for WebSocket connection management.
 * Provides reactive state for connection status and messages.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import webSocketAdapter, { WebSocketEvents } from '../../infrastructure/websocketAdapter';

/**
 * Use WebSocket hook
 * @returns {{
 *   isConnected: boolean,
 *   error: string|null,
 *   stats: Object,
 *   connect: (url?: string) => Promise<void>,
 *   disconnect: () => void,
 *   on: (event: string, callback: Function) => void,
 *   off: (event: string, callback: Function) => void
 * }}
 */
export function useWebSocket() {
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState({
    messagesReceived: 0,
    bytesReceived: 0,
    isConnected: false,
    reconnectAttempts: 0
  });

  const adapterRef = useRef(webSocketAdapter);

  useEffect(() => {
    const adapter = adapterRef.current;

    // Set up event listeners
    const handleConnected = () => {
      setIsConnected(true);
      setError(null);
    };

    const handleDisconnected = () => {
      setIsConnected(false);
    };

    const handleError = (err) => {
      setError(err.message || 'WebSocket error');
    };

    const updateStats = () => {
      setStats(adapter.getStats());
    };

    adapter.on(WebSocketEvents.CONNECTED, handleConnected);
    adapter.on(WebSocketEvents.DISCONNECTED, handleDisconnected);
    adapter.on(WebSocketEvents.ERROR, handleError);

    // Update stats periodically
    const statsInterval = setInterval(updateStats, 1000);

    return () => {
      adapter.off(WebSocketEvents.CONNECTED, handleConnected);
      adapter.off(WebSocketEvents.DISCONNECTED, handleDisconnected);
      adapter.off(WebSocketEvents.ERROR, handleError);
      clearInterval(statsInterval);
    };
  }, []);

  const connect = useCallback(async (url) => {
    try {
      await adapterRef.current.connect(url);
    } catch (err) {
      setError(err.message);
      throw err;
    }
  }, []);

  const disconnect = useCallback(() => {
    adapterRef.current.disconnect();
  }, []);

  const on = useCallback((event, callback) => {
    adapterRef.current.on(event, callback);
  }, []);

  const off = useCallback((event, callback) => {
    adapterRef.current.off(event, callback);
  }, []);

  return {
    isConnected,
    error,
    stats,
    connect,
    disconnect,
    on,
    off
  };
}

export default useWebSocket;
