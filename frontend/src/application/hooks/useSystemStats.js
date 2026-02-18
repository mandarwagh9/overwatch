/**
 * Hook for system stats.
 * Fetches and updates system information from backend.
 */

import { useState, useEffect, useCallback } from 'react';
import apiAdapter from '../../infrastructure/apiAdapter';

/**
 * Default system stats
 */
const DEFAULT_STATS = {
  cameras_active: 0,
  max_cameras: 4,
  target_fps: 24,
  connected_clients: 0,
  detection_engine_ready: false,
  tracking_active: false,
  detection_model: 'YOLOv8n'
};

/**
 * Use system stats hook
 * @returns {{
 *   stats: Object,
 *   isLoading: boolean,
 *   error: string|null,
 *   refresh: () => Promise<void>
 * }}
 */
export function useSystemStats() {
  const [stats, setStats] = useState(DEFAULT_STATS);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  /**
   * Fetch system stats
   */
  const fetchStats = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await apiAdapter.getStatus();
      
      if (response.isSuccess()) {
        setStats(prev => ({
          ...DEFAULT_STATS,
          ...response.data
        }));
      } else {
        // Don't set error for non-OK responses, just keep stale data
        console.warn('Failed to fetch system stats:', response.error);
      }
    } catch (err) {
      // Don't set error, just log it
      console.warn('Error fetching system stats:', err.message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Fetch stats periodically
  useEffect(() => {
    fetchStats();
    
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, [fetchStats]);

  return {
    stats,
    isLoading,
    error,
    refresh: fetchStats
  };
}

export default useSystemStats;
