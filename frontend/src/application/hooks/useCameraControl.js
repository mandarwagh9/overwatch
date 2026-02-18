/**
 * Hook for camera control operations.
 * Handles starting and stopping cameras via API.
 */

import { useState, useCallback } from 'react';
import apiAdapter from '../../infrastructure/apiAdapter';

/**
 * Use camera control hook
 * @returns {{
 *   isLoading: boolean,
 *   error: string|null,
 *   toggleCamera: (cameraId: number, isActive: boolean) => Promise<void>,
 *   startCamera: (cameraId: number) => Promise<void>,
 *   stopCamera: (cameraId: number) => Promise<void>,
 *   clearError: () => void
 * }}
 */
export function useCameraControl() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  /**
   * Toggle camera state
   * @param {number} cameraId
   * @param {boolean} isActive - Current active state
   */
  const toggleCamera = useCallback(async (cameraId, isActive) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = isActive 
        ? await apiAdapter.stopCamera(cameraId)
        : await apiAdapter.startCamera(cameraId);

      if (!response.isSuccess()) {
        throw new Error(response.error || `Failed to ${isActive ? 'stop' : 'start'} camera ${cameraId}`);
      }

    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Start a camera
   * @param {number} cameraId
   */
  const startCamera = useCallback(async (cameraId) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await apiAdapter.startCamera(cameraId);
      
      if (!response.isSuccess()) {
        throw new Error(response.error || `Failed to start camera ${cameraId}`);
      }

    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Stop a camera
   * @param {number} cameraId
   */
  const stopCamera = useCallback(async (cameraId) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await apiAdapter.stopCamera(cameraId);
      
      if (!response.isSuccess()) {
        throw new Error(response.error || `Failed to stop camera ${cameraId}`);
      }

    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Clear error state
   */
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    isLoading,
    error,
    toggleCamera,
    startCamera,
    stopCamera,
    clearError
  };
}

export default useCameraControl;
