/**
 * API adapter for HTTP communication with backend.
 */

import { getApiBaseUrl } from '../config';

/**
 * API response wrapper
 */
class ApiResponse {
  constructor(ok, data, error, status) {
    this.ok = ok;
    this.data = data;
    this.error = error;
    this.status = status;
  }

  /**
   * Check if response is successful
   * @returns {boolean}
   */
  isSuccess() {
    return this.ok;
  }

  /**
   * Get response data or throw error
   * @returns {any}
   * @throws {Error}
   */
  getOrThrow() {
    if (!this.ok) {
      throw new Error(this.error || `HTTP ${this.status}`);
    }
    return this.data;
  }
}

/**
 * API client
 */
class ApiAdapter {
  constructor() {
    this.baseUrl = getApiBaseUrl();
  }

  /**
   * Build full URL
   * @param {string} path
   * @returns {string}
   */
  buildUrl(path) {
    const cleanPath = path.startsWith('/') ? path : `/${path}`;
    return `${this.baseUrl}${cleanPath}`;
  }

  /**
   * Perform GET request
   * @param {string} path
   * @returns {Promise<ApiResponse>}
   */
  async get(path) {
    try {
      const url = this.buildUrl(path);
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        }
      });

      if (!response.ok) {
        return new ApiResponse(
          false,
          null,
          `HTTP ${response.status}: ${response.statusText}`,
          response.status
        );
      }

      const data = await response.json();
      return new ApiResponse(true, data, null, response.status);

    } catch (error) {
      return new ApiResponse(
        false,
        null,
        error.message || 'Network error',
        0
      );
    }
  }

  /**
   * Perform POST request
   * @param {string} path
   * @param {Object} [body]
   * @returns {Promise<ApiResponse>}
   */
  async post(path, body = null) {
    try {
      const url = this.buildUrl(path);
      const options = {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        }
      };

      if (body) {
        options.body = JSON.stringify(body);
      }

      const response = await fetch(url, options);

      if (!response.ok) {
        return new ApiResponse(
          false,
          null,
          `HTTP ${response.status}: ${response.statusText}`,
          response.status
        );
      }

      const data = await response.json();
      return new ApiResponse(true, data, null, response.status);

    } catch (error) {
      return new ApiResponse(
        false,
        null,
        error.message || 'Network error',
        0
      );
    }
  }

  /**
   * Health check
   * @returns {Promise<ApiResponse>}
   */
  async healthCheck() {
    return this.get('/health');
  }

  /**
   * Get system status
   * @returns {Promise<ApiResponse>}
   */
  async getStatus() {
    return this.get('/status');
  }

  /**
   * List cameras
   * @returns {Promise<ApiResponse>}
   */
  async listCameras() {
    return this.get('/cameras');
  }

  /**
   * Start camera
   * @param {number} cameraId
   * @returns {Promise<ApiResponse>}
   */
  async startCamera(cameraId) {
    return this.post(`/cameras/${cameraId}/start`);
  }

  /**
   * Stop camera
   * @param {number} cameraId
   * @returns {Promise<ApiResponse>}
   */
  async stopCamera(cameraId) {
    return this.post(`/cameras/${cameraId}/stop`);
  }
}

// Singleton instance
const apiAdapter = new ApiAdapter();

export { ApiAdapter, ApiResponse };
export default apiAdapter;
