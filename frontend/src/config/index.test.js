/**
 * Tests for the configuration layer (path resolution, defaults, URL builders).
 */
import {
  getConfig,
  getWebSocketUrl,
  getCameraWebSocketUrl,
  getApiBaseUrl,
  validateConfig,
  getFullConfig,
} from './index';

describe('getConfig', () => {
  it('resolves a configured default value', () => {
    expect(getConfig('camera.maxCameras')).toBe(4);
  });

  it('resolves a nested path', () => {
    expect(getConfig('mobile.targetFps')).toBe(15);
  });

  it('returns the fallback for an unknown path', () => {
    expect(getConfig('nope.not.here', 'fallback')).toBe('fallback');
  });

  it('returns undefined-safe fallback when a path dead-ends early', () => {
    expect(getConfig('camera.maxCameras.deeper', 'x')).toBe('x');
  });
});

describe('URL builders', () => {
  it('builds a /ws websocket URL', () => {
    expect(getWebSocketUrl()).toMatch(/^wss?:\/\/.+:\d+\/ws$/);
  });

  it('builds a /ws/camera URL', () => {
    expect(getCameraWebSocketUrl()).toMatch(/\/ws\/camera$/);
  });

  it('builds an API base URL with protocol and port', () => {
    expect(getApiBaseUrl()).toMatch(/^https?:\/\/.+:\d+$/);
  });
});

describe('validateConfig', () => {
  it('reports valid for the default configuration', () => {
    const result = validateConfig();
    expect(result.valid).toBe(true);
    expect(result.errors).toEqual([]);
  });
});

describe('getFullConfig', () => {
  it('returns the full nested config object', () => {
    const cfg = getFullConfig();
    expect(cfg.camera.maxCameras).toBe(4);
    expect(cfg.mobile.jpegQuality).toBeGreaterThan(0);
  });
});
