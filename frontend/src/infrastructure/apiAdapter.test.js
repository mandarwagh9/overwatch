/**
 * Tests for the HTTP API adapter (ApiResponse wrapper + fetch-based client).
 */
import { ApiAdapter, ApiResponse } from './apiAdapter';

describe('ApiResponse', () => {
  it('isSuccess reflects the ok flag', () => {
    expect(new ApiResponse(true, {}, null, 200).isSuccess()).toBe(true);
    expect(new ApiResponse(false, null, 'x', 500).isSuccess()).toBe(false);
  });

  it('getOrThrow returns data on success', () => {
    expect(new ApiResponse(true, { a: 1 }, null, 200).getOrThrow()).toEqual({ a: 1 });
  });

  it('getOrThrow throws on failure', () => {
    expect(() => new ApiResponse(false, null, 'boom', 500).getOrThrow()).toThrow('boom');
    expect(() => new ApiResponse(false, null, null, 404).getOrThrow()).toThrow('HTTP 404');
  });
});

describe('ApiAdapter.buildUrl', () => {
  const api = new ApiAdapter();

  it('adds a leading slash when missing', () => {
    expect(api.buildUrl('health')).toBe(`${api.baseUrl}/health`);
  });

  it('preserves an existing leading slash', () => {
    expect(api.buildUrl('/status')).toBe(`${api.baseUrl}/status`);
  });
});

describe('ApiAdapter requests', () => {
  let api;

  beforeEach(() => {
    // AbortSignal.timeout may be absent in the test environment.
    AbortSignal.timeout = jest.fn(() => new AbortController().signal);
    global.fetch = jest.fn();
    api = new ApiAdapter();
  });

  it('GET returns a successful ApiResponse with parsed data', async () => {
    global.fetch.mockResolvedValue({ ok: true, status: 200, json: async () => ({ hi: 1 }) });
    const res = await api.get('/health');
    expect(res.ok).toBe(true);
    expect(res.data).toEqual({ hi: 1 });
    expect(res.status).toBe(200);
  });

  it('GET maps an HTTP error to a failed ApiResponse', async () => {
    global.fetch.mockResolvedValue({ ok: false, status: 503, statusText: 'Unavailable' });
    const res = await api.get('/status');
    expect(res.ok).toBe(false);
    expect(res.status).toBe(503);
    expect(res.error).toMatch(/503/);
  });

  it('GET maps a network error to status 0', async () => {
    global.fetch.mockRejectedValue(new Error('network down'));
    const res = await api.get('/status');
    expect(res.ok).toBe(false);
    expect(res.status).toBe(0);
    expect(res.error).toBe('network down');
  });

  it('startCamera POSTs to the camera start path', async () => {
    global.fetch.mockResolvedValue({ ok: true, status: 200, json: async () => ({}) });
    await api.startCamera(2);
    const [url, options] = global.fetch.mock.calls[0];
    expect(url).toMatch(/\/cameras\/2\/start$/);
    expect(options.method).toBe('POST');
  });

  it('stopCamera POSTs to the camera stop path', async () => {
    global.fetch.mockResolvedValue({ ok: true, status: 200, json: async () => ({}) });
    await api.stopCamera(3);
    expect(global.fetch.mock.calls[0][0]).toMatch(/\/cameras\/3\/stop$/);
  });

  it('POST serializes a JSON body', async () => {
    global.fetch.mockResolvedValue({ ok: true, status: 200, json: async () => ({}) });
    await api.post('/api/token', { subject: 'me' });
    expect(global.fetch.mock.calls[0][1].body).toBe(JSON.stringify({ subject: 'me' }));
  });
});
