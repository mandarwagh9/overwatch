/**
 * Tests for the pure HUD helpers in CameraDisplay (IFF colour + distance).
 */
import { getIFFColor, estimateDistance } from './CameraDisplay';
import { IFF_COLORS } from '../domain/entities';

describe('getIFFColor', () => {
  it('returns FRIENDLY for a non-prediction (track/detection)', () => {
    expect(getIFFColor({}, false)).toBe(IFF_COLORS.FRIENDLY);
  });

  it('returns PROJECTED (green) for a HOMOGRAPHY prediction', () => {
    expect(getIFFColor({ prediction_method: 'HOMOGRAPHY' }, true)).toBe(IFF_COLORS.PROJECTED);
  });

  it('returns WORLD (orange) for a WORLD prediction', () => {
    expect(getIFFColor({ prediction_method: 'WORLD' }, true)).toBe(IFF_COLORS.WORLD);
  });

  it('returns HOSTILE (red) for an EXTRAP prediction', () => {
    expect(getIFFColor({ prediction_method: 'EXTRAP' }, true)).toBe(IFF_COLORS.HOSTILE);
  });

  it('falls back to the homography_source flag when no method is given', () => {
    expect(getIFFColor({ homography_source: true }, true)).toBe(IFF_COLORS.PROJECTED);
    expect(getIFFColor({}, true)).toBe(IFF_COLORS.HOSTILE);
  });
});

describe('estimateDistance', () => {
  it('returns null for a non-positive bbox height', () => {
    expect(estimateDistance(0)).toBeNull();
    expect(estimateDistance(-10)).toBeNull();
  });

  it('returns ~1.7 m at the reference pixel height (520 px)', () => {
    expect(estimateDistance(520)).toBeCloseTo(1.7, 1);
  });

  it('is inversely proportional to bbox height (larger bbox = closer)', () => {
    expect(estimateDistance(1040)).toBeLessThan(estimateDistance(260));
  });
});
