/**
 * Unit tests for pure domain helpers and constants.
 * These are deterministic and require no DOM.
 */
import {
  calculateIoU,
  getBoundingBoxCenter,
  getBoundingBoxDimensions,
  IFF_COLORS,
  COCO_SKELETON,
  COCO_KEYPOINTS,
  KEYPOINT_CONFIDENCE_THRESHOLD,
} from './entities';

describe('getBoundingBoxCenter', () => {
  it('returns the geometric center', () => {
    expect(getBoundingBoxCenter({ x1: 0, y1: 0, x2: 10, y2: 20 })).toEqual([5, 10]);
  });
});

describe('getBoundingBoxDimensions', () => {
  it('returns width and height', () => {
    expect(getBoundingBoxDimensions({ x1: 2, y1: 3, x2: 12, y2: 23 })).toEqual({
      width: 10,
      height: 20,
    });
  });
});

describe('calculateIoU', () => {
  const box = { x1: 0, y1: 0, x2: 10, y2: 10 };

  it('is 1.0 for identical boxes', () => {
    expect(calculateIoU(box, box)).toBeCloseTo(1.0, 6);
  });

  it('is 0 for fully disjoint boxes', () => {
    const far = { x1: 100, y1: 100, x2: 110, y2: 110 };
    expect(calculateIoU(box, far)).toBe(0);
  });

  it('computes partial overlap correctly', () => {
    // overlap region 5x10=50; union = 100 + 100 - 50 = 150 => 1/3
    const other = { x1: 5, y1: 0, x2: 15, y2: 10 };
    expect(calculateIoU(box, other)).toBeCloseTo(50 / 150, 6);
  });

  it('never divides by zero for degenerate boxes', () => {
    const degenerate = { x1: 0, y1: 0, x2: 0, y2: 0 };
    expect(calculateIoU(degenerate, degenerate)).toBe(0);
  });
});

describe('IFF_COLORS', () => {
  it('defines every IFF category used by the HUD renderer', () => {
    for (const key of ['UNKNOWN', 'FRIENDLY', 'PROJECTED', 'HOSTILE', 'WORLD', 'HUD']) {
      expect(IFF_COLORS[key]).toBeDefined();
      expect(typeof IFF_COLORS[key].stroke).toBe('string');
    }
  });
});

describe('COCO pose constants', () => {
  it('has 17 keypoint names', () => {
    expect(COCO_KEYPOINTS).toHaveLength(17);
  });

  it('only references valid keypoint indices in the skeleton', () => {
    for (const [a, b] of COCO_SKELETON) {
      expect(a).toBeGreaterThanOrEqual(0);
      expect(b).toBeGreaterThanOrEqual(0);
      expect(a).toBeLessThan(COCO_KEYPOINTS.length);
      expect(b).toBeLessThan(COCO_KEYPOINTS.length);
    }
  });

  it('uses a confidence threshold in [0, 1]', () => {
    expect(KEYPOINT_CONFIDENCE_THRESHOLD).toBeGreaterThanOrEqual(0);
    expect(KEYPOINT_CONFIDENCE_THRESHOLD).toBeLessThanOrEqual(1);
  });
});
