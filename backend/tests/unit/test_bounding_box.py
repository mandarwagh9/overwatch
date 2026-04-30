"""Tests locking BoundingBox behavior."""
import pytest
from app.domain.entities import BoundingBox


def test_iou_identical_boxes_returns_1():
    a = BoundingBox(0, 0, 100, 100)
    b = BoundingBox(0, 0, 100, 100)
    assert a.iou(b) == pytest.approx(1.0)


def test_iou_disjoint_returns_0():
    a = BoundingBox(0, 0, 50, 50)
    b = BoundingBox(100, 100, 200, 200)
    assert a.iou(b) == 0.0


def test_iou_touching_returns_0():
    a = BoundingBox(0, 0, 50, 50)
    b = BoundingBox(50, 0, 100, 50)
    assert a.iou(b) == 0.0


def test_iou_one_inside_other():
    outer = BoundingBox(0, 0, 100, 100)
    inner = BoundingBox(25, 25, 75, 75)
    iou = outer.iou(inner)
    assert iou == pytest.approx(2500 / 10000)


def test_iou_half_overlap():
    a = BoundingBox(0, 0, 100, 100)
    b = BoundingBox(50, 0, 150, 100)
    iou = a.iou(b)
    assert iou == pytest.approx(5000 / 15000)


def test_iou_is_symmetric():
    a = BoundingBox(0, 0, 100, 100)
    b = BoundingBox(50, 50, 150, 150)
    assert a.iou(b) == pytest.approx(b.iou(a))


def test_zero_area_box_raises():
    """Current behavior: __post_init__ rejects degenerate boxes.
    LOAD-BEARING: detection_adapter relies on the ValueError to filter."""
    with pytest.raises(ValueError):
        BoundingBox(10, 20, 10, 30)
    with pytest.raises(ValueError):
        BoundingBox(10, 20, 30, 20)


def test_inverted_box_raises():
    with pytest.raises(ValueError):
        BoundingBox(100, 100, 50, 50)


def test_properties():
    b = BoundingBox(10, 20, 110, 220)
    assert b.width == 100
    assert b.height == 200
    assert b.area == 20000
    assert b.center == (60, 120)


def test_scale():
    b = BoundingBox(10, 20, 110, 220)
    s = b.scale(2.0, 0.5)
    assert s.x1 == 20
    assert s.y1 == 10
    assert s.x2 == 220
    assert s.y2 == 110
