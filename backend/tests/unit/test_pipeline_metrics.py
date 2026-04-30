"""Tests for PipelineMetrics."""
import time

import pytest

from app.application.services import PipelineMetrics


def test_initial_metrics_are_zero():
    m = PipelineMetrics()
    assert m.frames_processed == 0
    assert m.average_processing_time_ms == 0.0
    assert m.total_processing_time_ms == 0.0
    assert m.current_fps == 0.0
    assert m.dropped_frames == 0


def test_update_increments_counts_and_averages():
    m = PipelineMetrics()
    m.update(10.0)
    m.update(20.0)
    assert m.frames_processed == 2
    assert m.total_processing_time_ms == pytest.approx(30.0)
    assert m.average_processing_time_ms == pytest.approx(15.0)


def test_fps_computed_after_two_updates():
    m = PipelineMetrics()
    m.update(5.0)
    time.sleep(0.05)
    m.update(5.0)
    assert m.current_fps > 0


def test_first_update_does_not_set_fps():
    """current_fps stays at zero after only one update because it requires
    two timestamps to compute elapsed time."""
    m = PipelineMetrics()
    m.update(5.0)
    assert m.current_fps == 0.0
