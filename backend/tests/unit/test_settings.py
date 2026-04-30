"""Tests for Pydantic Settings."""
import os

import pytest

from app.infrastructure.config_adapter import Settings


def _scrub_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("CAM_", "CAMERA_", "JWT_", "AUTH_")):
            monkeypatch.delenv(k, raising=False)


def test_default_settings_load(monkeypatch):
    """With no env vars, defaults should produce a valid Settings object."""
    _scrub_env(monkeypatch)
    s = Settings()
    assert s.host
    assert isinstance(s.port, int)


def test_max_cameras_default_is_at_least_one(monkeypatch):
    _scrub_env(monkeypatch)
    s = Settings()
    assert s.max_cameras >= 1


def test_target_fps_is_positive(monkeypatch):
    _scrub_env(monkeypatch)
    s = Settings()
    assert s.target_fps > 0


def test_log_level_uppercased(monkeypatch):
    """The validator coerces log level to uppercase."""
    _scrub_env(monkeypatch)
    monkeypatch.setenv("LOG_LEVEL", "debug")
    s = Settings()
    assert s.log_level == "DEBUG"
