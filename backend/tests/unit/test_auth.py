"""Tests for optional JWT auth helpers."""
from unittest.mock import Mock

import pytest

from app.infrastructure.auth import verify_token, issue_token


def _settings(enabled=False, secret="test-secret-must-be-long-enough-32chars"):
    s = Mock()
    s.auth_enabled = enabled
    s.jwt_secret = secret
    return s


def test_verify_token_returns_true_when_auth_disabled():
    assert verify_token(None, _settings(enabled=False)) is True
    assert verify_token("anything", _settings(enabled=False)) is True


def test_verify_token_rejects_missing_token_when_auth_enabled():
    assert verify_token(None, _settings(enabled=True)) is False
    assert verify_token("", _settings(enabled=True)) is False


def test_issue_then_verify_roundtrip():
    pytest.importorskip("jwt")
    s = _settings(enabled=True)
    token = issue_token("alice", s)
    assert token is not None
    assert verify_token(token, s) is True


def test_verify_rejects_token_with_wrong_secret():
    pytest.importorskip("jwt")
    s = _settings(enabled=True, secret="secret-a-must-be-long-enough-32chars")
    token = issue_token("alice", s)
    s2 = _settings(enabled=True, secret="secret-b-must-be-long-enough-32chars")
    assert verify_token(token, s2) is False
