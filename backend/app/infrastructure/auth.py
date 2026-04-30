"""Optional JWT verification — additive, gated by settings.auth_enabled."""
from __future__ import annotations
import logging
from typing import Optional

from app.infrastructure.config_adapter import Settings

logger = logging.getLogger(__name__)


def verify_token(token: Optional[str], settings: Settings) -> bool:
    """Return True if token is valid OR auth disabled. False otherwise.
    Default-off: when settings.auth_enabled is False, always returns True."""
    if not settings.auth_enabled:
        return True
    if not token:
        return False
    try:
        import jwt
    except ImportError:
        logger.error("auth_enabled but PyJWT not installed; refusing all connections")
        return False
    try:
        jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        return True
    except Exception as e:
        logger.warning(f"JWT verification failed: {type(e).__name__}")
        return False


def issue_token(subject: str, settings: Settings, expires_minutes: int = 60) -> Optional[str]:
    """Issue an HS256 JWT. Returns None if PyJWT missing or auth disabled."""
    if not settings.auth_enabled:
        return None
    try:
        import jwt
    except ImportError:
        return None
    from datetime import datetime, timedelta, timezone
    payload = {
        "sub": subject,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=expires_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")
