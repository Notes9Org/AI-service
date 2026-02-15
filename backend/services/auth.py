"""
Bearer token authentication using Supabase Auth JWTs.

Verifies the access_token (JWT) issued when a user logs in via Supabase Auth.
Protected routes require Authorization: Bearer <access_token>.
"""
from typing import Any, Dict, Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from services.config import get_supabase_config

security = HTTPBearer(auto_error=False)


class CurrentUser:
    """Authenticated user from JWT claims."""

    def __init__(self, sub: str, email: Optional[str] = None, role: Optional[str] = None):
        self.sub = sub
        self.email = email
        self.role = role

    @property
    def user_id(self) -> str:
        return self.sub


def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify Supabase Auth JWT and return decoded payload.
    Raises HTTPException 401 if token is invalid or expired.
    """
    config = get_supabase_config()
    jwt_secret = config.jwt_secret
    if not jwt_secret:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Authentication is not configured: SUPABASE_JWT_SECRET is required.",
        )
    try:
        payload = jwt.decode(
            token,
            jwt_secret,
            algorithms=["HS256"],
            options={"verify_exp": True, "verify_aud": False},
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired.",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token.",
        )


def payload_to_user(payload: Dict[str, Any]) -> CurrentUser:
    """Build CurrentUser from JWT payload."""
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing subject.",
        )
    return CurrentUser(
        sub=str(sub),
        email=payload.get("email"),
        role=payload.get("role"),
    )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> CurrentUser:
    """
    FastAPI dependency: extract Bearer token, verify JWT, return current user.
    Raises 401 if token is missing or invalid.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated: Bearer token required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    payload = verify_token(token)
    return payload_to_user(payload)


def verify_token_for_websocket(token: str) -> CurrentUser:
    """
    Verify JWT from WebSocket (query param or first message).
    Same logic as get_current_user but for raw token string.
    """
    payload = verify_token(token)
    return payload_to_user(payload)
