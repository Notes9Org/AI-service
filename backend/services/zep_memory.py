"""Zep memory service for long-term and short-term context.

Shared by Chat and Notes9 agent. Uses session_id (chat_sessions.id) as thread_id
and user_id so both agents read/write the same conversation context.

Uses Zep Cloud thread API (sessions/memory API deprecated).
"""
import os
from typing import Tuple, List, Dict, Any

import structlog

logger = structlog.get_logger()

# Lazy-initialized client
_zep_client = None


def _is_not_found(exc: BaseException) -> bool:
    """True if the error indicates thread/session does not exist (404)."""
    s = str(exc).lower()
    return "404" in s or "not found" in s or "page not found" in s


def _is_zep_enabled() -> bool:
    """True if Zep is configured and enabled."""
    enabled = os.getenv("ZEP_ENABLED", "true").lower() in ("true", "1", "yes")
    api_key = (os.getenv("ZEP_API_KEY") or "").strip()
    return enabled and bool(api_key)


def _get_client():
    """Get or create AsyncZep client. Raises if Zep not configured."""
    global _zep_client
    if _zep_client is None:
        from zep_cloud.client import AsyncZep

        api_key = os.getenv("ZEP_API_KEY")
        if not api_key:
            raise ValueError("ZEP_API_KEY is not set")
        _zep_client = AsyncZep(api_key=api_key)
    return _zep_client


async def _ensure_user(client, user_id: str) -> None:
    """Ensure Zep user exists. Required before creating threads. Idempotent."""
    try:
        await client.user.add(user_id=user_id)
        logger.debug("zep_user_created", user_id=user_id)
    except Exception as e:
        s = str(e).lower()
        if "already exists" in s or "409" in s or "duplicate" in s:
            pass  # User exists, continue
        else:
            logger.warning("zep_user_add_failed", user_id=user_id, error=str(e))


async def _ensure_thread(client, thread_id: str, user_id: str) -> None:
    """Ensure Zep thread exists. Idempotent."""
    try:
        await client.thread.create(thread_id=thread_id, user_id=user_id)
        logger.debug("zep_thread_created", thread_id=thread_id)
    except Exception as e:
        s = str(e).lower()
        if "already exists" in s or "409" in s or "duplicate" in s:
            pass  # Thread exists, continue
        else:
            logger.warning("zep_thread_create_failed", thread_id=thread_id, error=str(e))


async def get_context(session_id: str, user_id: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Get memory context for a session (thread).

    Returns:
        (context_string, recent_messages) where recent_messages are
        [{"role": "user"|"assistant", "content": "..."}] (last 4-6).
    """
    if not _is_zep_enabled():
        return "", []

    try:
        client = _get_client()
        await _ensure_user(client, user_id)
        await _ensure_thread(client, session_id, user_id)

        # Get context block (user summary + relevant facts)
        try:
            user_context = await client.thread.get_user_context(thread_id=session_id)
            context = getattr(user_context, "context", None) or ""
        except Exception as e:
            if _is_not_found(e):
                logger.debug("zep_thread_not_found", session_id=session_id)
                return "", []
            raise

        if not isinstance(context, str):
            context = str(context) if context else ""

        # Get recent messages (last 6)
        recent: List[Dict[str, Any]] = []
        try:
            thread_data = await client.thread.get(thread_id=session_id, lastn=6)
            messages = getattr(thread_data, "messages", None) or []
            for m in messages:
                role_val = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else "user")
                content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else "") or ""
                role = "user" if str(role_val).lower() in ("user", "human") else "assistant"
                recent.append({"role": role, "content": content})
        except Exception as e:
            if _is_not_found(e):
                pass  # No messages yet
            else:
                logger.warning("zep_get_messages_failed", session_id=session_id, error=str(e))

        return context, recent

    except Exception as e:
        logger.warning("zep_get_context_failed", session_id=session_id, error=str(e), exc_info=True)
        return "", []


async def add_messages(
    session_id: str,
    user_id: str,
    user_content: str,
    assistant_content: str,
) -> None:
    """
    Add a conversation turn to Zep memory.

    Call after each successful Chat or Notes9 response.
    """
    if not _is_zep_enabled():
        return

    try:
        from zep_cloud.types import Message

        client = _get_client()
        await _ensure_user(client, user_id)
        await _ensure_thread(client, session_id, user_id)

        # 3.x Message format: name, role, content (role_type -> role)
        messages = [
            Message(name="User", role="user", content=user_content),
            Message(name="Assistant", role="assistant", content=assistant_content),
        ]

        await client.thread.add_messages(thread_id=session_id, messages=messages)
        logger.debug("zep_add_messages_ok", session_id=session_id)

    except Exception as e:
        if _is_not_found(e):
            logger.debug("zep_add_messages_skipped", session_id=session_id, reason="thread_not_found")
        else:
            logger.warning("zep_add_messages_failed", session_id=session_id, error=str(e), exc_info=True)
