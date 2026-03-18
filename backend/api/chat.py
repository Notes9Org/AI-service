"""Chat API routes.

POST /chat        : regular JSON chat (no streaming).
POST /chat/stream : SSE streaming endpoint for general chat.
Both require Bearer token (Supabase Auth).
"""
import asyncio
import json
import os
import time
from queue import Empty, Queue
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

import structlog

from agents.prompt_loader import load_prompt
from agents.services.anthropic_client import AnthropicChatClient
from agents.services.llm_client import LLMError
from services.auth import CurrentUser, get_current_user
from services.config import get_app_config
from services.zep_memory import get_context as zep_get_context
from services.zep_memory import add_messages as zep_add_messages

logger = structlog.get_logger()
router = APIRouter(prefix="/chat", tags=["chat"])

# Keywords that suggest the query needs current/external info (web search)
_WEB_SEARCH_KEYWORDS = (
    "current", "latest", "recent", "today", "now", "breaking", "news",
    "search the web", "look up", "find online", "search for", "look it up",
    "stock", "price", "weather", "scores", "headlines", "happening",
    "2024", "2025", "2026","this year", "right now", "up to date", "recently",
)

# Override: set WEB_SEARCH_ALWAYS=true to always enable; WEB_SEARCH_NEVER=true to never
def _should_use_web_search(content: str) -> bool:
    """True only when the query likely needs current/external information."""
    if os.getenv("WEB_SEARCH_NEVER", "").lower() in ("true", "1", "yes"):
        return False
    if os.getenv("WEB_SEARCH_ALWAYS", "").lower() in ("true", "1", "yes"):
        return True
    q = (content or "").lower().strip()
    if len(q) < 3:
        return False
    return any(kw in q for kw in _WEB_SEARCH_KEYWORDS)


_CHAT_SYSTEM_PROMPT = load_prompt("chat", "chat_system")


class ChatMessage(BaseModel):
    """Single message in conversation history."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request schema for chat - content, session_id, history for memory management; user_id from auth."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "What is the molar mass of glucose?",
                "session_id": "session-456",
                "history": [
                    {"role": "user", "content": "What is glucose?"},
                    {"role": "assistant", "content": "Glucose is a simple sugar..."}
                ]
            }
        }
    )
    content: str = Field(..., description="User message content")
    session_id: str = Field(..., description="Session ID for tracking and context")
    history: List[ChatMessage] = Field(
        default_factory=list,
        description="Previous messages in the conversation for memory/context"
    )


@router.post("", summary="Chat with Claude / LLM (JSON, non-streaming)")
async def chat(
    request: ChatRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    **Chat endpoint (non-streaming).**

    Calls Anthropic once and returns the full assistant message as JSON:
    `{ "content": "<assistant text>", "role": "assistant", "sources"?: [...], "searched_web"?: true }`
    When web search is used, includes sources and searched_web for citation display.
    """
    user_id = current_user.user_id
    session_id = request.session_id
    use_web = _should_use_web_search(request.content)
    logger.info(
        "chat_request",
        user_id=user_id,
        session_id=session_id,
        content_len=len(request.content),
        history_len=len(request.history),
        use_web=use_web,
    )

    # Build messages and system from Zep or fallback to request.history
    config = get_app_config()
    if config.zep_enabled:
        zep_context, recent_messages = await zep_get_context(session_id, user_id)
        messages = recent_messages + [{"role": "user", "content": request.content}]
        system = _CHAT_SYSTEM_PROMPT
        if zep_context and zep_context.strip():
            system = f"{_CHAT_SYSTEM_PROMPT}\n\n[Relevant context from past conversations]\n{zep_context}"
    else:
        messages = [{"role": m.role, "content": m.content} for m in request.history] + [
            {"role": "user", "content": request.content}
        ]
        system = _CHAT_SYSTEM_PROMPT

    try:
        client = AnthropicChatClient()

        result = await asyncio.to_thread(
            client.chat,
            messages=messages,
            system=system,
            use_web=use_web,
        )
        if config.zep_enabled and result.get("content"):
            await zep_add_messages(
                session_id=session_id,
                user_id=user_id,
                user_content=request.content,
                assistant_content=result["content"],
            )
        return JSONResponse(status_code=200, content=result)
    except LLMError as e:
        logger.error("chat_failed_llm", error=str(e))
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        logger.error("chat_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Chat request failed") from e


def _format_sse(event_type: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def _stream_chat_generator(request: ChatRequest, current_user: CurrentUser):
    """Async generator that streams chat tokens as SSE events.

    Event lifecycle (matches notes9 pattern):
      thinking(chat started) → thinking(browsing) → token* → source* → thinking(completed) → done
    Uses ping keepalive when idle (like notes9) so long web searches don't timeout.
    """
    user_id = current_user.user_id
    session_id = request.session_id
    use_web = _should_use_web_search(request.content)
    logger.info(
        "chat_stream_request",
        user_id=user_id,
        session_id=session_id,
        content_len=len(request.content),
        history_len=len(request.history),
        use_web=use_web,
    )

    # Build messages and system from Zep or fallback to request.history
    config = get_app_config()
    if config.zep_enabled:
        zep_context, recent_messages = await zep_get_context(session_id, user_id)
        messages = recent_messages + [{"role": "user", "content": request.content}]
        system = _CHAT_SYSTEM_PROMPT
        if zep_context and zep_context.strip():
            system = f"{_CHAT_SYSTEM_PROMPT}\n\n[Relevant context from past conversations]\n{zep_context}"
    else:
        messages = [{"role": m.role, "content": m.content} for m in request.history] + [
            {"role": "user", "content": request.content}
        ]
        system = _CHAT_SYSTEM_PROMPT

    queue: Queue = Queue()

    def run_stream():
        try:
            client = AnthropicChatClient()
            content_parts: list[str] = []
            sources: list[dict] = []
            for item in client.chat_stream(
                messages=messages,
                system=system,
                use_web=use_web,
            ):
                if item.get("type") == "token":
                    text = item.get("text", "")
                    content_parts.append(text)
                    queue.put(("token", {"text": text}))
                elif item.get("type") == "source":
                    src = {"url": item.get("url", ""), "title": item.get("title", "")}
                    sources.append(src)
                    queue.put(("source", src))
            content = "".join(content_parts)
            done_data = {"content": content, "role": "assistant"}
            if sources:
                done_data["sources"] = sources
                done_data["searched_web"] = True
            queue.put(("stream_complete", done_data))
        except Exception as e:
            queue.put(("error", {"error": str(e)}))

    yield _format_sse("thinking", {
        "node": "chat",
        "status": "started",
        "message": "Generating response...",
    })
    await asyncio.sleep(0)
    yield _format_sse("thinking", {
        "node": "browsing",
        "status": "started",
        "message": "Searching the web when needed...",
    })
    await asyncio.sleep(0)

    task = asyncio.to_thread(run_stream)

    async def collect():
        await task

    stream_task = asyncio.create_task(collect())
    final_content: dict | None = None
    PING_INTERVAL_SEC = 15.0
    last_ping = 0.0

    try:
        loop = asyncio.get_event_loop()
        while True:
            try:
                event_type, data = await loop.run_in_executor(
                    None, lambda: queue.get(timeout=0.5)
                )
            except Empty:
                if stream_task.done():
                    while True:
                        try:
                            event_type, data = queue.get_nowait()
                            if event_type == "stream_complete":
                                final_content = data
                            elif event_type == "error":
                                yield _format_sse("error", data)
                                await asyncio.sleep(0)
                            else:
                                yield _format_sse(event_type, data)
                                await asyncio.sleep(0)
                        except Empty:
                            break
                    break
                now = time.time()
                if now - last_ping >= PING_INTERVAL_SEC:
                    last_ping = now
                    yield _format_sse("ping", {"ts": now})
                    await asyncio.sleep(0)
                continue

            if event_type == "stream_complete":
                final_content = data
                break
            if event_type == "error":
                yield _format_sse("error", data)
                await asyncio.sleep(0)
                break

            yield _format_sse(event_type, data)
            await asyncio.sleep(0)

        await stream_task

        if final_content:
            if config.zep_enabled and final_content.get("content"):
                await zep_add_messages(
                    session_id=session_id,
                    user_id=user_id,
                    user_content=request.content,
                    assistant_content=final_content["content"],
                )
            yield _format_sse("thinking", {
                "node": "browsing",
                "status": "completed",
                "message": "Search complete",
            })
            await asyncio.sleep(0)
            yield _format_sse("thinking", {
                "node": "chat",
                "status": "completed",
                "message": "Response complete",
            })
            await asyncio.sleep(0)
            yield _format_sse("done", final_content)
            await asyncio.sleep(0)
    except Exception as e:
        logger.error("chat_stream failed", error=str(e))
        yield _format_sse("error", {"error": str(e)})
        await asyncio.sleep(0)


@router.post("/stream", summary="Chat with Claude / LLM (SSE streaming)")
async def chat_stream(
    request: ChatRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    **Chat stream endpoint.** Same as POST /chat but returns text/event-stream.
    Events: thinking, ping (keepalive), token, source (web links), done, error.
    Uses web search when needed; sources/links are emitted as source events.
    """
    return StreamingResponse(
        _stream_chat_generator(request, current_user=current_user),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
