"""Chat API routes.

POST /chat        : regular JSON chat (no streaming).
POST /chat/stream : SSE streaming endpoint for general chat.
Both require Bearer token (Supabase Auth).
"""
import asyncio
import json
import time
from queue import Empty, Queue
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

import structlog

from agents.services.anthropic_client import AnthropicChatClient
from agents.services.llm_client import LLMError
from services.auth import CurrentUser, get_current_user

logger = structlog.get_logger()
router = APIRouter(prefix="/chat", tags=["chat"])

DEFAULT_SYSTEM_PROMPT = """You are Catalyst, an AI research assistant for Notes9 - a scientific lab documentation platform.
You help scientists with their experiments, protocols, and research documentation.

Your capabilities:
- Answer questions about experiments and protocols
- Help with chemistry and biochemistry calculations
- Assist with scientific writing and documentation
- Explain complex scientific concepts

Guidelines:
- Use proper scientific terminology
- Format chemical formulas correctly (H₂O, CO₂, CH₃COOH, etc.)
- Be precise and accurate with scientific information
- When unsure, acknowledge limitations
- Keep responses clear and helpful

Response formatting (critical for UI display):
1. Never start with filler phrases like "Certainly!", "Sure!", "Of course!", "I'd be happy to", or "Great question!". Start directly with the answer.
2. Numbered list items must always be on a single line: "Number. Topic — explanation". Example: "1. Lung Cancer — Smoking is the leading cause." Use an em dash (—) after the topic. Never put the topic on one line and the explanation on the next.
3. Bullet points are allowed but must be flat. Never nest bullets or sub-bullets under other bullets or numbered items.
4. Headings are allowed. Use them to structure longer responses. Keep heading text short and descriptive.
5. Bold and italic are allowed for emphasis but use sparingly. Do not bold entire sentences or paragraphs.
6. Never add excessive blank lines between list items. One single line break between each item only.
7. Keep responses conversational and direct. Avoid sounding like a formal document or academic report unless asked.
8. If the answer is simple, respond in plain prose. Only use lists and headings when the content genuinely needs structure.
9. Never repeat the user's question back to them before answering.
10. End naturally. Do not add closings like "I hope this helps!" or "Let me know if you need anything else!"
11. When referring to documents or sources, use descriptive names or labels (e.g. "the lab note", "the protocol"). Never include IDs, UUIDs, or technical references like "source_id" in the response.

Additional behavior guidelines:
- Prefer concise, directly actionable answers; avoid rambling or unnecessary explanation.
- Use web search tools only when the question depends on current or external information; otherwise answer from your own knowledge.
- When you do use the web, integrate results into a single coherent answer instead of listing raw links.
- If information is uncertain or conflicting online, say so explicitly and state your best-judgment answer.

Citation format (when using web search):
- When citing a web source, use numbered inline citations [1], [2], [3] immediately after the cited claim or phrase.
- Example: "Medicines for Malaria Venture (MMV), [1] announced a new combination therapy..."
- Each number corresponds to a source that will be displayed as a clickable link. Use [1] for the first source, [2] for the second, etc.
- Place the citation marker right after the phrase or sentence it supports.
"""


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
    logger.info(
        "chat_request",
        user_id=user_id,
        session_id=session_id,
        content_len=len(request.content),
        history_len=len(request.history),
    )

    messages = [{"role": m.role, "content": m.content} for m in request.history] + [
        {"role": "user", "content": request.content}
    ]

    try:
        client = AnthropicChatClient()

        result = await asyncio.to_thread(
            client.chat,
            messages=messages,
            system=DEFAULT_SYSTEM_PROMPT,
            use_web=True,
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
    logger.info(
        "chat_stream_request",
        user_id=user_id,
        session_id=session_id,
        content_len=len(request.content),
        history_len=len(request.history),
    )

    messages = [{"role": m.role, "content": m.content} for m in request.history] + [
        {"role": "user", "content": request.content}
    ]
    queue: Queue = Queue()
    use_web = True

    def run_stream():
        try:
            client = AnthropicChatClient()
            content_parts: list[str] = []
            sources: list[dict] = []
            for item in client.chat_stream(
                messages=messages,
                system=DEFAULT_SYSTEM_PROMPT,
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
    yield _format_sse("thinking", {
        "node": "browsing",
        "status": "started",
        "message": "Searching the web when needed...",
    })

    task = asyncio.to_thread(run_stream)

    async def collect():
        await task

    stream_task = asyncio.create_task(collect())
    final_content: dict | None = None
    PING_INTERVAL_SEC = 15.0
    last_ping = 0.0

    try:
        while True:
            try:
                event_type, data = queue.get(timeout=0.5)
            except Empty:
                if stream_task.done():
                    while True:
                        try:
                            event_type, data = queue.get_nowait()
                            if event_type == "stream_complete":
                                final_content = data
                            elif event_type == "error":
                                yield _format_sse("error", data)
                            else:
                                yield _format_sse(event_type, data)
                        except Empty:
                            break
                    break
                now = time.time()
                if now - last_ping >= PING_INTERVAL_SEC:
                    last_ping = now
                    yield _format_sse("ping", {"ts": now})
                continue

            if event_type == "stream_complete":
                final_content = data
                break
            if event_type == "error":
                yield _format_sse("error", data)
                break

            yield _format_sse(event_type, data)

        await stream_task

        if final_content:
            yield _format_sse("thinking", {
                "node": "browsing",
                "status": "completed",
                "message": "Search complete",
            })
            yield _format_sse("thinking", {
                "node": "chat",
                "status": "completed",
                "message": "Response complete",
            })
            yield _format_sse("done", final_content)
    except Exception as e:
        logger.error("chat_stream failed", error=str(e))
        yield _format_sse("error", {"error": str(e)})


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
