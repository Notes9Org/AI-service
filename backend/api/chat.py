"""
Claude (AWS Bedrock) chat endpoint.

POST /chat: send user message content and optional history; receive assistant reply.
Requires Bearer token (Supabase Auth).
Follows /agent/run input schema pattern - content, session_id, history for memory management.
"""
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

import structlog

from agents.services.llm_client import LLMClient, LLMError
from services.auth import CurrentUser, get_current_user
from services.config import get_bedrock_config

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
11. When referring to documents or sources, use descriptive names or labels (e.g. "the lab note", "the protocol"). Never include IDs, UUIDs, or technical references like "source_id" in the response."""


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


class ChatResponse(BaseModel):
    """Response for POST /chat."""
    content: str = Field(..., description="Assistant reply text")
    role: str = Field(default="assistant", description="Message role")


@router.post("", response_model=ChatResponse, summary="Chat with Claude / LLM")
async def chat(
    request: ChatRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> ChatResponse:
    """
    **Chat endpoint.** Send content; receive assistant reply. System prompt, model, temperature are internal.
    """
    user_id = current_user.user_id
    session_id = request.session_id
    logger.info("chat_request", user_id=user_id, session_id=session_id, content_len=len(request.content), history_len=len(request.history))
    messages = [{"role": m.role, "content": m.content} for m in request.history] + [{"role": "user", "content": request.content}]
    # Only Bedrock is supported as provider; use its configured chat model.
    model_id = get_bedrock_config().get_chat_model_id()
    try:
        client = LLMClient()
        content = client.chat(
            messages=messages,
            system=DEFAULT_SYSTEM_PROMPT,
            model=model_id,
            temperature=0.7,
        )
        return ChatResponse(content=content, role="assistant")
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
