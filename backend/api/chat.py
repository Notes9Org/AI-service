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
- Keep responses clear and helpful"""


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
