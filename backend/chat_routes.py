"""
Claude (Bedrock) / Azure OpenAI chat endpoint.

POST /chat: send message history and optional system prompt; receive assistant reply.
Requires Bearer token (Supabase Auth).
"""
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from agents.services.llm_client import LLMClient, LLMError
from services.auth import CurrentUser, get_current_user

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatMessageRequest(BaseModel):
    """Single message in the conversation."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request body for POST /chat."""
    messages: List[ChatMessageRequest] = Field(..., description="Conversation history (user and assistant turns)")
    system: Optional[str] = Field(None, description="Optional system prompt")
    model: Optional[str] = Field(None, description="Override default chat model")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")


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
    **Chat endpoint.** Send conversation history (messages + optional system prompt) to Claude (Bedrock) or Azure OpenAI and receive the assistant reply as JSON. Requires Bearer token.
    """
    try:
        client = LLMClient()
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        content = client.chat(
            messages=messages,
            system=request.system,
            model=request.model,
            temperature=request.temperature,
        )
        return ChatResponse(content=content, role="assistant")
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
