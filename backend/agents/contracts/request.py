""" Request contract for agent schemas."""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any


class ChatMessage(BaseModel):
    """Chat message in conversation history."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class Scope(BaseModel):
    """Access scope for the agent query."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "organization_id": "cedbb951-4b9f-440a-96ad-0373fe059a1b",
                "project_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "experiment_id": "f1e2d3c4-b5a6-9876-5432-10fedcba9876"
            }
        }
    )
    
    organization_id: Optional[str] = Field(
        None,
        description="Organization ID (UUID) - Optional, filters results to specific organization"
    )
    project_id: Optional[str] = Field(
        None,
        description="Project ID (UUID) - Optional, filters results to specific project"
    )
    experiment_id: Optional[str] = Field(
        None,
        description="Experiment ID (UUID) - Optional, filters results to specific experiment"
    )


class AgentRequest(BaseModel):
    """Request schema for agent execution - query, session_id, history for memory; user_id required (from auth or request)."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "How many experiments were completed last month?",
                "session_id": "session-456",
                "user_id": "user-uuid-from-auth",
                "history": [
                    {"role": "user", "content": "Show me experiment stats"},
                    {"role": "assistant", "content": "Here are the stats..."}
                ]
            }
        }
    )
    query: str = Field(..., description="User query to process")
    session_id: str = Field(..., description="Session ID for tracking and context")
    user_id: Optional[str] = Field(
        default=None,
        description="User ID for filtering data (required - from auth or request; when provided must match authenticated user)"
    )
    history: List[ChatMessage] = Field(
        default_factory=list,
        description="Previous messages in the conversation for memory/context"
    )
    scope: Optional[Scope] = Field(
        default=None,
        description="Access scope - optional, not used for filtering (deprecated)"
    )
    options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Options: debug (bool), max_retries (int), etc."
    )