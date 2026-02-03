""" Normalized query contract for agent schemas."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Literal

DomainKind = Literal["lab", "general", "unknown"]
IntentKind = Literal["aggregate", "search", "hybrid", "other"]


class NormalizedQuery(BaseModel):
    """ Normalized query schema for agent execution."""
    domain: DomainKind = Field(
        ...,
        description="Domain classification: lab (in-scope), general (out-of-scope), unknown."
    )
    in_scope: bool = Field(
        ...,
        description="True if the query is about lab management; false if outside domain."
    )
    out_of_scope_reason: Optional[str] = Field(
        default=None,
        description="When in_scope is false, short reason why (e.g. 'general knowledge', 'weather')."
    )
    intent: IntentKind = Field(
        ...,
        description="Query intent: aggregate (SQL), search (RAG), hybrid (both), or other (out-of-scope)."
    )
    normalized_query: str = Field(
        ...,
        description="Cleaned and normalized query text for SQL or RAG processing."
    )
    entities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Entities extracted: dates, numbers, experiment_ids, etc."
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Conversation context and metadata."
    )
    history_summary: Optional[str] = Field(
        default=None,
        description="Optional summary of relevant conversation history."
    )

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v: str) -> str:
        """Validate domain value."""
        allowed = ["lab", "general", "unknown"]
        if v not in allowed:
            raise ValueError(f"Invalid domain: {v}. Must be one of: {allowed}")
        return v

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, v: str) -> str:
        """Validate intent value."""
        allowed_intents = ["aggregate", "search", "hybrid", "other"]
        if v not in allowed_intents:
            raise ValueError(f"Invalid intent: {v}. Must be one of: {allowed_intents}")
        return v