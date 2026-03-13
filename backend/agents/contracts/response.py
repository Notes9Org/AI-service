"""Response schemas for agent API."""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional, Literal


class CitationResponse(BaseModel):
    """Citation as returned in the API. Only display-safe fields; source_id and chunk_id are omitted and stored server-side for reference."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "display_label": "Lab note: PCR Protocol",
                "source_type": "lab_note",
                "source_name": "PCR Protocol",
                "relevance": 0.95,
                "excerpt": "Relevant text excerpt..."
            }
        }
    )
    display_label: Optional[str] = Field(None, description="Human-readable label for display (e.g. 'Lab note: Cross reference').")
    source_type: str = Field(..., description="Source type: lab_note, protocol, etc.")
    source_name: Optional[str] = Field(None, description="Document name (e.g. lab note title).")
    relevance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score"
    )
    excerpt: Optional[str] = Field(None, description="Relevant excerpt from source")


class Citation(BaseModel):
    """Internal citation with full fields. source_id and chunk_id are not sent in the API response; use CitationResponse for the response."""
    display_label: Optional[str] = Field(None, description="Human-readable label for UI.")
    source_type: str = Field(..., description="Source type.")
    source_id: str = Field(..., description="Source ID (UUID). Not included in API response; stored for reference.")
    source_name: Optional[str] = Field(None, description="Human-readable name.")
    chunk_id: Optional[str] = Field(None, description="Chunk ID. Not included in API response; stored for reference.")
    relevance: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    excerpt: Optional[str] = Field(None, description="Relevant excerpt from source")


class FinalResponse(BaseModel):
    """Final response from agent."""
    model_config = ConfigDict(
        json_schema_serialization_defaults_required=True
    )
    
    answer: str = Field(..., description="Generated answer")
    citations: List[CitationResponse] = Field(
        default_factory=list,
        description="Source citations (display-safe only; no source_id or chunk_id)"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the answer"
    )
    tool_used: Literal["sql", "rag", "hybrid", "none"] = Field(
        ...,
        description="Tool(s) used to generate answer; 'none' for out-of-scope responses."
    )
    debug: Optional[Dict[str, Any]] = Field(
        None,
        description="Debug trace (node outputs, latency) if debug=true"
    )