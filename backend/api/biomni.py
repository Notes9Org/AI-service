"""
BioMni biomedical AI agent endpoint.

POST /biomni/query  – run a biomedical query through the BioMni agent.
GET  /biomni/health – lightweight check that the agent can be initialised.
"""

import asyncio
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/biomni", tags=["biomni"])

_agent_instance = None


def _get_agent():
    """Return a lazily-initialised singleton BioMniAgent."""
    global _agent_instance
    if _agent_instance is None:
        from biomni_runner.agent import BioMniAgent
        _agent_instance = BioMniAgent()
    return _agent_instance


# ── Request / Response models ────────────────────────────────────────────────

class BioMniQueryRequest(BaseModel):
    """Request body for POST /biomni/query."""
    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language biomedical question or task.",
        json_schema_extra={"example": "Predict ADMET properties for aspirin"},
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Max retries on Bedrock throttling errors.",
    )


class StepEntry(BaseModel):
    """One step in the agent's reasoning / execution log."""
    content: str


class BioMniQueryResponse(BaseModel):
    """Response from POST /biomni/query."""
    answer: str = Field(..., description="Final answer produced by the BioMni agent.")
    steps: List[StepEntry] = Field(
        default_factory=list,
        description="Intermediate reasoning and execution steps.",
    )


class BioMniHealthResponse(BaseModel):
    """Response from GET /biomni/health."""
    status: str
    data_path: Optional[str] = None
    llm_model: Optional[str] = None


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=BioMniQueryResponse,
    summary="Run a biomedical query through BioMni",
    description=(
        "Sends a natural-language biomedical question to the BioMni A1 agent "
        "(Stanford). The agent creates a plan, executes Python/R/Bash code "
        "against the local data lake, and returns a final answer together with "
        "intermediate steps."
    ),
)
async def biomni_query(request: BioMniQueryRequest) -> BioMniQueryResponse:
    try:
        agent = _get_agent()
        log, answer = await asyncio.to_thread(
            agent.go, request.query, request.max_retries
        )
        steps = [StepEntry(content=str(entry)) for entry in (log or [])]
        return BioMniQueryResponse(answer=answer, steps=steps)
    except Exception as exc:
        logger.exception("BioMni query failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/health",
    response_model=BioMniHealthResponse,
    summary="BioMni agent health check",
)
async def biomni_health() -> BioMniHealthResponse:
    try:
        from biomni_runner.config import get_data_path, get_llm_model
        return BioMniHealthResponse(
            status="ok",
            data_path=get_data_path(),
            llm_model=get_llm_model(),
        )
    except Exception as exc:
        return BioMniHealthResponse(status=f"error: {exc}")
