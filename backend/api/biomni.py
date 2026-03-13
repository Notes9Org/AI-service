"""
BioMni biomedical AI agent endpoints.

POST /biomni/query  – run a biomedical query through the BioMni agent (biomni_runner).
GET  /biomni/health – lightweight check that the agent can be initialised.
POST /biomni/run   – run a biomedical task via biomni_svc (auth, S3 upload).
"""

import asyncio
import logging
from typing import List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

logger = structlog.get_logger()
router = APIRouter(prefix="/biomni", tags=["biomni"])

# Optional biomni_svc / auth (main branch)
try:
    from biomni_svc import run_biomni_task
    from biomni_svc.storage import upload_biomni_result_to_s3
    from services.auth import CurrentUser, get_current_user
    _HAS_BIOMNI_SVC = True
except ImportError:
    _HAS_BIOMNI_SVC = False

# ── biomni_runner (query / health) ───────────────────────────────────────────

_agent_instance = None


def _get_agent():
    """Return a lazily-initialised singleton BioMniAgent."""
    global _agent_instance
    if _agent_instance is None:
        from biomni_runner.agent import BioMniAgent
        _agent_instance = BioMniAgent()
    return _agent_instance


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


# ── biomni_svc (run with auth / S3) ───────────────────────────────────────────

class BiomniRunRequest(BaseModel):
    """Request schema for POST /biomni/run."""
    prompt: str = Field(..., description="Natural language prompt for the biomedical task")
    session_id: Optional[str] = Field(None, description="Session ID for tracking and context")


class BiomniRunResponse(BaseModel):
    """Response schema for POST /biomni/run."""
    result: str = Field(..., description="Biomni agent output text")
    success: bool = Field(..., description="Whether the task completed successfully")
    error: Optional[str] = Field(None, description="Error message if success is False")
    artifact_url: Optional[str] = Field(None, description="S3 URL of persisted result when BIOMNI_S3_BUCKET is set")


# ── Endpoints ─────────────────────────────────────────────────────────────────

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
        logging.getLogger(__name__).exception("BioMni query failed")
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


if _HAS_BIOMNI_SVC:

    @router.post("/run", response_model=BiomniRunResponse, summary="Run Biomni biomedical task")
    async def biomni_run(
        request: BiomniRunRequest,
        current_user: CurrentUser = Depends(get_current_user),
    ) -> BiomniRunResponse:
        """
        Execute a biomedical research task via Biomni agent.
        Requires Bearer token (Supabase Auth).
        """
        user_id = current_user.user_id
        session_id = request.session_id or ""

        try:
            outcome = run_biomni_task(
                query=request.prompt,
                user_id=user_id,
                session_id=session_id,
            )
            artifact_url = None
            if outcome.get("success") and outcome.get("result"):
                artifact_url = upload_biomni_result_to_s3(
                    result=outcome["result"],
                    session_id=session_id,
                    user_id=user_id,
                    metadata={"prompt": request.prompt[:500]},
                )
            return BiomniRunResponse(
                result=outcome.get("result", ""),
                success=outcome.get("success", False),
                error=outcome.get("error"),
                artifact_url=artifact_url,
            )
        except RuntimeError as e:
            if "not installed" in str(e).lower():
                raise HTTPException(
                    status_code=503,
                    detail="Biomni service unavailable: package not installed. Run: pip install biomni langchain-aws",
                ) from e
            raise HTTPException(status_code=500, detail=str(e)) from e
        except Exception as e:
            logger.error("biomni_run_unexpected_error", error=str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=f"Biomni task failed: {str(e)}") from e
else:

    @router.post("/run", response_model=BiomniRunResponse, summary="Run Biomni biomedical task")
    async def biomni_run(request: BiomniRunRequest) -> BiomniRunResponse:
        raise HTTPException(
            status_code=503,
            detail="Biomni service unavailable: package not installed. Run: pip install biomni langchain-aws",
        )
