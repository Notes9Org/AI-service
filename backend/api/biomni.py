"""Biomni biomedical agent API endpoints."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import structlog

from biomni_svc import run_biomni_task
from biomni_svc.storage import upload_biomni_result_to_s3
from services.auth import CurrentUser, get_current_user

logger = structlog.get_logger()
router = APIRouter(prefix="/biomni", tags=["biomni"])


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
