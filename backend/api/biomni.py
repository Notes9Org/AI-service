"""
Biomni biomedical AI agent endpoints.

POST /biomni/run    - Run a biomedical task (auth, S3 upload, steps).
POST /biomni/stream - SSE streaming of Biomni execution.
GET  /biomni/health - Lightweight check that the agent can be initialized.
"""

from typing import List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, WebSocket
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from biomni_svc import run_biomni_task
from biomni_svc.config import get_biomni_config
from biomni_svc.pdf import generate_session_pdf, generate_and_upload_run_pdf
from biomni_svc.session import add_run, get_session, list_sessions
from biomni_svc.stream import stream_biomni_events
from biomni_svc.storage import upload_biomni_result_to_s3
from biomni_svc.ws_handler import handle_biomni_websocket
from services.auth import CurrentUser, get_current_user

logger = structlog.get_logger()
router = APIRouter(prefix="/biomni", tags=["biomni"])


class _BiomniRequestBase(BaseModel):
    """Base request schema aligned with agent/run: query, session_id, history, options. user_id always from Bearer token."""

    query: Optional[str] = Field(None, description="User query (same as agent/run). Use query or prompt.")
    prompt: Optional[str] = Field(None, description="Natural language prompt (alias for query, backwards compat)")
    user_id: Optional[str] = Field(
        None,
        description="Ignored. user_id is always from Bearer token (auth). Do not send; server uses JWT sub.",
    )
    session_id: Optional[str] = Field(None, description="Session ID for tracking and context (same as agent/run)")
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retries on Bedrock throttling")
    history: List[dict] = Field(
        default_factory=list,
        description="Previous messages: [{role, content}] (same format as agent/run)",
    )
    options: Optional[dict] = Field(
        None,
        description="skip_clarify, max_clarify_rounds, generate_pdf",
    )

    @model_validator(mode="after")
    def require_query_or_prompt(self):
        if not self.query and not self.prompt:
            raise ValueError("Either query or prompt is required")
        return self

    def get_prompt(self) -> str:
        """Resolve effective prompt (query takes precedence, aligned with agent/run)."""
        return (self.query or self.prompt or "").strip()


class BiomniRunRequest(_BiomniRequestBase):
    """Request schema for POST /biomni/run. Aligned with agent/run: query, session_id, history. user_id from Bearer."""


class BiomniStreamRequest(_BiomniRequestBase):
    """Request schema for POST /biomni/stream. Aligned with agent/run: query, session_id, history. user_id from Bearer."""


class StepEntry(BaseModel):
    """One step in the agent's reasoning / execution log."""

    content: str


class BiomniRunResponse(BaseModel):
    """Response schema for POST /biomni/run."""

    result: str = Field(..., description="Biomni agent output text")
    success: bool = Field(..., description="Whether the task completed successfully")
    error: Optional[str] = Field(None, description="Error message if success is False")
    steps: List[StepEntry] = Field(default_factory=list, description="Intermediate execution steps")
    artifact_url: Optional[str] = Field(
        None, description="S3 URL of persisted result when BIOMNI_S3_BUCKET is set"
    )
    clarify_question: Optional[str] = Field(
        None, description="When clarification needed, the question to ask the user"
    )
    clarify_options: Optional[List[str]] = Field(
        None, description="Optional suggested answers for clarification"
    )
    pdf_url: Optional[str] = Field(
        None, description="S3 URL of PDF report when generate_pdf was requested"
    )


class BioMniHealthResponse(BaseModel):
    """Response from GET /biomni/health."""

    status: str
    data_path: Optional[str] = None
    llm_model: Optional[str] = None


@router.post("/run", response_model=BiomniRunResponse, summary="Run Biomni biomedical task")
async def biomni_run(
    request: BiomniRunRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> BiomniRunResponse:
    """
    Execute a biomedical research task via Biomni agent.
    Requires Bearer token (Supabase Auth). user_id is always from auth (JWT sub), never from request body.
    """
    user_id = current_user.user_id  # Always from Bearer token; request.user_id is ignored
    session_id = request.session_id or ""
    prompt = request.get_prompt()
    opts = request.options or {}
    skip_clarify = opts.get("skip_clarify", False)
    max_clarify_rounds = opts.get("max_clarify_rounds", 2)

    try:
        # Clarification check
        if not skip_clarify and len(request.history) < max_clarify_rounds:
            from biomni_svc.clarify import evaluate_clarification

            clarify_result = await evaluate_clarification(prompt, request.history)
            if clarify_result.needs_clarification and clarify_result.question:
                return BiomniRunResponse(
                    result="",
                    success=False,
                    clarify_question=clarify_result.question,
                    clarify_options=clarify_result.options,
                )

        outcome = run_biomni_task(
            query=prompt,
            user_id=user_id,
            session_id=session_id,
            max_retries=request.max_retries,
        )
        artifact_url = None
        pdf_url = None
        if outcome.get("success") and outcome.get("result"):
            artifact_url = upload_biomni_result_to_s3(
                result=outcome["result"],
                session_id=session_id,
                user_id=user_id,
                metadata={"prompt": prompt[:500]},
            )
            if opts.get("generate_pdf"):
                pdf_url = generate_and_upload_run_pdf(
                    prompt=prompt,
                    result=outcome["result"],
                    steps=outcome.get("steps", []),
                    session_id=session_id,
                    user_id=user_id,
                )
        steps = [StepEntry(content=s) for s in outcome.get("steps", [])]
        # Persist to session if session_id provided
        if session_id and outcome.get("success"):
            add_run(
                session_id=session_id,
                user_id=user_id,
                prompt=prompt,
                result=outcome.get("result", ""),
                steps=outcome.get("steps", []),
            )
        return BiomniRunResponse(
            result=outcome.get("result", ""),
            success=outcome.get("success", False),
            error=outcome.get("error"),
            steps=steps,
            artifact_url=artifact_url,
            pdf_url=pdf_url,
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


@router.post("/stream", summary="Stream Biomni biomedical task (SSE)")
async def biomni_stream(
    request: BiomniStreamRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Execute a biomedical task via Biomni with SSE streaming.
    Events: started, step, result, error, ping, done.
    Requires Bearer token (Supabase Auth). user_id is always from auth, never from request body.
    """
    opts = request.options or {}
    try:
        return StreamingResponse(
            stream_biomni_events(
                prompt=request.get_prompt(),
                user_id=current_user.user_id,
                session_id=request.session_id or "",
                max_retries=request.max_retries,
                history=request.history,
                skip_clarify=opts.get("skip_clarify", False),
                max_clarify_rounds=opts.get("max_clarify_rounds", 2),
                generate_pdf=opts.get("generate_pdf", False),
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except RuntimeError as e:
        if "not installed" in str(e).lower():
            raise HTTPException(
                status_code=503,
                detail="Biomni service unavailable: package not installed. Run: pip install biomni langchain-aws",
            ) from e
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.websocket("/ws")
async def biomni_websocket(websocket: WebSocket, token: str | None = None):
    """
    WebSocket endpoint for Biomni streaming.
    Connect with ?token=JWT or send {\"type\": \"auth\", \"token\": \"...\"} as first message.
    Send {\"type\": \"run\", \"prompt\": \"...\", \"session_id\": \"...\"} to execute.
    """
    await handle_biomni_websocket(websocket, token=token)


@router.get("/mcp/servers", summary="List MCP servers and tools")
async def biomni_mcp_servers(
    current_user: CurrentUser = Depends(get_current_user),
):
    """List registered MCP servers and their tools. Requires auth."""
    from biomni_svc.mcp import list_mcp_servers

    servers = list_mcp_servers()
    return {"servers": servers}


@router.get("/sessions", summary="List Biomni sessions")
async def biomni_list_sessions(
    current_user: CurrentUser = Depends(get_current_user),
):
    """List sessions for the current user."""
    sessions = list_sessions(current_user.user_id)
    return {"sessions": sessions}


@router.get("/sessions/{session_id}", summary="Get Biomni session")
async def biomni_get_session(
    session_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Get a session with all runs."""
    session = get_session(session_id, current_user.user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/sessions/{session_id}/pdf", summary="Download session PDF")
async def biomni_session_pdf(
    session_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Generate and return session PDF report."""
    session = get_session(session_id, current_user.user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    runs = session.get("runs", [])
    pdf_bytes = generate_session_pdf(session_id, runs)
    if not pdf_bytes:
        raise HTTPException(status_code=500, detail="PDF generation failed")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="biomni-session-{session_id}.pdf"'},
    )


@router.get("/mcp/health", summary="Test MCP connections")
async def biomni_mcp_health(
    current_user: CurrentUser = Depends(get_current_user),
):
    """Test MCP server connections. Requires auth."""
    from biomni_svc.mcp import test_mcp_connection

    return test_mcp_connection()


@router.get("/health", response_model=BioMniHealthResponse, summary="Biomni agent health check")
async def biomni_health() -> BioMniHealthResponse:
    """Check that Biomni agent can be initialized."""
    try:
        cfg = get_biomni_config()
        return BioMniHealthResponse(
            status="ok",
            data_path=cfg.path,
            llm_model=cfg.llm,
        )
    except Exception as exc:
        return BioMniHealthResponse(status=f"error: {exc}")
