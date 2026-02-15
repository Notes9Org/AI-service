"""API routes for agent."""
import time
from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException
import structlog

from agents.contracts.request import AgentRequest
from agents.contracts.response import FinalResponse
from agents.graph.state import AgentState
from agents.graph.build_graph import build_agent_graph
from services.auth import CurrentUser, get_current_user
from services.trace_service import TraceService

logger = structlog.get_logger()

router = APIRouter(prefix="/agent", tags=["agent"])

# Singleton graph (compiled once)
_agent_graph = None


def get_agent_graph():
    """Get or create agent graph singleton."""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph


@router.post("/run", response_model=FinalResponse)
async def run_agent(
    request: AgentRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> FinalResponse:
    """Execute agent graph: normalize → router → tools → summarizer → judge → final."""
    start_time = time.time()
    run_id = str(uuid4())
    trace_service = TraceService()
    # user_id is required: from request or auth; when provided in request must match authenticated user
    user_id = request.user_id or current_user.user_id
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required (from auth or request)")
    if request.user_id and request.user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="user_id cannot differ from authenticated user")
    session_id = request.session_id
    logger.info("agent_run started", run_id=run_id, query=request.query[:100], user_id=user_id, session_id=session_id)

    try:
        try:
            trace_service.create_run(
                run_id=run_id,
                organization_id=None,
                created_by=user_id,
                session_id=request.session_id,
                query=request.query,
                project_id=None
            )
        except Exception as e:
            logger.warning("Trace logging failed, continuing", error=str(e))
        
        initial_state: AgentState = {
            "run_id": run_id,
            "request": {
                "query": request.query,
                "user_id": user_id,
                "session_id": session_id,
                "scope": {},
                "history": [msg.model_dump() if hasattr(msg, "model_dump") else msg for msg in request.history],
                "options": request.options or {}
            },
            "normalized_query": None,
            "router_decision": None,
            "sql_result": None,
            "rag_result": None,
            "summary": None,
            "judge_result": None,
            "retry_count": 0,
            "final_response": None,
            "trace": []
        }
        
        graph = get_agent_graph()
        final_state = graph.invoke(initial_state)
        final_response = final_state.get("final_response")
        
        if not final_response:
            final_response = FinalResponse(
                answer="Agent execution completed but no response generated.",
                citations=[],
                confidence=0.0,
                tool_used="rag"
            )
        
        total_latency_ms = int((time.time() - start_time) * 1000)
        trace_service.update_run_status(
            run_id=run_id,
            status="completed",
            final_confidence=final_response.confidence,
            tool_used=final_response.tool_used,
            total_latency_ms=total_latency_ms
        )
        
        logger.info("agent_run completed", run_id=run_id, confidence=final_response.confidence)
        return final_response
        
    except HTTPException:
        trace_service.update_run_status(run_id=run_id, status="failed", total_latency_ms=int((time.time() - start_time) * 1000))
        raise
    except Exception as e:
        trace_service.update_run_status(run_id=run_id, status="failed", total_latency_ms=int((time.time() - start_time) * 1000))
        logger.error("agent_run failed", run_id=run_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")