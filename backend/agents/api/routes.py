"""API routes for agent."""
import asyncio
import json
import time
from queue import Empty, Queue
from uuid import uuid4
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, Depends, HTTPException
import structlog

from agents.contracts.request import AgentRequest
from agents.contracts.response import FinalResponse
from agents.graph.state import AgentState
from agents.graph.build_graph import build_agent_graph
from services.auth import CurrentUser, get_current_user
from services.trace_service import TraceService

logger = structlog.get_logger()

router = APIRouter(prefix="/notes9", tags=["notes9"])

# Singleton graph (compiled once)
_agent_graph = None


def get_agent_graph():
    """Get or create agent graph singleton."""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph


def _format_sse(event_type: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def _stream_agent_generator(request: AgentRequest, current_user: CurrentUser):
    """Async generator that runs the agent graph and yields SSE events."""
    queue: Queue = Queue()
    run_id = str(uuid4())
    trace_service = TraceService()
    user_id = current_user.user_id

    def stream_callback(event_type: str, data: dict):
        queue.put((event_type, data))

    initial_state: AgentState = {
        "run_id": run_id,
        "request": {
            "query": request.query,
            "user_id": user_id,
            "session_id": request.session_id,
            "scope": {},
            "history": [msg.model_dump() if hasattr(msg, "model_dump") else msg for msg in request.history],
            "options": request.options or {}
        },
        "normalized_query": None,
        "router_decision": None,
        "sql_result": None,
        "rag_result": None,
        "sql_runs": [],
        "rag_chunks_all": [],
        "summary": None,
        "judge_result": None,
        "retry_count": 0,
        "attempted_tools": [],
        "flags": None,
        "retry_context": None,
        "best_summary": None,
        "best_judge_result": None,
        "best_tool_used": None,
        "final_response": None,
        "run_process_log": [],
        "run_citations": [],
        "trace": [],
        "stream_callback": stream_callback,
    }

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

    final_state_ref = {"value": None}

    def run_graph():
        graph = get_agent_graph()
        final_state = graph.invoke(initial_state)
        final_state_ref["value"] = final_state
        queue.put(("done", None))

    task = asyncio.to_thread(run_graph)

    async def run_and_collect():
        await task

    graph_task = asyncio.create_task(run_and_collect())

    # Ping only when idle this long (keep-alive); avoid pings every 100ms
    PING_INTERVAL_SEC = 15.0
    last_ping = 0.0

    try:
        while True:
            try:
                event_type, data = queue.get(timeout=0.5)
            except Empty:
                if graph_task.done():
                    break
                now = time.time()
                if now - last_ping >= PING_INTERVAL_SEC:
                    last_ping = now
                    yield _format_sse("ping", {"ts": now})
                continue

            if event_type == "done":
                break

            if event_type == "thinking":
                yield _format_sse("thinking", data)
            elif event_type == "token":
                yield _format_sse("token", data)
            elif event_type == "error":
                yield _format_sse("error", data)

        await graph_task

        final_state = final_state_ref["value"]
        final_response = final_state.get("final_response") if final_state else None

        if not final_response:
            from agents.constants import TOOL_RAG, CONFIDENCE_ERROR_OR_MISSING
            final_response = FinalResponse(
                answer="Agent execution completed but no response generated.",
                citations=[],
                confidence=CONFIDENCE_ERROR_OR_MISSING,
                tool_used=TOOL_RAG
            )

        trace_service.update_run_status(
            run_id=run_id,
            status="completed",
            final_confidence=final_response.confidence,
            tool_used=final_response.tool_used,
        )

        yield _format_sse("done", final_response.model_dump())
    except Exception as e:
        logger.error("agent_stream failed", run_id=run_id, error=str(e))
        yield _format_sse("error", {"error": str(e)})


@router.post("/stream")
async def stream_agent(
    request: AgentRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Execute agent with SSE streaming: thinking events, then final response. Bearer token required."""
    return StreamingResponse(
    _stream_agent_generator(request, current_user=current_user),
    media_type="text/event-stream",
    headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    },
)