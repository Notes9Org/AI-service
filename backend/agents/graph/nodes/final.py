"""Final node for response formatting."""
import time
import structlog
from agents.graph.state import AgentState
from agents.graph.stream_utils import emit_stream_event
from agents.contracts.response import FinalResponse, CitationResponse
from agents.constants import (
    TOOL_SQL,
    TOOL_RAG,
    TOOL_HYBRID,
    VERDICT_PASS,
    CONFIDENCE_JUDGE_PASS_DEFAULT,
    CONFIDENCE_HYBRID_BONUS,
    CONFIDENCE_SQL_ONLY,
    CONFIDENCE_RAG_ONLY,
    CONFIDENCE_ERROR_OR_MISSING,
)
from services.trace_service import TraceService

logger = structlog.get_logger()

# Singleton trace service
_trace_service: TraceService = None


def get_trace_service() -> TraceService:
    """Get or create trace service singleton."""
    global _trace_service
    if _trace_service is None:
        _trace_service = TraceService()
    return _trace_service


def _citation_display_label(source_type: str) -> str:
    """Human-readable label for citations so UI can show this instead of UUID."""
    if not source_type or source_type == "unknown":
        return "Source"
    if source_type.lower() == "sql":
        return "Database"
    return source_type.replace("_", " ").strip().title()


def _best_is_better_than_current(
    best_summary: dict,
    best_judge: dict,
    best_tool: str,
    current_summary: dict,
    current_judge: dict,
    current_tool: str,
) -> bool:
    """True if best answer should be preferred over current (e.g. hybrid vs RAG-only retry)."""
    if not best_summary or not current_summary:
        return bool(best_summary and not current_summary)
    best_pass = best_judge and best_judge.get("verdict") == VERDICT_PASS
    cur_pass = current_judge and current_judge.get("verdict") == VERDICT_PASS
    if best_pass and not cur_pass:
        return True
    if cur_pass and not best_pass:
        return False
    if best_pass and cur_pass:
        return len(best_summary.get("answer", "")) >= len(current_summary.get("answer", ""))
    # Both failed or current has no judge (skip-judge path): prefer current (latest) over a failed best
    if not best_pass and not cur_pass and not current_judge:
        return False
    best_prefer = best_tool in (TOOL_SQL, TOOL_HYBRID)
    cur_prefer = current_tool in (TOOL_SQL, TOOL_HYBRID)
    if best_prefer and not cur_prefer:
        return True
    if cur_prefer and not best_prefer:
        return False
    return len(best_summary.get("answer", "")) >= len(current_summary.get("answer", ""))


def final_node(state: AgentState) -> AgentState:
    """Format final response with answer, citations, confidence, and debug trace.
    Uses best_summary/best_judge_result when current summary was cleared after retries, or when best is better than current (e.g. hybrid vs RAG-only retry)."""
    emit_stream_event(state, "thinking", {"node": "final", "status": "started", "message": "Formatting response..."})
    start_time = time.time()
    summary = state.get("summary")
    judge = state.get("judge_result")
    router = state.get("router_decision")
    best_summary = state.get("best_summary")
    best_judge = state.get("best_judge_result")
    best_tool = state.get("best_tool_used") or TOOL_RAG
    # Derive current tool from router for comparison
    current_tool = TOOL_RAG
    if router:
        tools = router.get("tools", []) if isinstance(router, dict) else getattr(router, "tools", [])
        if isinstance(tools, list):
            if TOOL_SQL in tools and TOOL_RAG in tools:
                current_tool = TOOL_HYBRID
            elif TOOL_SQL in tools:
                current_tool = TOOL_SQL
            elif TOOL_RAG in tools:
                current_tool = TOOL_RAG
    # Prefer best when: no current summary (max retries path), or best is strictly better (e.g. hybrid vs RAG-only retry)
    if best_summary and (
        summary is None
        or _best_is_better_than_current(
            best_summary, best_judge or {}, best_tool,
            summary, judge or {}, current_tool,
        )
    ):
        summary = best_summary
        judge = best_judge
    elif summary is None and best_summary:
        summary = best_summary
        judge = best_judge
    retry_count = state.get("retry_count", 0)
    request = state["request"]
    trace = state.get("trace", [])
    run_id = state.get("run_id")
    trace_service = get_trace_service()
    
    logger.info(
        "final_node started",
        agent_node="final",
        run_id=run_id,
        has_summary=summary is not None,
        has_judge=judge is not None,
        retry_count=retry_count,
        payload={
            "input_has_summary": summary is not None,
            "input_has_judge": judge is not None,
            "input_retry_count": retry_count
        }
    )
    
    if run_id:
        try:
            trace_service.log_event(run_id=run_id, node_name="final", event_type="input",
                                   payload={"has_summary": summary is not None, "has_judge": judge is not None})
        except Exception:
            pass
    
    # Preserve final_response already set (e.g. out-of-scope from normalize)
    if state.get("final_response") is not None:
        return state
    
    try:
        # tool_used: when we're outputting best_summary use best_tool_used, else derive from router
        use_best = summary is state.get("best_summary") and state.get("best_summary") is not None
        tool_used = (state.get("best_tool_used") or TOOL_RAG) if use_best else TOOL_RAG
        if not use_best and router:
            if isinstance(router, dict):
                tools = router.get("tools", [])
            else:
                tools = getattr(router, "tools", [])
            if isinstance(tools, list):
                if TOOL_SQL in tools and TOOL_RAG in tools:
                    tool_used = TOOL_HYBRID
                elif TOOL_SQL in tools:
                    tool_used = TOOL_SQL
                elif TOOL_RAG in tools:
                    tool_used = TOOL_RAG
        
        if not summary:
            answer = "Unable to generate answer. Please try rephrasing your query."
            citations = []
            confidence = CONFIDENCE_ERROR_OR_MISSING
        else:
            answer = summary.get("answer", "")
            citations = []
            for cit in summary.get("citations", []):
                source_type = cit.get("source_type") or "unknown"
                if not isinstance(source_type, str):
                    source_type = "unknown"
                raw_relevance = float(cit.get("relevance", 0.0))
                relevance = max(0.0, min(1.0, raw_relevance))
                display_label = cit.get("display_label") or _citation_display_label(source_type)
                citations.append(CitationResponse(
                    display_label=display_label,
                    source_type=source_type,
                    source_name=cit.get("source_name"),
                    relevance=relevance,
                    excerpt=cit.get("excerpt")
                ))
            if not citations and state.get("run_citations"):
                seen = set()
                for cit in state.get("run_citations") or []:
                    if not isinstance(cit, dict):
                        continue
                    st = (cit.get("source_type") or "unknown")
                    sid = str(cit.get("source_id") or "")
                    key = (st, sid)
                    if key in seen:
                        continue
                    seen.add(key)
                    raw_relevance = float(cit.get("relevance", 0.0))
                    relevance = max(0.0, min(1.0, raw_relevance))
                    citations.append(CitationResponse(
                        display_label=_citation_display_label(st),
                        source_type=st if isinstance(st, str) else "unknown",
                        source_name=None,
                        relevance=relevance,
                        excerpt=cit.get("excerpt")
                    ))
            
            if judge and judge.get("verdict") == VERDICT_PASS:
                confidence = judge.get("confidence", CONFIDENCE_JUDGE_PASS_DEFAULT)
                if tool_used == TOOL_HYBRID:
                    confidence = min(confidence + CONFIDENCE_HYBRID_BONUS, 1.0)
            elif tool_used == TOOL_SQL:
                confidence = CONFIDENCE_SQL_ONLY
            else:
                confidence = CONFIDENCE_RAG_ONLY
        
        debug = None
        options = request.get("options", {}) if isinstance(request, dict) else getattr(request, "options", {}) if hasattr(request, "options") else {}
        if (isinstance(options, dict) and options.get("debug")) or (hasattr(options, "debug") and getattr(options, "debug", False)):
            router_tools = []
            if router:
                router_tools = router.get("tools", []) if isinstance(router, dict) else getattr(router, "tools", [])
            judge_verdict = judge.get("verdict") if judge and isinstance(judge, dict) else getattr(judge, "verdict", None) if judge else None
            debug = {
                "trace": trace,
                "retry_count": retry_count,
                "router_decision": {"tools": router_tools},
                "judge_verdict": judge_verdict,
                "run_process_log": state.get("run_process_log") or [],
                "run_citations_count": len(state.get("run_citations") or []),
            }
        
        final_response = FinalResponse(
            answer=answer, citations=citations, confidence=confidence, tool_used=tool_used, debug=debug
        )
        
        latency_ms = int((time.time() - start_time) * 1000)
        logger.info("final_node completed", agent_node="final", run_id=run_id,
                   answer_length=len(answer), confidence=confidence, tool_used=tool_used,
                   payload={"input_has_summary": summary is not None, "input_has_judge": judge is not None,
                           "output_answer_length": len(answer), "output_confidence": confidence,
                           "output_tool_used": tool_used, "output_citations_count": len(citations)})
        state["final_response"] = final_response
        
        if run_id:
            try:
                trace_service.log_event(run_id=run_id, node_name="final", event_type="output",
                                      payload={"answer_length": len(answer), "confidence": confidence},
                                      latency_ms=latency_ms)
            except Exception:
                pass
        
        return state
        
    except Exception as e:
        logger.error("final_node failed", run_id=run_id, error=str(e))
        
        if run_id:
            try:
                trace_service.log_event(run_id=run_id, node_name="final", event_type="error",
                                      payload={"error": str(e)})
            except Exception:
                pass
        
        state["final_response"] = FinalResponse(
            answer=f"Error formatting response: {str(e)}", citations=[], confidence=CONFIDENCE_ERROR_OR_MISSING, tool_used=TOOL_RAG
        )
        return state