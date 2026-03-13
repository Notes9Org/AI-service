"""Router node for tool selection."""
import time
import structlog
from agents.graph.state import AgentState
from agents.graph.stream_utils import emit_stream_event
from agents.contracts.router import RouterDecision
from agents.contracts.response import FinalResponse
from agents.constants import (
    TOOL_SQL,
    TOOL_RAG,
    TOOL_HYBRID,
    ROUTE_IN_SCOPE,
    ROUTE_OUT_OF_SCOPE,
    CONFIDENCE_ROUTER_FALLBACK,
    CONFIDENCE_ROUTER_OUT_OF_SCOPE,
    CONFIDENCE_RAG_ONLY,
)
from services.trace_service import TraceService
from agents.services.thinking_logger import get_thinking_logger

# Reused when router defensively marks out-of-scope (normalize usually handles this)
DEFAULT_OUT_OF_SCOPE_MESSAGE = (
    "This question isn't related to lab management. "
    "I can help with experiments, projects, samples, protocols, and lab notes."
)

logger = structlog.get_logger()

_trace_service: TraceService = None


def get_trace_service() -> TraceService:
    """Get or create trace service singleton."""
    global _trace_service
    if _trace_service is None:
        _trace_service = TraceService()
    return _trace_service


def router_node(state: AgentState) -> AgentState:
    """Route to tools based on intent."""
    emit_stream_event(state, "thinking", {"node": "router", "status": "started", "message": "Choosing data sources..."})
    start_time = time.time()
    normalized = state.get("normalized_query")
    request = state.get("request")
    run_id = state.get("run_id")
    trace_service = get_trace_service()

    if not normalized:
        logger.error("No normalized query found in state.", run_id=run_id, request=request)

        decision = RouterDecision(
            tools=[TOOL_RAG],
            route=ROUTE_IN_SCOPE,
            confidence=CONFIDENCE_ROUTER_FALLBACK,
            reasoning="Fallback: normalized query missing.",
            constraints={}
        )
        state["router_decision"] = decision
        # Stream thinking for UI: decision and rationale while loading
        emit_stream_event(state, "thinking", {
            "node": "router",
            "status": "completed",
            "message": f"Route to {', '.join(decision.tools)}",
            "decision": f"Route to {', '.join(decision.tools)}",
            "rationale": decision.reasoning,
            "confidence": decision.confidence,
        })
        # Log error event
        if run_id:
            try:
                trace_service.log_event(
                    run_id=run_id,
                    node_name="router",
                    event_type="error",
                    payload={"error": "normalized query missing"}
                )
            except Exception:
                pass
        
        return state
    
    normalized_query_text = normalized.normalized_query[:100] if normalized and normalized.normalized_query else "None"
    logger.info(
        "Router node started",
        agent_node="router",
        run_id=run_id,
        normalized_query=normalized_query_text,
        payload={
            "input_intent": normalized.intent if normalized else None,
            "input_normalized_query": normalized.normalized_query if normalized else None
        }
    )

    # Log input event
    if run_id:
        try:
            trace_service.log_event(
                run_id=run_id,
                node_name="router",
                event_type="input",
                payload={"intent": normalized.intent}
            )
        except Exception:
            pass

    # Defensive: if not in scope, set final_response and let graph route to final (no extra node)
    if not getattr(normalized, "in_scope", True):
        reason = getattr(normalized, "out_of_scope_reason", None) or "Query not in domain."
        decision = RouterDecision(
            tools=[],
            route=ROUTE_OUT_OF_SCOPE,
            confidence=CONFIDENCE_ROUTER_OUT_OF_SCOPE,
            reasoning=reason,
            constraints={},
        )
        state["router_decision"] = decision
        state["final_response"] = FinalResponse(
            answer=DEFAULT_OUT_OF_SCOPE_MESSAGE,
            citations=[],
            confidence=0.0,
            tool_used="none",
        )
        latency_ms = int((time.time() - start_time) * 1000)
        logger.info("router_node completed (out_of_scope)", agent_node="router", run_id=run_id,
                   route="out_of_scope", reasoning=reason[:100], latency_ms=round(latency_ms, 2),
                   payload={"input_intent": normalized.intent if normalized else None,
                            "input_normalized_query": normalized.normalized_query[:200] if normalized else None,
                            "output_route": "out_of_scope", "output_confidence": CONFIDENCE_ROUTER_OUT_OF_SCOPE})
        if run_id:
            try:
                trace_service.log_event(
                    run_id=run_id, node_name="router", event_type="output",
                    payload={"route": ROUTE_OUT_OF_SCOPE, "confidence": CONFIDENCE_ROUTER_OUT_OF_SCOPE}, latency_ms=latency_ms
                )
            except Exception:
                pass
        return state

    # Strategy-aware retry: use retry_context to force different tool and avoid ping-pong
    attempted_tools = state.get("attempted_tools") or []
    retry_ctx = state.get("retry_context")
    if retry_ctx:
        failure = retry_ctx.get("failure_reason") or {}
        rewrite_hint = (retry_ctx.get("rewrite_hint") or "").strip()
        attempted = retry_ctx.get("attempted_tools") or attempted_tools

        if failure.get("sql_empty") and TOOL_RAG not in attempted:
            tools = [TOOL_RAG]
            confidence = 0.85
            reasoning = "Retry: SQL returned no rows, trying RAG."
            logger.info("router: retry strategy sql_empty → RAG", run_id=run_id)
        elif failure.get("rag_weak") and TOOL_SQL not in attempted:
            tools = [TOOL_SQL]
            confidence = 0.85
            reasoning = "Retry: RAG retrieval weak, trying SQL."
            logger.info("router: retry strategy rag_weak → SQL", run_id=run_id)
        elif failure.get("wrong_intent"):
            if TOOL_SQL in attempted and TOOL_RAG not in attempted:
                tools = [TOOL_RAG]
            elif TOOL_RAG in attempted and TOOL_SQL not in attempted:
                tools = [TOOL_SQL]
            else:
                tools = [TOOL_SQL, TOOL_RAG]
            confidence = 0.7
            reasoning = "Retry: wrong intent, trying different tool(s)."
            logger.info("router: retry strategy wrong_intent", run_id=run_id, tools=tools)
        elif failure.get("entities_missing") and rewrite_hint:
            # Rewrite is done in retry node before looping back; route to RAG with fresh attempt
            tools = [TOOL_RAG]
            decision = RouterDecision(
                tools=tools,
                route=ROUTE_IN_SCOPE,
                confidence=0.6,
                reasoning="Retry: entities missing, rewritten query → RAG.",
                constraints={},
            )
            state["router_decision"] = decision
            _lat = int((time.time() - start_time) * 1000)
            logger.info("router_node completed", agent_node="router", run_id=run_id, tools=tools,
                       confidence=0.6, reasoning="Retry: entities missing, rewritten query → RAG.",
                       latency_ms=round(_lat, 2),
                       payload={"input_intent": normalized.intent if normalized else None,
                                "input_normalized_query": normalized.normalized_query[:200] if normalized else None,
                                "output_tools": tools, "output_confidence": 0.6, "output_reasoning": "Retry: entities missing → RAG."})
            if run_id:
                try:
                    trace_service.log_event(
                        run_id=run_id, node_name="router", event_type="output",
                        payload={"tools": tools, "retry_strategy": "entities_missing_rewrite"}, latency_ms=_lat
                    )
                except Exception:
                    pass
            return state
        elif set(attempted) >= {TOOL_SQL, TOOL_RAG} and rewrite_hint:
            tools = [TOOL_RAG]
            decision = RouterDecision(
                tools=tools,
                route=ROUTE_IN_SCOPE,
                confidence=0.5,
                reasoning="Retry: both tools tried, rewritten query → RAG.",
                constraints={},
            )
            state["router_decision"] = decision
            _lat = int((time.time() - start_time) * 1000)
            logger.info("router_node completed", agent_node="router", run_id=run_id, tools=tools,
                       confidence=0.5, reasoning="Retry: both tools tried, rewritten query → RAG.",
                       latency_ms=round(_lat, 2),
                       payload={"input_intent": normalized.intent if normalized else None,
                                "input_normalized_query": normalized.normalized_query[:200] if normalized else None,
                                "output_tools": tools, "output_confidence": 0.5, "output_reasoning": "Retry: both tried → RAG."})
            if run_id:
                try:
                    trace_service.log_event(
                        run_id=run_id, node_name="router", event_type="output",
                        payload={"tools": tools, "retry_strategy": "both_tried_rewrite"}, latency_ms=_lat
                    )
                except Exception:
                    pass
            return state
        elif set(attempted) >= {TOOL_SQL, TOOL_RAG}:
            # Both tools tried: go to summarizer with accumulated context (no separate ask_clarifying node)
            state["router_decision"] = RouterDecision(
                tools=[],
                route=ROUTE_IN_SCOPE,
                confidence=0.0,
                reasoning="Retry: both tools tried, synthesizing from existing context.",
                constraints={},
            )
            _lat = int((time.time() - start_time) * 1000)
            logger.info("router_node completed", agent_node="router", run_id=run_id, tools=[],
                       confidence=0.0, reasoning="Retry: both tools tried → summarizer.",
                       latency_ms=round(_lat, 2),
                       payload={"input_intent": normalized.intent if normalized else None,
                                "input_normalized_query": normalized.normalized_query[:200] if normalized else None,
                                "output_tools": [], "output_confidence": 0.0})
            return state
        else:
            tools = [TOOL_SQL] if TOOL_SQL not in attempted else [TOOL_RAG]
            confidence = 0.75
            reasoning = f"Retry: trying {tools} (attempted: {attempted})."

        decision = RouterDecision(
            tools=tools,
            route=ROUTE_IN_SCOPE,
            confidence=confidence,
            reasoning=reasoning,
            constraints={},
        )
        state["router_decision"] = decision
        latency_ms = int((time.time() - start_time) * 1000)
        logger.info("router_node completed", agent_node="router", run_id=run_id, tools=tools,
                   confidence=confidence, reasoning=reasoning, latency_ms=round(latency_ms, 2),
                   payload={"input_intent": normalized.intent if normalized else None,
                            "input_normalized_query": normalized.normalized_query[:200] if normalized else None,
                            "output_tools": tools, "output_confidence": confidence, "output_reasoning": reasoning})
        if run_id:
            try:
                trace_service.log_event(
                    run_id=run_id, node_name="router", event_type="output",
                    payload={"tools": tools, "retry_strategy": True}, latency_ms=latency_ms
                )
            except Exception:
                pass
        return state

    try:
        thinking_logger = get_thinking_logger()
        if normalized.intent == "aggregate":
            tools = [TOOL_SQL]
            confidence = 0.9
            reasoning = "Intent: aggregate (SQL) data analysis."
        elif normalized.intent == "search":
            tools = [TOOL_RAG]
            confidence = 0.8
            reasoning = "Intent: search (RAG) semantic retrieval."
        elif normalized.intent == "hybrid":
            tools = [TOOL_SQL, TOOL_RAG]
            confidence = 0.85
            reasoning = "Intent: hybrid (SQL + RAG) comprehensive analysis."
        elif normalized.intent == "detail":
            tools = [TOOL_SQL, TOOL_RAG]
            confidence = 0.9
            reasoning = "Intent: detail (tell me about X) — always use both SQL and RAG for comprehensive answer."
        else:
            tools = [TOOL_RAG]
            confidence = 0.5
            reasoning = f"Fallback: unknown intent: {normalized.intent}."
        
        if run_id:
            thinking_logger.log_decision(
                run_id=run_id, node_name="router", decision=f"Route to {', '.join(tools)}",
                alternatives=[TOOL_SQL, TOOL_RAG, TOOL_HYBRID], rationale=reasoning, confidence=confidence
            )

        constraints = {}
        entities = normalized.entities if normalized and isinstance(normalized.entities, dict) else {}
        
        if entities.get("dates"):
            constraints["date_range"] = entities["dates"]
        if entities.get("statuses"):
            constraints["status"] = entities["statuses"]
        if entities.get("sample_types"):
            constraints["sample_type"] = entities["sample_types"]
        if entities.get("time_range"):
            constraints["time_range"] = entities["time_range"]

        decision = RouterDecision(
            tools=tools,
            route=ROUTE_IN_SCOPE,
            confidence=confidence,
            reasoning=reasoning,
            constraints=constraints
        )

        latency_ms = int((time.time() - start_time) * 1000)
        logger.info("router_node completed", agent_node="router", run_id=run_id,
                   tools=tools, confidence=confidence, latency_ms=round(latency_ms, 2),
                   payload={"input_intent": normalized.intent if normalized else None,
                           "input_normalized_query": normalized.normalized_query[:200] if normalized else None,
                           "output_tools": tools, "output_confidence": confidence, "output_reasoning": reasoning})
        state["router_decision"] = decision

        if run_id:
            try:
                trace_service.log_event(run_id=run_id, node_name="router", event_type="output",
                                       payload={"tools": tools, "confidence": confidence}, latency_ms=latency_ms)
            except Exception:
                pass

        options = request.get("options", {}) if isinstance(request, dict) else getattr(request, "options", {}) if hasattr(request, "options") else {}
        if (isinstance(options, dict) and options.get("debug")) or (hasattr(options, "debug") and getattr(options, "debug", False)):
            state["trace"].append({
                "node": "router", "input": {"intent": normalized.intent},
                "output": {"tools": tools, "confidence": confidence}, "latency_ms": round(latency_ms, 2)
            })
        
        return state
        
    except Exception as e:
        logger.error("router_node failed", run_id=run_id, error=str(e))
        
        if run_id:
            try:
                trace_service.log_event(run_id=run_id, node_name="router", event_type="error",
                                       payload={"error": str(e)})
            except Exception:
                pass
        
        state["router_decision"] = RouterDecision(
            tools=[TOOL_RAG], route=ROUTE_IN_SCOPE, confidence=0.5, reasoning=f"Error in routing: {str(e)}", constraints={}
        )
        return state