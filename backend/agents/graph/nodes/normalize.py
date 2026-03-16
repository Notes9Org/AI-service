"""Normalize user query node."""
import time
import structlog
from typing import List, Tuple
from tenacity import RetryError

from agents.prompt_loader import load_prompt
from agents.graph.state import AgentState
from agents.graph.stream_utils import emit_stream_event
from agents.contracts.normalized import NormalizedQuery
from agents.services.llm_client import LLMClient, LLMError
from services.trace_service import TraceService
from agents.services.thinking_logger import get_thinking_logger

logger = structlog.get_logger()

# Singleton LLM client (will be initialized on first use)
_llm_client: LLMClient = None
_trace_service: TraceService = None


def get_llm_client() -> LLMClient:
    """Get or create LLM client singleton."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def get_trace_service() -> TraceService:
    """Get or create trace service singleton."""
    global _trace_service
    if _trace_service is None:
        _trace_service = TraceService()
    return _trace_service


def _validate_normalized(normalized: NormalizedQuery, request: dict) -> Tuple[bool, List[str]]:
    """Validate normalized output against invariants. Returns (is_valid, issues)."""
    issues: List[str] = []
    if not normalized.in_scope:
        if not normalized.out_of_scope_reason or not str(normalized.out_of_scope_reason).strip():
            issues.append("in_scope is false but out_of_scope_reason is empty")
    else:
        if normalized.intent not in ("aggregate", "search", "hybrid", "detail"):
            issues.append("in_scope is true but intent must be one of aggregate, search, hybrid, detail")
    if normalized.in_scope:
        if normalized.intent == "aggregate" and not normalized.context.get("requires_aggregation"):
            issues.append("Intent is 'aggregate' but context.requires_aggregation is not True")
        if normalized.intent == "search" and not normalized.context.get("requires_semantic_search"):
            issues.append("Intent is 'search' but context.requires_semantic_search is not True")
        if normalized.intent == "hybrid":
            if not normalized.context.get("requires_aggregation"):
                issues.append("Intent is 'hybrid' but context.requires_aggregation is not True")
            if not normalized.context.get("requires_semantic_search"):
                issues.append("Intent is 'hybrid' but context.requires_semantic_search is not True")
        if normalized.intent == "detail":
            if not normalized.context.get("requires_aggregation"):
                issues.append("Intent is 'detail' but context.requires_aggregation is not True")
            if not normalized.context.get("requires_semantic_search"):
                issues.append("Intent is 'detail' but context.requires_semantic_search is not True")
    if not normalized.normalized_query or not normalized.normalized_query.strip():
        issues.append("normalized_query is empty")
    if not isinstance(normalized.entities, dict):
        issues.append("entities is not a dict")
    if not isinstance(normalized.context, dict):
        issues.append("context is not a dict")
    return len(issues) == 0, issues


def normalize_node(state: AgentState) -> AgentState:
    """Normalize user query into structured format with intent, entities, and context."""
    emit_stream_event(state, "thinking", {"node": "normalize", "status": "started", "message": "Understanding your query..."})
    start_time = time.time()
    request = state["request"]
    run_id = state.get("run_id")
    trace_service = get_trace_service()
    
    query_text = request.get("query", "") if isinstance(request, dict) else getattr(request, "query", "")
    user_id = request.get("user_id", "") if isinstance(request, dict) else getattr(request, "user_id", "")
    history = request.get("history", []) if isinstance(request, dict) else getattr(request, "history", [])
    
    logger.info(
        "normalize_node started",
        agent_node="normalize",
        run_id=run_id,
        query=query_text[:100],
        user_id=user_id,
        has_history=len(history) > 0,
        payload={
            "input_query": query_text,
            "user_id": user_id,
            "history_length": len(history),
            "history_preview": [{"role": msg.get("role", getattr(msg, "role", "user")), "content": (msg.get("content", getattr(msg, "content", ""))[:100])} for msg in history[-2:]] if history else []
        }
    )
    
    if run_id:
        try:
            trace_service.log_event(run_id=run_id, node_name="normalize", event_type="input",
                                   payload={"query": query_text[:200], "has_history": len(history) > 0})
        except Exception:
            pass
    
    try:
        history_text = ""
        if history:
            history_text = "\n".join([
                f"{msg.get('role', getattr(msg, 'role', 'user'))}: {msg.get('content', getattr(msg, 'content', ''))}"
                for msg in history[-5:]
            ])
        if not history_text:
            history_text = "None"

        prompt_template = load_prompt("normalize", "classify_query")
        prompt = prompt_template.format(
            query_text=query_text,
            history_text=history_text,
        )

        llm_client = get_llm_client()
        schema = {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "enum": ["lab", "general", "unknown"]},
                "in_scope": {"type": "boolean"},
                "out_of_scope_reason": {"type": ["string", "null"]},
                "intent": {"type": "string", "enum": ["aggregate", "search", "hybrid", "detail", "other"]},
                "normalized_query": {"type": "string"},
                "entities": {"type": "object"},
                "context": {"type": "object"},
                "history_summary": {"type": ["string", "null"]}
            },
            "required": ["domain", "in_scope", "out_of_scope_reason", "intent", "normalized_query", "entities", "context"]
        }
        
        from services.config import get_app_config
        normalize_temperature = get_app_config().normalize_temperature
        
        llm_retries = 0
        try:
            result = llm_client.complete_json(prompt, schema, temperature=normalize_temperature)
        except LLMError as e:
            llm_retries = 1
            logger.warning("Normalization failed, retrying once", error=str(e))
            try:
                result = llm_client.complete_json(prompt, schema, temperature=normalize_temperature)
            except LLMError as e2:
                logger.error("Normalization failed after retry", error=str(e2))
                raise
        
        if not result:
            raise ValueError("LLM returned empty result")
        
        normalized = NormalizedQuery(**result)
        
        thinking_logger = get_thinking_logger()
        if run_id:
            thinking_logger.log_reasoning(
                run_id=run_id,
                node_name="normalize",
                reasoning=f"Normalized query from '{query_text}' to '{normalized.normalized_query}'",
                factors=[
                    f"Domain: {normalized.domain}, in_scope: {normalized.in_scope}",
                    f"Intent: {normalized.intent}",
                    f"Entities: {len(normalized.entities)}"
                ],
                conclusion=f"Query classified as {normalized.intent} intent"
            )
                # Stream thinking for UI: intent and conclusion while loading
        emit_stream_event(state, "thinking", {
            "node": "normalize",
            "status": "completed",
            "message": "Query understood",
            "intent": normalized.intent,
            "conclusion": f"Query classified as {normalized.intent} intent",
        })
        is_valid, validation_issues = _validate_normalized(normalized, request)
        if not is_valid:
            logger.warning("Normalize validation issues", run_id=run_id, issues=validation_issues)
            if run_id:
                thinking_logger.log_validation(
                    run_id=run_id, node_name="normalize", validation_type="invariant",
                    criteria=["intent matches context", "query not empty", "in_scope/out_of_scope_reason"],
                    result="fail", issues=validation_issues
                )
        
        state["normalized_query"] = normalized
        
        # Out-of-scope: generate polite message via LLM and set final_response so graph goes to final
        if not normalized.in_scope:
            out_prompt_template = load_prompt("normalize", "out_of_scope_response")
            out_prompt = out_prompt_template.format(
                query_text=query_text,
                out_of_scope_reason=normalized.out_of_scope_reason or "Not about lab management",
            )

            try:
                out_msg = llm_client.complete_text(
                    prompt=out_prompt,
                    temperature=0.3,
                )
                out_msg = (out_msg or "").strip() or "This question isn't related to lab management. I can help with experiments, projects, samples, protocols, and lab notes."
            except Exception as e:
                logger.warning("Out-of-scope message generation failed, using fallback", error=str(e), run_id=run_id)
                out_msg = "This question isn't related to lab management. I can help with experiments, projects, samples, protocols, and lab notes."
            from agents.contracts.response import FinalResponse
            state["final_response"] = FinalResponse(
                answer=out_msg,
                citations=[],
                confidence=0.0,
                tool_used="none",
            )
            latency_ms = int((time.time() - start_time) * 1000)
            logger.info("normalize_node completed (out-of-scope)", agent_node="normalize", run_id=run_id,
                       in_scope=False, domain=normalized.domain, out_of_scope_reason=normalized.out_of_scope_reason,
                       latency_ms=round(latency_ms, 2))
            if run_id:
                try:
                    trace_service.log_event(
                        run_id=run_id, node_name="normalize", event_type="output",
                        payload={"in_scope": False, "domain": normalized.domain, "out_of_scope_reason": normalized.out_of_scope_reason},
                        latency_ms=latency_ms
                    )
                except Exception:
                    pass
            return state
        
        latency_ms = int((time.time() - start_time) * 1000)
        logger.info("normalize_node completed", agent_node="normalize", run_id=run_id,
                   intent=normalized.intent, in_scope=normalized.in_scope, normalized_query=normalized.normalized_query[:100],
                   entities_count=len(normalized.entities), latency_ms=round(latency_ms, 2),
                   payload={"input_query": query_text[:200], "output_intent": normalized.intent,
                           "output_normalized_query": normalized.normalized_query[:200], "output_entities_count": len(normalized.entities)})
        
        if run_id:
            try:
                trace_service.log_event(
                    run_id=run_id, node_name="normalize", event_type="output",
                    payload={"intent": normalized.intent, "in_scope": True, "entities_count": len(normalized.entities)},
                    latency_ms=latency_ms
                )
            except Exception:
                pass
        
        options = request.get("options", {}) if isinstance(request, dict) else getattr(request, "options", {})
        if (isinstance(options, dict) and options.get("debug")) or (hasattr(options, "debug") and getattr(options, "debug", False)):
            state["trace"].append({
                "node": "normalize",
                "input": {"query": query_text[:200]},
                "output": {"intent": normalized.intent, "normalized_query": normalized.normalized_query[:200], "in_scope": normalized.in_scope},
                "latency_ms": round(latency_ms, 2)
            })
        
        return state
        
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        cause = e
        if isinstance(e, RetryError):
            future = getattr(e, "last_attempt", None) or (e.args[0] if e.args else None)
            if future is not None and getattr(future, "exception", None):
                try:
                    inner = future.exception()
                    if inner is not None:
                        cause = inner
                except Exception:
                    pass
        err_msg = (str(cause) or str(e))[:200]
        logger.error("normalize_node failed", run_id=run_id, error=err_msg, raw_error=str(e))
        if run_id:
            try:
                trace_service.log_event(run_id=run_id, node_name="normalize", event_type="error",
                                       payload={"error": str(e)}, latency_ms=latency_ms)
            except Exception:
                pass
        
        from agents.contracts.response import FinalResponse
        state["final_response"] = FinalResponse(
            answer=f"Error normalizing query: {str(e)}", citations=[], confidence=0.0, tool_used="none"
        )
        return state