"""Normalize user query node."""
import time
import structlog
from tenacity import RetryError
from agents.graph.state import AgentState
from agents.contracts.normalized import NormalizedQuery
from agents.services.llm_client import LLMClient, LLMError
from services.trace_service import TraceService
from agents.graph.nodes.normalize_validator import validate_normalized_output
from agents.graph.nodes._debug import node_start, node_end
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


def normalize_node(state: AgentState) -> AgentState:
    """Normalize user query into structured format with intent, entities, and context."""
    node_start("normalize")
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
        
        prompt = f"""Normalize the following user query for a scientific lab management system.

DOMAIN (in-scope only): This system supports queries about:
- Experiments, projects, samples, protocols, equipment, reports, lab notes, literature within the system.
- Counting/listing/filtering experiments, projects, samples; searching notes and literature; status and metadata.
If the user's query is NOT about this domain (e.g. weather, recipes, general knowledge, other subjects), set in_scope=false, domain="general" or "unknown", intent="other", and set out_of_scope_reason to a short explanation.

User Query: {query_text}

Conversation History:
{history_text if history_text else 'None'}

Extract and return JSON with:
1. domain: One of "lab", "general", "unknown"
   - "lab": Query is clearly about lab management (experiments, projects, samples, protocols, equipment, reports, lab notes, literature).
   - "general": Query is about something else (general knowledge, weather, etc.).
   - "unknown": Unclear; treat as out-of-scope if doubtful.
2. in_scope: boolean. True only if the query is about the lab domain above; false otherwise.
3. out_of_scope_reason: string or null. When in_scope is false, set a short reason (e.g. "general knowledge", "weather", "unrelated topic"). When in_scope is true, set null.
4. intent: One of "aggregate", "search", "hybrid", "other"
   - "aggregate": In-scope queries needing SQL (counts, status, filters, IDs).
   - "search": In-scope semantic/conceptual search (notes, literature, content).
   - "hybrid": In-scope queries needing both SQL and semantic search.
   - "other": Use when in_scope is false (query not about lab domain).
5. normalized_query: Cleaned query text preserving scientific terms (or original intent if out-of-scope).
6. entities: Dict with extracted entities (dates, experiment_ids, project_names, etc.) when in-scope; empty when out-of-scope.
7. context: Dict with requires_aggregation, requires_semantic_search, time_range when in-scope; empty when out-of-scope.
8. history_summary: Optional string summarizing relevant conversation history (only if needed).

Return ONLY valid JSON matching this structure:
{{
  "domain": "lab|general|unknown",
  "in_scope": true or false,
  "out_of_scope_reason": null or "short reason string",
  "intent": "aggregate|search|hybrid|other",
  "normalized_query": "cleaned query text",
  "entities": {{}},
  "context": {{}},
  "history_summary": null or "summary string"
}}"""

        llm_client = get_llm_client()
        schema = {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "enum": ["lab", "general", "unknown"]},
                "in_scope": {"type": "boolean"},
                "out_of_scope_reason": {"type": ["string", "null"]},
                "intent": {"type": "string", "enum": ["aggregate", "search", "hybrid", "other"]},
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
        
        is_valid, validation_issues = validate_normalized_output(normalized, request)
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
            out_prompt = f"""The user asked something that is not related to this system's domain.

User query: {query_text}
Reason out of scope: {normalized.out_of_scope_reason or 'Not about lab management'}

This system only helps with: experiments, projects, samples, protocols, equipment, reports, lab notes, and literature within the lab management system.

Generate a single short, polite response (1-2 sentences) that:
1. States that this question is not within the system's domain or not something we can help with (similar meaning).
2. Briefly mentions what we can help with (lab management: experiments, projects, samples, etc.).
Do not apologize excessively. Be clear and helpful. Return ONLY the response text, no JSON, no quotes."""

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
            node_end("normalize", int(latency_ms))
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
        node_end("normalize", int(latency_ms))
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
        node_end("normalize", int(latency_ms), f"error: {err_msg}")
        logger.error("normalize_node failed", run_id=run_id, error=err_msg, raw_error=str(e))
        print(f"[NODE] NORMALIZE failed: {err_msg}", flush=True)
        if run_id:
            try:
                trace_service.log_event(run_id=run_id, node_name="normalize", event_type="error",
                                       payload={"error": str(e)}, latency_ms=latency_ms)
            except Exception:
                pass
        
        from agents.contracts.response import FinalResponse
        state["final_response"] = FinalResponse(
            answer=f"Error normalizing query: {str(e)}", citations=[], confidence=0.0, tool_used="rag"
        )
        return state