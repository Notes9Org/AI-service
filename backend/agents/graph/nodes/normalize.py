"""Normalize user query node."""
import time
import structlog
from typing import List, Tuple
from tenacity import RetryError
from agents.graph.state import AgentState
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
        if normalized.intent not in ("aggregate", "search", "hybrid"):
            issues.append("in_scope is true but intent must be one of aggregate, search, hybrid")
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
    if not normalized.normalized_query or not normalized.normalized_query.strip():
        issues.append("normalized_query is empty")
    if not isinstance(normalized.entities, dict):
        issues.append("entities is not a dict")
    if not isinstance(normalized.context, dict):
        issues.append("context is not a dict")
    return len(issues) == 0, issues


def normalize_node(state: AgentState) -> AgentState:
    """Normalize user query into structured format with intent, entities, and context."""
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

DOMAIN and IN-SCOPE rules (important):
- Users store lab notes, literature, protocols, and reports IN the system. They often ask to "explain X", "what is X", "tell me about X", "find notes about X" for topics they have documented (e.g. IOCL workflow, attention mechanism, PCR, a procedure). Such queries are IN-SCOPE with intent "search" (RAG): we search their content and answer from it.
- Set in_scope=TRUE and intent="search" for: "Explain about X", "What is X", "Tell me about X", "Find notes about X", "Summarize X" — so we search lab notes and literature. Only set in_scope=FALSE for queries that clearly cannot be in lab content (e.g. weather, recipes, sports, "write a poem", "current news").
- In-scope: (1) Structured/data queries → aggregate or hybrid (experiments, projects, counts, status, IDs). (2) Explanation or search-for-information queries → search (RAG will find user's notes/literature about the topic).
- Out-of-scope only when: query is plainly unrelated to any possible lab/research content (weather, cooking, entertainment, etc.).

User Query: {query_text}

Conversation History:
{history_text if history_text else 'None'}

Extract and return JSON with:
1. domain: "lab" if the query could be answered from lab content (notes, literature, experiments, etc.); "general" only if clearly not (weather, recipes); "unknown" if doubtful — when doubtful, prefer "lab" so we try search.
2. in_scope: true if we should try to answer (search user content and/or DB). false only for clearly off-topic queries.
3. out_of_scope_reason: string or null. When in_scope is false, set a short reason. When in_scope is true, set null.
4. intent: "aggregate" (SQL), "search" (RAG — use for explain/find notes/what is X), "hybrid", or "other" (only when in_scope is false).
5. normalized_query: Cleaned query text preserving scientific terms.
6. entities: Dict with extracted entities when relevant; empty when pure search.
7. context: Dict with requires_aggregation, requires_semantic_search (true for search intent), time_range when relevant.
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
            answer=f"Error normalizing query: {str(e)}", citations=[], confidence=0.0, tool_used="rag"
        )
        return state