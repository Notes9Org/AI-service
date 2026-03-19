"""Normalize user query node."""
import re
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

# ---------------------------------------------------------------------------
# Compiled regex patterns for programmatic entity extraction (post-LLM patch)
# ---------------------------------------------------------------------------

# "Here is the lab note: X", "lab note titled X", "lab note called X"
_LAB_NOTE_ASSERTION_RE = re.compile(
    r"(?:here\s+is\s+the\s+lab\s*note|"
    r"lab\s*note\s*(?:titled?|called|named|is)\s*)"
    r"[:\s]+[\"']?(.+?)[\"']?\s*(?:\.{3}|[.!?]|$)",
    re.IGNORECASE,
)
# "Lab Note: Intro and Background" in assistant messages
_LAB_NOTE_IN_HISTORY_RE = re.compile(
    r"(?:Lab\s+Note|lab\s*note)\s*:\s*(.+?)(?:\n|[,;\")\]]|$)",
    re.IGNORECASE,
)
# "in my ASOs PhD project", "notes in Cancer drug", "under project X"
_PROJECT_SCOPE_RE = re.compile(
    r"(?:in\s+(?:my\s+)?(?:project\s+)?|notes?\s+in\s+|under\s+(?:project\s+)?)"
    r"[\"']?([A-Z][\w\s'-]{2,40}?)(?:\s+(?:project|and\s+answer|about)|[\"']|$)",
    re.IGNORECASE,
)
# "pull out my notes", "fetch the content", "show my section"
_FETCH_VERB_RE = re.compile(
    r"\b(?:pull\s+out|fetch|show|get|retrieve|display)\b.*\b(?:notes?|content|section)\b",
    re.IGNORECASE,
)
# "what are X", "how does X work", "explain X"
_TOPIC_QUESTION_RE = re.compile(
    r"\b(?:what\s+(?:are|is|were|was)|how\s+(?:are|is|do|does|to)|explain|tell\s+me\s+about|describe)\b",
    re.IGNORECASE,
)
_FOLLOW_UP_PRONOUNS = {"this", "that", "the", "it", "those", "these"}

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


def _smart_truncate(role: str, content: str, max_len: int = 800) -> str:
    """Truncate message content, preserving entity references in assistant messages."""
    if len(content) <= max_len:
        return content
    if role != "assistant":
        return content[: max_len - 3] + "..."
    # For assistant messages: keep the answer lead + entity-bearing lines
    lead = content[:250]
    entity_lines = []
    for line in content.split("\n"):
        if re.search(
            r"Lab\s+Note:|Project:|Experiment:|Protocol:|\[Referenced:",
            line,
            re.IGNORECASE,
        ):
            entity_lines.append(line.strip())
    entity_block = "\n".join(entity_lines)[:400]
    if entity_block:
        result = lead + "\n...\n" + entity_block
    else:
        result = content[: max_len - 3] + "..."
    return result[:max_len]


def _patch_entities_from_text(
    normalized: NormalizedQuery, query_text: str, history: list
) -> NormalizedQuery:
    """Post-LLM safety net: extract entities the LLM missed using regex patterns.

    Handles:
    - Explicit lab note assertions: "Here is the lab note: X"
    - Project scoping: "in my ASOs PhD project"
    - Follow-up pronoun resolution: "this section" → look up entity from history
    - Fetch verb detection → upgrade intent to aggregate
    - Topic + entity combo → upgrade intent to hybrid
    """
    entities = dict(normalized.entities) if isinstance(normalized.entities, dict) else {}
    context = dict(normalized.context) if isinstance(normalized.context, dict) else {}
    intent = normalized.intent
    in_scope = normalized.in_scope
    changes: List[str] = []

    # 1. Explicit lab note assertion in current query
    m = _LAB_NOTE_ASSERTION_RE.search(query_text)
    if m:
        title = m.group(1).strip().rstrip(".")
        if title and len(title) > 1:
            existing = entities.get("lab_note_titles", [])
            if title not in existing:
                entities.setdefault("lab_note_titles", []).append(title)
                in_scope = True
                changes.append(f"assertion: lab_note_titles=[{title}]")

    # 2. Project scoping in current query
    m = _PROJECT_SCOPE_RE.search(query_text)
    if m:
        project = m.group(1).strip().rstrip(".")
        if project and len(project) > 2:
            existing = entities.get("project_names", [])
            if project not in existing:
                entities.setdefault("project_names", []).append(project)
                in_scope = True
                changes.append(f"scope: project_names=[{project}]")

    # 3. Follow-up pronoun resolution from history
    query_words = set(query_text.lower().split())
    is_follow_up = bool(query_words & _FOLLOW_UP_PRONOUNS) or bool(_FETCH_VERB_RE.search(query_text))
    if is_follow_up and not entities.get("lab_note_titles"):
        # Scan assistant messages in history (most recent first) for lab note names
        for msg in reversed(history[-6:]):
            role = msg.get("role", getattr(msg, "role", "user"))
            content = msg.get("content", getattr(msg, "content", ""))
            if role != "assistant":
                continue
            # Check for [Referenced: ...] tags first (most reliable, added by Zep enrichment)
            ref_match = re.search(r"\[Referenced:.*?lab.note.title?:\s*(.+?)(?:;|\])", content, re.I)
            if ref_match:
                title = ref_match.group(1).strip()
                if title:
                    entities.setdefault("lab_note_titles", []).append(title)
                    in_scope = True
                    changes.append(f"history-ref: lab_note_titles=[{title}]")
                    break
            # Check for "Lab Note: X" pattern in assistant response
            for ln_match in _LAB_NOTE_IN_HISTORY_RE.finditer(content):
                title = ln_match.group(1).strip()
                if title and len(title) > 1:
                    entities.setdefault("lab_note_titles", []).append(title)
                    in_scope = True
                    changes.append(f"history-scan: lab_note_titles=[{title}]")
                    break
            if entities.get("lab_note_titles"):
                break

    # 4. Intent upgrade: fetch verb + named document → aggregate
    if _FETCH_VERB_RE.search(query_text) and entities.get("lab_note_titles"):
        if intent not in ("aggregate", "hybrid"):
            intent = "aggregate"
            context["requires_aggregation"] = True
            changes.append(f"intent-upgrade: {normalized.intent}->aggregate")

    # 5. Intent upgrade: topic question + entity reference → hybrid
    elif _TOPIC_QUESTION_RE.search(query_text) and (
        entities.get("project_names") or entities.get("lab_note_titles")
    ):
        if intent == "search":
            intent = "hybrid"
            context["requires_aggregation"] = True
            context["requires_semantic_search"] = True
            changes.append("intent-upgrade: search->hybrid")
        elif intent == "other":
            intent = "search"
            context["requires_semantic_search"] = True
            in_scope = True
            changes.append("intent-upgrade: other->search (has entities)")

    # 6. Any entity extracted but still out-of-scope → force in-scope
    if not in_scope and any(entities.get(k) for k in ("lab_note_titles", "project_names", "experiment_names")):
        in_scope = True
        if intent == "other":
            intent = "search"
            context["requires_semantic_search"] = True
        changes.append("force-in-scope: entities found")

    # Apply patches if any changes were made
    if not changes:
        return normalized

    try:
        update = {
            "entities": entities,
            "context": context,
            "intent": intent,
            "in_scope": in_scope,
        }
        if in_scope:
            update["out_of_scope_reason"] = None
        copy_fn = getattr(normalized, "model_copy", None) or getattr(normalized, "copy", None)
        if copy_fn:
            patched = copy_fn(update=update)
        else:
            patched = NormalizedQuery(**{**normalized.__dict__, **update})
        logger.info(
            "normalize: entities patched by programmatic extraction",
            changes=changes,
            patched_entities=list(entities.keys()),
            patched_intent=intent,
        )
        return patched
    except Exception as exc:
        logger.warning("normalize: entity patching failed", error=str(exc))
        return normalized


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
            history_lines = []
            for msg in history[-6:]:
                role = msg.get('role', getattr(msg, 'role', 'user'))
                content = msg.get('content', getattr(msg, 'content', ''))
                content = _smart_truncate(role, content)
                history_lines.append(f"{role}: {content}")
            history_text = "\n".join(history_lines)
        if not history_text:
            history_text = "None"

        zep_context = request.get("zep_context", "") if isinstance(request, dict) else getattr(request, "zep_context", "") or ""
        if not zep_context or not str(zep_context).strip():
            zep_context = "None"

        prompt_template = load_prompt("normalize", "classify_query")
        prompt = prompt_template.format(
            query_text=query_text,
            history_text=history_text,
            zep_context=zep_context,
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

        # Post-LLM safety net: patch entities the LLM missed using regex patterns
        normalized = _patch_entities_from_text(normalized, query_text, history)

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