"""RAG retrieval node. Uses SQL result IDs when available to fetch descriptions by UUID filter."""
import time
import structlog
from typing import List, Dict, Any, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from agents.graph.state import AgentState
from agents.graph.stream_utils import emit_stream_event
from services.rag import RAGService
from services.embedder import EmbeddingService
from services.trace_service import TraceService
from services.config import get_app_config
from services.db import SupabaseService
from agents.constants import (
    TOOL_RAG,
    RAG_WEAK_MIN_AVG_CONTENT_LEN,
    RAG_CHUNK_CONTENT_TRUNCATE,
    RAG_TOP_CHUNKS,
    RAG_TOP_CHUNKS_PER_ENTITY,
    RAG_MAX_ENTITIES_FOR_ID_FETCH,
    RAG_MAX_CHUNKS_PER_SECTION,
    RAG_DEEP_FETCH_PER_ENTITY,
    RAG_DEEP_FETCH_TOP,
)
from agents.services.thinking_logger import get_thinking_logger

logger = structlog.get_logger()


def _entity_ids_from_sql_state(state: AgentState) -> Tuple[List[str], List[str]]:
    """Extract distinct project_ids and experiment_ids from sql_result or sql_runs for UUID-filtered RAG."""
    project_ids: List[str] = []
    experiment_ids: List[str] = []
    seen_p: Set[str] = set()
    seen_e: Set[str] = set()

    def _collect_from_rows(rows: List[Dict]) -> None:
        if not rows:
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            for k, v in row.items():
                if v is None or not str(v).strip():
                    continue
                k_lower = k.lower()
                if k_lower == "project_id":
                    s = str(v).strip()
                    if s and s not in seen_p:
                        seen_p.add(s)
                        project_ids.append(s)
                if k_lower == "experiment_id":
                    s = str(v).strip()
                    if s and s not in seen_e:
                        seen_e.add(s)
                        experiment_ids.append(s)

    sql_runs = state.get("sql_runs") or []
    if sql_runs:
        for run in sql_runs[:5]:
            if isinstance(run, dict) and not run.get("error"):
                _collect_from_rows(run.get("data") or [])
    sql_result = state.get("sql_result")
    if not sql_runs and sql_result and isinstance(sql_result, dict) and not sql_result.get("error"):
        _collect_from_rows(sql_result.get("data") or [])

    return (
        project_ids[:RAG_MAX_ENTITIES_FOR_ID_FETCH],
        experiment_ids[:RAG_MAX_ENTITIES_FOR_ID_FETCH],
    )


def _lab_note_ids_from_entities(state: AgentState, user_id: str) -> List[str]:
    """Resolve lab_note_titles from entities to lab note IDs for RAG scoping."""
    normalized = state.get("normalized_query")
    if not normalized or not user_id:
        return []
    entities = getattr(normalized, "entities", {}) or {}
    if not isinstance(entities, dict):
        return []
    titles = entities.get("lab_note_titles")
    if not titles or not isinstance(titles, list):
        return []
    try:
        db = SupabaseService()
        return db.get_lab_note_ids_by_titles(user_id, [str(t) for t in titles[:5]])
    except Exception as e:
        logger.warning("Failed to resolve lab_note_titles to IDs", error=str(e))
        return []


# Get similarity threshold from config
_app_config = None
def _get_rag_threshold():
    global _app_config
    if _app_config is None:
        _app_config = get_app_config()
    return _app_config.rag_similarity_threshold

DEFAULT_RAG_THRESHOLD = _get_rag_threshold()

# Singleton services
_rag_service: RAGService = None
_embedding_service: EmbeddingService = None
_trace_service: TraceService = None


def get_rag_service() -> RAGService:
    """Get or create RAG service singleton."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def get_trace_service() -> TraceService:
    """Get or create trace service singleton."""
    global _trace_service
    if _trace_service is None:
        _trace_service = TraceService()
    return _trace_service


def rag_node(state: AgentState) -> AgentState:
    """Execute RAG tool: embed query and search semantic chunks."""
    emit_stream_event(state, "thinking", {"node": "rag", "status": "started", "message": "Searching documents..."})
    start_time = time.time()
    router = state.get("router_decision")
    normalized = state.get("normalized_query")
    request = state["request"]
    run_id = state.get("run_id")
    trace_service = get_trace_service()
    
    if not router or TOOL_RAG not in router.tools:
        return state

    if not normalized:
        logger.error("rag_node: normalized_query missing", run_id=run_id)
        state["rag_result"] = []
        _lat = int((time.time() - start_time) * 1000)
        logger.info("rag_node completed", agent_node="rag", run_id=run_id,
                   chunks_found=0, avg_similarity=0.0, latency_ms=round(_lat, 2),
                   payload={"input_normalized_query": None, "output_chunks_found": 0, "output_error": "normalized_query missing"})
        if run_id:
            try:
                trace_service.log_event(
                    run_id=run_id, node_name="rag", event_type="error",
                    payload={"error": "normalized_query missing"}
                )
            except Exception:
                pass
        return state

    logger.info(
        "rag_node started",
        agent_node="rag",
        run_id=run_id,
        normalized_query=normalized.normalized_query[:100],
        payload={
            "input_query": request.get("query", "") if isinstance(request, dict) else getattr(request, "query", ""),
            "input_normalized_query": normalized.normalized_query,
            "input_intent": normalized.intent
        }
    )
    
    # Log input event
    if run_id:
        try:
            trace_service.log_event(
                run_id=run_id,
                node_name="rag",
                event_type="input",
                payload={"normalized_query": normalized.normalized_query[:200]}
            )
        except Exception:
            pass
    
    try:
        embedding_service = get_embedding_service()
        rag_service = get_rag_service()
        
        # Generate query embedding with error handling (base query)
        try:
            _t0 = time.time()
            query_embedding = embedding_service.embed_text(normalized.normalized_query)
            _embed_ms = int((time.time() - _t0) * 1000)
            logger.info("RAG embedding completed", run_id=run_id, latency_ms=_embed_ms)
            
            if not query_embedding or len(query_embedding) == 0:
                logger.error("RAG search: empty embedding generated", run_id=run_id)
                state["rag_result"] = []
                _lat = int((time.time() - start_time) * 1000)
                logger.info("rag_node completed", agent_node="rag", run_id=run_id,
                           chunks_found=0, avg_similarity=0.0, latency_ms=round(_lat, 2),
                           payload={"input_normalized_query": normalized.normalized_query[:200],
                                    "output_chunks_found": 0, "output_avg_similarity": 0.0, "output_error": "Empty embedding"})
                if run_id:
                    try:
                        trace_service.log_event(
                            run_id=run_id, node_name="rag", event_type="error",
                            payload={"error": "Empty embedding generated"}
                        )
                    except Exception:
                        pass
                return state
            
            logger.debug(
                "Query embedding generated",
                run_id=run_id,
                embedding_dim=len(query_embedding),
                query_preview=normalized.normalized_query[:100]
            )
        except Exception as e:
            logger.error(
                "RAG search: failed to generate embedding",
                run_id=run_id,
                error=str(e),
                error_type=type(e).__name__
            )
            state["rag_result"] = []
            _lat = int((time.time() - start_time) * 1000)
            logger.info("rag_node completed", agent_node="rag", run_id=run_id,
                       chunks_found=0, avg_similarity=0.0, latency_ms=round(_lat, 2),
                       payload={"input_normalized_query": normalized.normalized_query[:200],
                                "output_chunks_found": 0, "output_avg_similarity": 0.0, "output_error": str(e)[:100]})
            if run_id:
                try:
                    trace_service.log_event(
                        run_id=run_id, node_name="rag", event_type="error",
                        payload={"error": f"Embedding generation failed: {str(e)}"}
                    )
                except Exception:
                    pass
            return state

        user_id = request.get("user_id", "") if isinstance(request, dict) else getattr(request, "user_id", "")

        if not user_id:
            logger.error("RAG search: user_id required for data isolation, returning empty", run_id=run_id)
            state["rag_result"] = []
            _lat = int((time.time() - start_time) * 1000)
            logger.info("rag_node completed", agent_node="rag", run_id=run_id,
                       chunks_found=0, avg_similarity=0.0, latency_ms=round(_lat, 2),
                       payload={"input_normalized_query": normalized.normalized_query[:200],
                                "output_chunks_found": 0, "output_error": "user_id missing"})
            return state

        _t_search = time.time()
        app_config = get_app_config()
        match_threshold = _get_rag_threshold()
        project_ids, experiment_ids = _entity_ids_from_sql_state(state)
        lab_note_ids = _lab_note_ids_from_entities(state, user_id)
        if lab_note_ids:
            entities = getattr(normalized, "entities", {}) or {}
            logger.info(
                "RAG lab note scoping",
                run_id=run_id,
                lab_note_titles=entities.get("lab_note_titles", [])[:3],
                lab_note_ids_count=len(lab_note_ids),
            )
        # Deep fetch: when user asks to extract/pull/fetch content from a named document,
        # pull more chunks per entity for richer context
        _nq_lower = (normalized.normalized_query or "").lower()
        is_extraction_request = (
            normalized.intent == "aggregate"
            or (
                normalized.intent in ("search", "hybrid", "detail")
                and any(kw in _nq_lower for kw in ("pull out", "fetch", "extract", "show me", "get my", "retrieve"))
            )
        )
        chunks_per_note = RAG_DEEP_FETCH_PER_ENTITY if (is_extraction_request and lab_note_ids) else RAG_TOP_CHUNKS_PER_ENTITY
        if is_extraction_request and lab_note_ids:
            logger.info(
                "RAG deep fetch mode activated",
                run_id=run_id,
                chunks_per_note=chunks_per_note,
                lab_note_ids_count=len(lab_note_ids),
            )

        id_filtered_chunks: List[Dict[str, Any]] = []
        if lab_note_ids:
            for ln_id in lab_note_ids[:RAG_MAX_ENTITIES_FOR_ID_FETCH]:
                try:
                    chunks = rag_service.search_chunks(
                        query_embedding=query_embedding,
                        user_id=user_id,
                        source_ids=[ln_id],
                        match_threshold=match_threshold,
                        match_count=chunks_per_note,
                        return_below_threshold_for_entity=True,
                    )
                    id_filtered_chunks.extend(chunks)
                except Exception as e:
                    logger.warning("RAG lab note fetch failed", lab_note_id=ln_id[:8], error=str(e), run_id=run_id)
        if experiment_ids or project_ids:
            def _fetch_entity(entity_type: str, entity_id: str) -> List[Dict[str, Any]]:
                try:
                    return rag_service.search_chunks(
                        query_embedding=query_embedding,
                        user_id=user_id,
                        project_id=entity_id if entity_type == "project" else None,
                        experiment_id=entity_id if entity_type == "experiment" else None,
                        match_threshold=match_threshold,
                        match_count=RAG_TOP_CHUNKS_PER_ENTITY,
                        return_below_threshold_for_entity=True,
                    )
                except Exception as e:
                    logger.warning(f"RAG ID fetch failed for {entity_type}", entity_id=entity_id[:8], error=str(e), run_id=run_id)
                    return []

            tasks = [("experiment", eid) for eid in experiment_ids] + [("project", pid) for pid in project_ids]
            with ThreadPoolExecutor(max_workers=min(len(tasks), 6)) as pool:
                futures = {pool.submit(_fetch_entity, t, eid): (t, eid) for t, eid in tasks}
                for fut in as_completed(futures):
                    id_filtered_chunks.extend(fut.result())

            if id_filtered_chunks:
                _id_ms = int((time.time() - _t_search) * 1000)
                logger.info(
                    "RAG UUID-filtered fetch",
                    run_id=run_id,
                    experiment_ids=len(experiment_ids),
                    project_ids=len(project_ids),
                    id_chunks=len(id_filtered_chunks),
                    latency_ms=_id_ms,
                )

        # Global semantic search: optionally use hybrid search and fetch more raw candidates,
        # then group by (source_type, source_id, section_index, chunk_version).
        raw_multiplier = max(getattr(app_config, "rag_raw_chunks_multiplier", 3), 1)
        global_match_count = max(RAG_TOP_CHUNKS * raw_multiplier, RAG_TOP_CHUNKS)
        use_hybrid = getattr(app_config, "rag_use_hybrid", False)

        query_pairs: List[Tuple[str, List[float]]] = [(normalized.normalized_query, query_embedding)]

        chunks_semantic: List[Dict[str, Any]] = []
        source_ids_for_search = lab_note_ids if lab_note_ids else None
        for q_text, q_embedding in query_pairs:
            if use_hybrid:
                try:
                    res = rag_service.hybrid_search_chunks(
                        query_embedding=q_embedding,
                        query_text=q_text,
                        user_id=user_id,
                        organization_id=None,
                        project_id=None,
                        experiment_id=None,
                        source_ids=source_ids_for_search,
                        match_threshold=match_threshold,
                        match_count=global_match_count,
                        vector_weight=getattr(app_config, "rag_hybrid_vector_weight", 0.7),
                        text_weight=getattr(app_config, "rag_hybrid_text_weight", 0.3),
                    )
                    for ch in res:
                        if "combined_score" in ch:
                            ch["similarity"] = float(ch.get("combined_score", 0.0))
                    chunks_semantic.extend(res)
                    continue
                except Exception as e:
                    logger.warning(
                        "Hybrid RAG search failed for query, falling back to vector-only",
                        run_id=run_id,
                        error=str(e),
                    )
            # Vector-only path
            try:
                res = rag_service.search_chunks(
                    query_embedding=q_embedding,
                    user_id=user_id,
                    organization_id=None,
                    project_id=None,
                    experiment_id=None,
                    source_ids=source_ids_for_search,
                    match_threshold=match_threshold,
                    match_count=global_match_count,
                    return_below_threshold_for_entity=bool(source_ids_for_search),
                )
                chunks_semantic.extend(res)
            except Exception as e:
                logger.warning("Vector RAG search failed for query", run_id=run_id, error=str(e))

        _search_ms = int((time.time() - _t_search) * 1000)
        logger.info("RAG search phase completed", run_id=run_id, latency_ms=_search_ms)

        # Merge ID-filtered and global chunks, dedup by chunk id
        seen_chunk_ids: Set[str] = set()
        merged: List[Dict[str, Any]] = []
        for c in id_filtered_chunks + (chunks_semantic or []):
            cid = c.get("id")
            if cid and cid in seen_chunk_ids:
                continue
            if cid:
                seen_chunk_ids.add(cid)
            merged.append(c)

        # Sort by similarity (or combined score normalized above)
        merged.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)

        # Log raw retrieval stats before grouping
        raw_count = len(merged)

        # Group by (source_type, source_id, section_index, chunk_version) using metadata,
        # and keep up to RAG_MAX_CHUNKS_PER_SECTION chunks per group.
        grouped_chunks: List[Dict[str, Any]] = []
        groups: Dict[Tuple[str, str, Any, int], List[Dict[str, Any]]] = {}
        for c in merged:
            meta = c.get("metadata") or {}
            source_type = str(c.get("source_type") or "unknown")
            source_id = str(c.get("source_id") or "")
            section_index = meta.get("section_index")
            chunk_version = int(meta.get("chunk_version", 1))
            group_key = (source_type, source_id, section_index, chunk_version)
            bucket = groups.setdefault(group_key, [])
            if len(bucket) < RAG_MAX_CHUNKS_PER_SECTION:
                bucket.append(c)

        for bucket in groups.values():
            grouped_chunks.extend(bucket)

        # Trim to a reasonable size before experiment-level dedupe
        grouped_chunks.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        chunks = grouped_chunks[: max(RAG_TOP_CHUNKS * raw_multiplier, len(id_filtered_chunks))]

        # Deduplicate by experiment_id: keep multiple chunks from the same source
        # when they are all highly relevant (within 0.15 of the top similarity for that source).
        # This ensures long documents contribute full context, not just one snippet.
        from collections import defaultdict
        exp_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        no_exp = []
        for chunk in chunks:
            exp_id = chunk.get("experiment_id")
            if exp_id:
                exp_buckets[exp_id].append(chunk)
            else:
                no_exp.append(chunk)

        deduplicated = list(no_exp)
        for exp_id, exp_chunks in exp_buckets.items():
            exp_chunks.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
            top_sim = exp_chunks[0].get("similarity", 0.0)
            for c in exp_chunks:
                if top_sim - c.get("similarity", 0.0) <= 0.15:
                    deduplicated.append(c)

        deduplicated.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        effective_top_chunks = RAG_DEEP_FETCH_TOP if is_extraction_request else RAG_TOP_CHUNKS
        final_chunks = deduplicated[:effective_top_chunks]

        # Trace retrieval coverage if tracing is enabled
        if run_id:
            try:
                trace_service.log_event(
                    run_id=run_id,
                    node_name="rag",
                    event_type="retrieval_stats",
                    payload={
                        "raw_chunks": raw_count,
                        "groups": len(groups),
                        "grouped_chunks": len(grouped_chunks),
                        "final_chunks": len(final_chunks),
                    },
                )
            except Exception:
                pass
        
        # RAG-weak gate: empty, or max similarity below threshold, or low coverage (chunks too short)
        max_sim = max([c.get("similarity", 0.0) for c in final_chunks], default=0.0)
        avg_content_len = (
            sum(len(c.get("content", "") or "") for c in final_chunks) / len(final_chunks)
            if final_chunks else 0
        )
        min_content_len = getattr(get_app_config(), "agent_rag_weak_min_content_len", RAG_WEAK_MIN_AVG_CONTENT_LEN)
        rag_weak = (
            len(final_chunks) == 0
            or max_sim < match_threshold
            or (final_chunks and avg_content_len < min_content_len)
        )
        if rag_weak:
            flags = state.get("flags") or {}
            flags["rag_weak"] = True
            state["flags"] = flags
            logger.info(
                "rag_node: rag_weak flag set",
                run_id=run_id,
                chunks=len(final_chunks),
                max_sim=round(max_sim, 3),
                avg_content_len=round(avg_content_len, 0),
            )
        
        rag_result = []
        for chunk in final_chunks:
            rag_result.append({
                "chunk_id": chunk.get("id"),
                "source_type": chunk.get("source_type"),
                "source_id": chunk.get("source_id"),
                "experiment_id": chunk.get("experiment_id"),
                "content": chunk.get("content", "")[:RAG_CHUNK_CONTENT_TRUNCATE],
                "similarity": chunk.get("similarity", 0.0),
                "metadata": chunk.get("metadata", {})
            })
        
        latency_ms = int((time.time() - start_time) * 1000)
        avg_similarity = sum(c.get("similarity", 0.0) for c in final_chunks) / len(final_chunks) if final_chunks else 0.0
        logger.info("rag_node completed", agent_node="rag", run_id=run_id,
                   chunks_found=len(final_chunks), avg_similarity=round(avg_similarity, 3),
                   latency_ms=round(latency_ms, 2),
                   payload={"input_normalized_query": normalized.normalized_query[:200],
                           "output_chunks_found": len(final_chunks), "output_avg_similarity": round(avg_similarity, 3),
                           "output_top_similarity": round(max([c.get("similarity", 0.0) for c in final_chunks], default=0.0), 3) if final_chunks else 0.0})
        state["rag_result"] = rag_result
        # Emit RAG chunks to stream so client can display retrieved documents
        emit_stream_event(state, "thinking", {
            "node": "rag",
            "status": "completed",
            "message": f"Retrieved {len(rag_result)} document(s)",
        })
        # Accumulate chunks for summarizer (complete context across retries)
        rag_chunks_all = state.get("rag_chunks_all") or []
        state["rag_chunks_all"] = list(rag_chunks_all) + list(rag_result)
        attempted = state.get("attempted_tools") or []
        if TOOL_RAG not in attempted:
            attempted.append(TOOL_RAG)
            state["attempted_tools"] = attempted

        run_log = state.get("run_process_log") or []
        run_log.append({
            "phase": "rag",
            "attempt": state.get("retry_count", 0),
            "chunks_count": len(rag_result),
            "latency_ms": latency_ms,
        })
        state["run_process_log"] = run_log
        run_cits = state.get("run_citations") or []
        for chunk in rag_result:
            if isinstance(chunk, dict) and chunk.get("source_id") is not None:
                run_cits.append({
                    "source_type": chunk.get("source_type", "unknown"),
                    "source_id": str(chunk.get("source_id")),
                    "chunk_id": chunk.get("chunk_id"),
                    "relevance": float(chunk.get("similarity", 0.0)),
                    "excerpt": (chunk.get("content") or "")[:200],
                })
        state["run_citations"] = run_cits

        thinking_logger = get_thinking_logger()
        if run_id:
            thinking_logger.log_analysis(
                run_id=run_id, node_name="rag",
                analysis=f"Retrieved {len(final_chunks)} semantic chunks",
                data_summary={"chunks_found": len(final_chunks), "avg_similarity": round(avg_similarity, 3)},
                insights=[f"Average similarity: {avg_similarity:.3f}"]
            )
        
        if run_id:
            try:
                trace_service.log_event(
                    run_id=run_id, node_name="rag", event_type="output",
                    payload={"chunks_found": len(final_chunks), "avg_similarity": round(avg_similarity, 3)},
                    latency_ms=latency_ms
                )
            except Exception:
                pass
        
        options = request.get("options", {}) if isinstance(request, dict) else getattr(request, "options", {}) if hasattr(request, "options") else {}
        if (isinstance(options, dict) and options.get("debug")) or (hasattr(options, "debug") and getattr(options, "debug", False)):
            state["trace"].append({
                "node": "rag", "input": {"normalized_query": normalized.normalized_query[:200]},
                "output": {"chunks_found": len(final_chunks)}, "latency_ms": round(latency_ms, 2)
            })
        
        return state
        
    except Exception as e:
        logger.error("rag_node failed", run_id=run_id, error=str(e))
        
        if run_id:
            try:
                trace_service.log_event(run_id=run_id, node_name="rag", event_type="error",
                                       payload={"error": str(e)})
            except Exception:
                pass
        
        state["rag_result"] = []
        return state