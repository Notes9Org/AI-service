"""RAG (Retrieval Augmented Generation) service for semantic search."""
import json
import re
import threading
from typing import Optional, List, Dict, Any
import numpy as np
import structlog

from services.db import SupabaseService
from services.config import get_database_config
from services.config_errors import ConfigurationError

logger = structlog.get_logger()

# Try pgvector for server-side vector search; fall back to client-side if unavailable
try:
    from pgvector.psycopg2 import register_vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    register_vector = None  # type: ignore

# Try connection pooling; fall back to per-call connections
try:
    from psycopg2.pool import ThreadedConnectionPool
    _POOL_AVAILABLE = True
except ImportError:
    _POOL_AVAILABLE = False
    ThreadedConnectionPool = None  # type: ignore

_pool_lock = threading.Lock()
_connection_pool: Optional["ThreadedConnectionPool"] = None


def _get_connection_pool() -> Optional["ThreadedConnectionPool"]:
    """Lazily create a shared connection pool for pgvector searches."""
    global _connection_pool
    if not _POOL_AVAILABLE or not PGVECTOR_AVAILABLE:
        return None
    if _connection_pool is not None:
        return _connection_pool
    with _pool_lock:
        if _connection_pool is not None:
            return _connection_pool
        try:
            db_config = get_database_config()
            args, kwargs = db_config.get_pool_connection_params()
            kwargs["sslmode"] = kwargs.get("sslmode", "require")
            _connection_pool = ThreadedConnectionPool(1, 8, *args, **kwargs)
            logger.info("RAG connection pool created", minconn=1, maxconn=8)
            return _connection_pool
        except Exception as e:
            logger.warning("Failed to create RAG connection pool", error=str(e))
            return None


def parse_embedding(embedding) -> Optional[List[float]]:
    """
    Parse embedding from various formats (list, string, etc.)
    Returns list of floats or None if parsing fails.
    """
    if embedding is None:
        return None
    
    # Already a list
    if isinstance(embedding, list):
        try:
            return [float(x) for x in embedding]
        except (ValueError, TypeError):
            return None
    
    # String representation
    if isinstance(embedding, str):
        try:
            # Try JSON parsing first
            parsed = json.loads(embedding)
            if isinstance(parsed, list):
                return [float(x) for x in parsed]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        
        # Try parsing as comma-separated or space-separated
        try:
            # Remove brackets if present
            cleaned = embedding.strip('[]')
            # Split by comma or space
            if ',' in cleaned:
                values = cleaned.split(',')
            else:
                values = cleaned.split()
            return [float(x.strip()) for x in values if x.strip()]
        except (ValueError, AttributeError):
            return None
    
    return None


class RAGService:
    """Service for semantic search and retrieval."""

    def __init__(self, db_service: Optional[SupabaseService] = None):
        """
        Initialize RAG service.

        Args:
            db_service: Optional SupabaseService instance. If not provided, creates a new one.
        """
        self.db = db_service if db_service else SupabaseService()
        logger.info("RAG service initialized")

    def _search_chunks_pgvector(
        self,
        query_embedding: List[float],
        user_id: str,
        organization_id: Optional[str] = None,
        project_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        source_types: Optional[List[str]] = None,
        match_threshold: float = 0.75,
        match_count: int = 6,
        apply_threshold: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Server-side vector search using pgvector. Returns chunks with similarity scores.
        Uses cosine distance (<=>) - similarity = 1 - distance.
        """
        if not PGVECTOR_AVAILABLE or register_vector is None:
            raise RuntimeError("pgvector not available")

        query_vec = np.array(query_embedding, dtype=np.float32)
        conditions = ["created_by = %s", "embedding IS NOT NULL"]

        if organization_id:
            conditions.append("organization_id = %s")
        if project_id:
            conditions.append("project_id = %s")
        if experiment_id:
            conditions.append("experiment_id = %s")
        if source_types:
            conditions.append("source_type = ANY(%s)")

        where_sql = " AND ".join(conditions)
        threshold_sql = " AND (1 - (embedding <=> %s)) >= %s" if apply_threshold else ""
        limit_val = match_count

        cols = "id, source_type, source_id, organization_id, project_id, experiment_id, chunk_index, content, metadata, created_at, created_by"
        sql = f"""
            SELECT {cols},
                   1 - (embedding <=> %s) AS similarity
            FROM semantic_chunks
            WHERE {where_sql}{threshold_sql}
            ORDER BY embedding <=> %s, id
            LIMIT %s
        """
        # Params in SQL order: SELECT %s, WHERE (user_id, org?, proj?, exp?, src?), threshold (vec, thresh)?, ORDER BY %s, LIMIT %s
        params_for_sql: List[Any] = [query_vec]  # SELECT
        params_for_sql.append(user_id)
        if organization_id:
            params_for_sql.append(organization_id)
        if project_id:
            params_for_sql.append(project_id)
        if experiment_id:
            params_for_sql.append(experiment_id)
        if source_types:
            params_for_sql.append(source_types)
        if apply_threshold:
            params_for_sql.extend([query_vec, match_threshold])
        params_for_sql.extend([query_vec, limit_val])  # ORDER BY, LIMIT

        pool = _get_connection_pool()
        conn = None
        from_pool = False
        try:
            if pool:
                conn = pool.getconn()
                from_pool = True
            else:
                db_config = get_database_config()
                conn = db_config.get_connection(autocommit=True)
            conn.autocommit = True
            register_vector(conn)
            cur = conn.cursor()
            cur.execute(sql, params_for_sql)
            colnames = [d[0] for d in cur.description]
            rows = cur.fetchall()
            cur.close()

            results = []
            for row in rows:
                chunk = dict(zip(colnames, row))
                chunk["similarity"] = float(chunk.get("similarity", 0.0))
                results.append(chunk)
            return results
        except Exception as e:
            logger.warning("pgvector search failed, will fall back to client-side", error=str(e))
            raise
        finally:
            if conn:
                try:
                    if from_pool and pool:
                        pool.putconn(conn)
                    elif not conn.closed:
                        conn.close()
                except Exception:
                    pass

    def search_chunks(
        self,
        query_embedding: List[float],
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        project_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        source_types: Optional[List[str]] = None,
        match_threshold: float = 0.75,
        match_count: int = 6,
        return_below_threshold_for_entity: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search semantic chunks using vector similarity.
        Uses pgvector server-side search when available; falls back to client-side.
        """
        if not user_id:
            logger.warning("RAG search called without user_id - security risk, returning empty results")
            return []

        # Try pgvector server-side search first
        if PGVECTOR_AVAILABLE:
            try:
                results = self._search_chunks_pgvector(
                    query_embedding=query_embedding,
                    user_id=user_id,
                    organization_id=organization_id,
                    project_id=project_id,
                    experiment_id=experiment_id,
                    source_types=source_types,
                    match_threshold=match_threshold,
                    match_count=match_count,
                    apply_threshold=True,
                )
                if (
                    return_below_threshold_for_entity
                    and (project_id or experiment_id)
                    and len(results) == 0
                ):
                    results = self._search_chunks_pgvector(
                        query_embedding=query_embedding,
                        user_id=user_id,
                        organization_id=organization_id,
                        project_id=project_id,
                        experiment_id=experiment_id,
                        source_types=source_types,
                        match_threshold=match_threshold,
                        match_count=match_count,
                        apply_threshold=False,
                    )
                    if results:
                        logger.info(
                            "Returning below-threshold chunks for entity (pgvector)",
                            project_id=project_id,
                            experiment_id=experiment_id,
                            chunks_returned=len(results),
                        )
                logger.info(
                    "Vector search completed (pgvector)",
                    results_count=len(results),
                    match_threshold=match_threshold,
                )
                return results
            except (ConfigurationError, RuntimeError, Exception) as e:
                logger.warning("pgvector search failed, falling back to client-side", error=str(e))

        return self._search_chunks_client_side(
            query_embedding=query_embedding,
            user_id=user_id,
            organization_id=organization_id,
            project_id=project_id,
            experiment_id=experiment_id,
            source_types=source_types,
            match_threshold=match_threshold,
            match_count=match_count,
            return_below_threshold_for_entity=return_below_threshold_for_entity,
        )

    def _search_chunks_client_side(
        self,
        query_embedding: List[float],
        user_id: str,
        organization_id: Optional[str] = None,
        project_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        source_types: Optional[List[str]] = None,
        match_threshold: float = 0.75,
        match_count: int = 6,
        return_below_threshold_for_entity: bool = False,
    ) -> List[Dict[str, Any]]:
        """Client-side vector search (fallback when pgvector unavailable)."""
        try:
            query = self.db.client.table("semantic_chunks").select("*").not_.is_("embedding", "null")
            query = query.eq("created_by", user_id)
            if organization_id:
                query = query.eq("organization_id", organization_id)
            if project_id:
                query = query.eq("project_id", project_id)
            if experiment_id:
                query = query.eq("experiment_id", experiment_id)
            if source_types:
                query = query.in_("source_type", source_types)

            response = query.limit(1000).execute()
            chunks = response.data if response.data else []

            if not chunks:
                return []

            query_vec = np.array(query_embedding, dtype=np.float32)
            results = []

            for chunk in chunks:
                parsed_embedding = parse_embedding(chunk.get("embedding"))
                if parsed_embedding is None or len(parsed_embedding) != len(query_embedding):
                    continue
                chunk_vec = np.array(parsed_embedding, dtype=np.float32)
                dot_product = np.dot(query_vec, chunk_vec)
                norm_query = np.linalg.norm(query_vec)
                norm_chunk = np.linalg.norm(chunk_vec)
                similarity = (
                    float(dot_product / (norm_query * norm_chunk))
                    if (norm_query > 0 and norm_chunk > 0)
                    else 0.0
                )
                if similarity >= match_threshold:
                    chunk["similarity"] = similarity
                    results.append(chunk)

            results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
            results = results[:match_count]

            if (
                return_below_threshold_for_entity
                and (project_id or experiment_id)
                and len(chunks) > 0
                and len(results) == 0
            ):
                for chunk in chunks:
                    parsed_embedding = parse_embedding(chunk.get("embedding"))
                    if parsed_embedding is None or len(parsed_embedding) != len(query_embedding):
                        continue
                    chunk_vec = np.array(parsed_embedding, dtype=np.float32)
                    dot_product = np.dot(query_vec, chunk_vec)
                    norm_query = np.linalg.norm(query_vec)
                    norm_chunk = np.linalg.norm(chunk_vec)
                    similarity = (
                        float(dot_product / (norm_query * norm_chunk))
                        if (norm_query > 0 and norm_chunk > 0)
                        else 0.0
                    )
                    chunk["similarity"] = similarity
                    results.append(chunk)
                results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                results = results[:match_count]
                logger.info(
                    "Returning below-threshold chunks for entity",
                    project_id=project_id,
                    experiment_id=experiment_id,
                    chunks_returned=len(results),
                )

            logger.info(
                "Vector search completed (client-side)",
                results_count=len(results),
                match_threshold=match_threshold,
                chunks_checked=len(chunks),
            )
            return results
        except Exception as e:
            logger.error("Error searching chunks", error=str(e))
            return []
    
    def _hybrid_search_chunks_pgvector(
        self,
        query_embedding: List[float],
        query_text: str,
        user_id: str,
        organization_id: Optional[str] = None,
        project_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        source_types: Optional[List[str]] = None,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        match_threshold: float = 0.5,
        match_count: int = 10,
    ) -> List[Dict[str, Any]]:
        """Server-side hybrid search using pgvector + ts_rank."""
        if not PGVECTOR_AVAILABLE or register_vector is None:
            raise RuntimeError("pgvector not available")

        query_vec = np.array(query_embedding, dtype=np.float32)
        conditions = ["created_by = %s", "embedding IS NOT NULL"]
        params_for_sql: List[Any] = []

        if organization_id:
            conditions.append("organization_id = %s")
        if project_id:
            conditions.append("project_id = %s")
        if experiment_id:
            conditions.append("experiment_id = %s")
        if source_types:
            conditions.append("source_type = ANY(%s)")

        where_sql = " AND ".join(conditions)
        # ts_rank: normalize by dividing by max possible (1.0) - ts_rank can be > 1, we clamp
        # Use COALESCE to handle NULL fts or no match
        cols = "id, source_type, source_id, organization_id, project_id, experiment_id, chunk_index, content, metadata, created_at, created_by"
        sql = f"""
            WITH scored AS (
                SELECT {cols},
                   1 - (embedding <=> %s) AS vector_sim,
                   LEAST(1.0, COALESCE(ts_rank(fts, plainto_tsquery('english', %s)), 0) * 10) AS text_rank
                FROM semantic_chunks
                WHERE {where_sql}
            )
            SELECT *, (vector_sim * %s + text_rank * %s) AS combined_score
            FROM scored
            WHERE (vector_sim * %s + text_rank * %s) >= %s
            ORDER BY combined_score DESC, id
            LIMIT %s
        """
        params_for_sql = [query_vec, query_text]
        params_for_sql.append(user_id)
        if organization_id:
            params_for_sql.append(organization_id)
        if project_id:
            params_for_sql.append(project_id)
        if experiment_id:
            params_for_sql.append(experiment_id)
        if source_types:
            params_for_sql.append(source_types)
        # SELECT combined_score needs (vector_weight, text_weight),
        # WHERE clause needs them again plus match_threshold, then LIMIT
        params_for_sql.extend([
            vector_weight, text_weight,
            vector_weight, text_weight, match_threshold,
            match_count,
        ])

        pool = _get_connection_pool()
        conn = None
        from_pool = False
        try:
            if pool:
                conn = pool.getconn()
                from_pool = True
            else:
                db_config = get_database_config()
                conn = db_config.get_connection(autocommit=True)
            conn.autocommit = True
            register_vector(conn)
            cur = conn.cursor()
            cur.execute(sql, params_for_sql)
            colnames = [d[0] for d in cur.description]
            rows = cur.fetchall()
            cur.close()

            results = []
            for row in rows:
                chunk = dict(zip(colnames, row))
                chunk["vector_similarity"] = float(chunk.get("vector_sim", 0.0))
                chunk["text_rank"] = float(chunk.get("text_rank", 0.0))
                chunk["combined_score"] = float(chunk.get("combined_score", 0.0))
                results.append(chunk)
            return results
        finally:
            if conn:
                try:
                    if from_pool and pool:
                        pool.putconn(conn)
                    elif not conn.closed:
                        conn.close()
                except Exception:
                    pass

    def hybrid_search_chunks(
        self,
        query_embedding: List[float],
        query_text: str,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        project_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        source_types: Optional[List[str]] = None,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        match_threshold: float = 0.5,
        match_count: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector similarity and full-text search.
        Uses pgvector + ts_rank when available; falls back to client-side.
        """
        if not user_id:
            logger.warning("Hybrid RAG search called without user_id - security risk, returning empty results")
            return []

        if PGVECTOR_AVAILABLE:
            try:
                results = self._hybrid_search_chunks_pgvector(
                    query_embedding=query_embedding,
                    query_text=query_text,
                    user_id=user_id,
                    organization_id=organization_id,
                    project_id=project_id,
                    experiment_id=experiment_id,
                    source_types=source_types,
                    vector_weight=vector_weight,
                    text_weight=text_weight,
                    match_threshold=match_threshold,
                    match_count=match_count,
                )
                logger.info("Hybrid search completed (pgvector)", results_count=len(results))
                return results
            except (ConfigurationError, RuntimeError, Exception) as e:
                logger.warning("pgvector hybrid search failed, falling back to client-side", error=str(e))

        return self._hybrid_search_chunks_client_side(
            query_embedding=query_embedding,
            query_text=query_text,
            user_id=user_id,
            organization_id=organization_id,
            project_id=project_id,
            experiment_id=experiment_id,
            source_types=source_types,
            vector_weight=vector_weight,
            text_weight=text_weight,
            match_threshold=match_threshold,
            match_count=match_count,
        )

    def _hybrid_search_chunks_client_side(
        self,
        query_embedding: List[float],
        query_text: str,
        user_id: str,
        organization_id: Optional[str] = None,
        project_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        source_types: Optional[List[str]] = None,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        match_threshold: float = 0.5,
        match_count: int = 10,
    ) -> List[Dict[str, Any]]:
        """Client-side hybrid search (fallback)."""
        try:
            query = self.db.client.table("semantic_chunks").select("*").not_.is_("embedding", "null")
            query = query.eq("created_by", user_id)
            if organization_id:
                query = query.eq("organization_id", organization_id)
            if project_id:
                query = query.eq("project_id", project_id)
            if experiment_id:
                query = query.eq("experiment_id", experiment_id)
            if source_types:
                query = query.in_("source_type", source_types)

            response = query.limit(1000).execute()
            chunks = response.data if response.data else []
            if not chunks:
                return []

            query_words = set(re.findall(r"\b\w+\b", query_text.lower()))
            query_vec = np.array(query_embedding, dtype=np.float32)
            results = []

            for chunk in chunks:
                parsed_embedding = parse_embedding(chunk.get("embedding"))
                if parsed_embedding is None:
                    continue
                chunk_vec = np.array(parsed_embedding, dtype=np.float32)
                dot_product = np.dot(query_vec, chunk_vec)
                norm_query = np.linalg.norm(query_vec)
                norm_chunk = np.linalg.norm(chunk_vec)
                vector_sim = (
                    float(dot_product / (norm_query * norm_chunk))
                    if (norm_query > 0 and norm_chunk > 0)
                    else 0.0
                )
                content_words = set(re.findall(r"\b\w+\b", chunk.get("content", "").lower()))
                text_rank = (
                    min(1.0, len(query_words.intersection(content_words)) / len(query_words))
                    if query_words
                    else 0.0
                )
                combined_score = (vector_sim * vector_weight) + (text_rank * text_weight)
                if combined_score >= match_threshold:
                    chunk["vector_similarity"] = vector_sim
                    chunk["text_rank"] = text_rank
                    chunk["combined_score"] = combined_score
                    results.append(chunk)

            results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
            return results[:match_count]
        except Exception as e:
            logger.error("Error in hybrid search", error=str(e))
            return []

