"""SQL service with LLM-generated queries for safe execution."""
import time
import re
from typing import Dict, Any, Optional, List, Tuple
import structlog
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

from services.db import SupabaseService
from services.config import get_database_config, ConfigurationError
from agents.services.llm_client import LLMClient, LLMError
from agents.services.db_schema import USER_FACING_SCHEMA
from agents.prompt_loader import load_prompt

logger = structlog.get_logger()

# Pool size: min 1, max 20 connections
POOL_MIN_CONN = 1
POOL_MAX_CONN = 20


class SQLService:
    """Service for generating and executing SQL queries using LLM."""

    def __init__(self, db_service: Optional[SupabaseService] = None, llm_client: Optional[LLMClient] = None):
        """
        Initialize SQL service.

        Args:
            db_service: Optional SupabaseService instance. If not provided, creates a new one.
            llm_client: Optional LLMClient instance. If not provided, creates a new one.
        """
        self.db = db_service if db_service else SupabaseService()
        self.llm_client = llm_client if llm_client else LLMClient()

        self._db_config = get_database_config()
        self._pg_pool: Optional[pool.ThreadedConnectionPool] = None

        logger.info("SQL service initialized")
    
    def _get_pg_pool(self) -> pool.ThreadedConnectionPool:
        """Get or create the connection pool."""
        if self._pg_pool is not None:
            return self._pg_pool
        try:
            args, kwargs = self._db_config.get_pool_connection_params()
            self._pg_pool = pool.ThreadedConnectionPool(
                POOL_MIN_CONN,
                POOL_MAX_CONN,
                *args,
                **kwargs,
            )
            logger.info("PostgreSQL connection pool created", minconn=POOL_MIN_CONN, maxconn=POOL_MAX_CONN)
            return self._pg_pool
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(
                f"Database service is not available: Failed to create connection pool. Error: {str(e)}"
            ) from e

    def _get_pg_connection(self, force_new: bool = False):
        """
        Get a connection from the pool. Autocommit is enabled for read-only queries.
        Caller must return the connection to the pool via _return_pg_connection.
        """
        if force_new and self._pg_pool:
            try:
                self._pg_pool.closeall()
            except Exception:
                pass
            self._pg_pool = None
        p = self._get_pg_pool()
        conn = p.getconn()
        conn.autocommit = True
        return conn

    def _return_pg_connection(self, conn, close: bool = False):
        """Return a connection to the pool. If close=True, close it instead (e.g. after error)."""
        if conn is None:
            return
        try:
            if close:
                conn.close()
            else:
                self._pg_pool.putconn(conn)
        except Exception as e:
            logger.warning("Error returning connection to pool", error=str(e))
            try:
                conn.close()
            except Exception:
                pass
    
    def _validate_sql_safety(self, sql: str, scope: Dict[str, Optional[str]]) -> Tuple[bool, str]:
        """
        Validate SQL query is safe to execute.
        
        Validates that query is read-only (SELECT only).
        Note: User_id filtering is enforced in generate_sql, not validated here.
        
        Returns:
            (is_safe, error_message)
        """
        sql_upper = sql.upper().strip()
        
        # Check for dangerous operations
        dangerous_keywords = [
            "DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE", "INSERT", 
            "UPDATE", "GRANT", "REVOKE", "EXEC", "EXECUTE"
        ]
        
        for keyword in dangerous_keywords:
            if f" {keyword} " in sql_upper or sql_upper.startswith(keyword):
                return False, f"Dangerous operation detected: {keyword}. Only SELECT queries are allowed."
        
        # Must be a SELECT query
        if not sql_upper.startswith("SELECT"):
            return False, "Only SELECT queries are allowed."
        
        # NO ORGANIZATION_ID VALIDATION - Users have complete access to all data
        
        return True, ""
    
    def generate_sql(
        self,
        query: str,
        user_id: Optional[str] = None,
        normalized_query: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None,
        scope: Optional[Dict[str, Optional[str]]] = None
    ) -> str:
        """
        Generate SQL query using LLM based on user query and database schema.
        
        Args:
            query: Original user query
            user_id: User ID for filtering (REQUIRED for security - filters by created_by)
            normalized_query: Normalized query text
            entities: Extracted entities from normalization (used for query generation)
            scope: Access scope (deprecated - ignored, not used for filtering)
            
        Returns:
            Generated SQL query string
        """
        # SECURITY: Always filter by user_id (created_by) to ensure users only see their own data
        if not user_id:
            raise ValueError("user_id is required for SQL generation - security requirement")
        
        # Build concise entities section
        entities_text = ""
        entity_filters = []
        if entities and isinstance(entities, dict):
            entities_text = "\n".join([f"- {k}: {v}" for k, v in list(entities.items())[:15]])
            # Concise filter hints
            exp_ids = list(entities.get("experiment_ids") or []) if isinstance(entities.get("experiment_ids"), list) else []
            if not exp_ids and entities.get("experiment_id"):
                eid = entities["experiment_id"]
                exp_ids = [eid] if not isinstance(eid, list) else eid
            proj_ids = list(entities.get("project_ids") or []) if isinstance(entities.get("project_ids"), list) else []
            if not proj_ids and entities.get("project_id"):
                pid = entities["project_id"]
                proj_ids = [pid] if not isinstance(pid, list) else pid
            if exp_ids:
                uuid_list = ", ".join([f"'{e}'::uuid" for e in exp_ids[:10]])
                entity_filters.append(f"Filter experiments: e.id IN ({uuid_list})")
            if proj_ids:
                uuid_list = ", ".join([f"'{p}'::uuid" for p in proj_ids[:10]])
                entity_filters.append(f"Filter projects: p.id IN ({uuid_list})")
            for key in ("experiment_names", "project_names"):
                vals = entities.get(key)
                if vals and isinstance(vals, list):
                    names = ", ".join([str(v)[:50] for v in vals[:5]])
                    col = "e.name" if "experiment" in key else "p.name"
                    entity_filters.append(f"Filter by {key} ({names}): use REPLACE(LOWER({col}), '_', ' ') ILIKE '%'||REPLACE(LOWER('<name>'), '_', ' ')||'%'")
            protocol_names = entities.get("protocol_names")
            if protocol_names and isinstance(protocol_names, list):
                names = ", ".join([str(n)[:80] for n in protocol_names[:5]])
                entity_filters.append(
                    f"Filter protocols by name ({names}): use protocols table with "
                    "REPLACE(LOWER(protocols.name), '_', ' ') ILIKE '%'||REPLACE(LOWER('<name>'), '_', ' ')||'%'"
                )
            lab_note_titles = entities.get("lab_note_titles")
            if lab_note_titles and isinstance(lab_note_titles, list):
                titles = ", ".join([str(t)[:80] for t in lab_note_titles[:5]])
                entity_filters.append(
                    f"Filter lab_notes by title ({titles}): use lab_notes table with "
                    "REPLACE(LOWER(lab_notes.title), '_', ' ') ILIKE '%'||REPLACE(LOWER('<title>'), '_', ' ')||'%'"
                )
            if entities.get("person_names"):
                entity_filters.append("Filter by person: JOIN profiles pr ON e.created_by=pr.id, use CONCAT(pr.first_name,' ',pr.last_name) ILIKE '%name%'")

        entity_section = "\n".join(entity_filters) if entity_filters else ""
        normalized_section = f"**Normalized:** {normalized_query}\n" if normalized_query else ""

        prompt_template = load_prompt("sql", "generate_query")
        prompt = prompt_template.format(
            query=query,
            normalized_section=normalized_section,
            user_id=user_id,
            entities_text=entities_text or "None",
            entity_section=entity_section,
            schema=USER_FACING_SCHEMA,
        )

        try:
            # Use text completion for SQL (not JSON). Use SQL-specific model when configured (e.g. Bedrock).
            model = getattr(self.llm_client, "chat_model_id_sql", None) or self.llm_client.default_deployment
            sql = self.llm_client.complete_text(
                prompt=prompt,
                model=model,
                temperature=0.0  # Deterministic for SQL
            )
            
            # Clean up SQL (remove markdown code blocks if present)
            sql = sql.strip()
            if sql.startswith("```"):
                # Remove markdown code blocks
                sql = re.sub(r'^```(?:sql)?\s*', '', sql, flags=re.IGNORECASE)
                sql = re.sub(r'\s*```$', '', sql)
            sql = sql.strip()
            
            # Remove trailing semicolon if present (we'll add it if needed)
            if sql.endswith(';'):
                sql = sql[:-1]
            
            # SECURITY: Reject SQL that does not filter by user_id — never return other users' data
            if user_id and user_id not in sql:
                logger.error("SQL rejected: missing user_id filter", user_id_preview=user_id[:8] + "...")
                raise ValueError(
                    "Generated SQL does not filter by user_id. Refusing to execute — users must only see their own data."
                )
            
            logger.info("SQL generated", query_length=len(sql), sql_preview=sql[:100], sql_full=sql)
            
            return sql
            
        except LLMError as e:
            logger.error("SQL generation failed", error=str(e))
            raise ValueError(f"Failed to generate SQL: {str(e)}")
    
    def execute_sql(
        self,
        sql: str,
        scope: Dict[str, Optional[str]]
    ) -> Dict[str, Any]:
        """
        Execute SQL query safely.
        
        SECURITY: SQL should already be filtered by user_id (created_by) from generate_sql.
        Only validates that query is read-only (SELECT only).
        
        Args:
            sql: SQL query to execute (should already include user_id filtering)
            scope: Access scope (deprecated - not used for filtering)
            
        Returns:
            Dict with data, row_count, execution_time_ms, or error
        """
        start_time = time.time()
        conn = None
        try:
            is_safe, error_msg = self._validate_sql_safety(sql, scope)
            if not is_safe:
                raise ValueError(f"SQL safety validation failed: {error_msg}")

            conn = self._get_pg_connection(force_new=False)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(sql)
            rows = cursor.fetchall()
            data = [dict(row) for row in rows]
            row_count = len(data)
            cursor.close()
            self._return_pg_connection(conn)
            conn = None

            execution_time_ms = (time.time() - start_time) * 1000
            logger.info(
                "SQL executed successfully",
                row_count=row_count,
                execution_time_ms=round(execution_time_ms, 2),
                sql_full=sql,
            )
            return {
                "data": data,
                "row_count": row_count,
                "execution_time_ms": round(execution_time_ms, 2),
            }

        except psycopg2.Error as e:
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
                self._return_pg_connection(conn, close=True)
                conn = None
            execution_time_ms = (time.time() - start_time) * 1000
            error_str = str(e)
            if "transaction is aborted" in error_str.lower():
                logger.warning("Connection had aborted transaction", error=error_str)
                raise ConfigurationError(
                    f"Database service is not available: Connection has aborted transaction. Error: {error_str}"
                ) from e
            logger.error("SQL execution failed (PostgreSQL error)", error=str(e), execution_time_ms=round(execution_time_ms, 2))
            raise ConfigurationError(f"Database service is not available: SQL execution failed. Error: {error_str}") from e
        except Exception as e:
            if conn:
                self._return_pg_connection(conn, close=True)
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error("SQL execution failed", error=str(e), execution_time_ms=round(execution_time_ms, 2))
            return {
                "data": [],
                "row_count": 0,
                "error": str(e),
                "execution_time_ms": round(execution_time_ms, 2),
            }
    
    def generate_and_execute(
        self,
        query: str,
        user_id: Optional[str] = None,
        normalized_query: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None,
        scope: Optional[Dict[str, Optional[str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate SQL and execute it in one call.
        
        Args:
            query: Original user query
            user_id: User ID for filtering (REQUIRED for security)
            normalized_query: Normalized query text
            entities: Extracted entities
            scope: Access scope (deprecated)
            
        Returns:
            Dict with data, row_count, execution_time_ms, or error
        """
        try:
            # Generate SQL
            sql = self.generate_sql(
                query=query,
                user_id=user_id,
                normalized_query=normalized_query,
                entities=entities,
                scope=scope
            )
            
            # Execute SQL
            result = self.execute_sql(sql, scope)
            
            # Add generated SQL to result for debugging
            result["generated_sql"] = sql
            
            return result
            
        except Exception as e:
            logger.error("Generate and execute failed", error=str(e))
            return {
                "data": [],
                "row_count": 0,
                "error": str(e),
                "execution_time_ms": 0
            }
    
    def __del__(self):
        """Close connection pool on cleanup."""
        try:
            if hasattr(self, "_pg_pool") and self._pg_pool:
                self._pg_pool.closeall()
                self._pg_pool = None
        except Exception:
            pass
    