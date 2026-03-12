"""Trace service for persistent agent execution logging."""
import threading
import time
from typing import Optional, Dict, Any, List
from uuid import uuid4
import structlog
from services.db import SupabaseService

logger = structlog.get_logger()


class TraceService:
    """Service for logging agent execution traces to database."""
    
    def __init__(self, db_service: Optional[SupabaseService] = None):
        """
        Initialize trace service.
        
        Args:
            db_service: Optional SupabaseService instance. If not provided, creates a new one.
        """
        self.db = db_service if db_service else SupabaseService()
        logger.info("Trace service initialized")
    
    def create_run(
        self,
        run_id: str,
        organization_id: str,
        created_by: str,
        session_id: str,
        query: str,
        project_id: Optional[str] = None
    ) -> bool:
        """
        Create a new agent run record.
        
        Args:
            run_id: UUID string for the run
            organization_id: Organization ID
            created_by: User ID who created the run
            session_id: Session ID
            query: User query text
            project_id: Optional project ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            row = {
                "run_id": run_id,
                "organization_id": organization_id or "default",
                "created_by": created_by,
                "session_id": session_id,
                "query": query,
                "status": "running",
            }
            if project_id is not None:
                row["project_id"] = project_id
            self.db.client.table("agent_runs").insert(row).execute()
            logger.debug("agent_run created", run_id=run_id)
            return True
        except Exception as e:
            logger.debug("Failed to create agent run", run_id=run_id, error=str(e))
            return False
    
    def _do_log_event(
        self,
        run_id: str,
        node_name: str,
        event_type: str,
        payload: Dict[str, Any],
        latency_ms: Optional[int] = None,
    ) -> bool:
        """Internal: perform the actual Supabase insert."""
        try:
            row = {
                "run_id": run_id,
                "node_name": node_name,
                "event_type": event_type,
                "payload": payload if isinstance(payload, dict) else {"value": payload},
            }
            if latency_ms is not None:
                row["latency_ms"] = latency_ms
            self.db.client.table("agent_trace_events").insert(row).execute()
            logger.debug("trace_event logged", run_id=run_id, node_name=node_name, event_type=event_type)
            return True
        except Exception as e:
            err = str(e).lower()
            if "409" in err or "conflict" in err or "duplicate" in err:
                logger.debug("trace_event conflict (ignored)", run_id=run_id, node_name=node_name)
            else:
                logger.debug("Failed to log trace event", run_id=run_id, node_name=node_name, error=str(e))
            return False

    def log_event(
        self,
        run_id: str,
        node_name: str,
        event_type: str,
        payload: Dict[str, Any],
        latency_ms: Optional[int] = None,
        fire_and_forget: bool = True,
    ) -> bool:
        """
        Log a trace event for a node execution.
        When fire_and_forget=True (default), writes to DB in a background thread to avoid blocking the critical path.
        """
        if fire_and_forget:
            t = threading.Thread(
                target=self._do_log_event,
                args=(run_id, node_name, event_type, payload),
                kwargs={"latency_ms": latency_ms},
                daemon=True,
            )
            t.start()
            return True
        return self._do_log_event(run_id, node_name, event_type, payload, latency_ms)
    
    def update_run_status(
        self,
        run_id: str,
        status: str,
        final_confidence: Optional[float] = None,
        tool_used: Optional[str] = None,
        total_latency_ms: Optional[int] = None
    ) -> bool:
        """
        Update agent run status and final metrics.
        
        Args:
            run_id: UUID string for the run
            status: Final status (completed, failed)
            final_confidence: Optional final confidence score
            tool_used: Optional tool used (sql, rag, hybrid)
            total_latency_ms: Optional total execution time in milliseconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            update_data = {
                "status": status,
                "completed_at": "now()"
            }
            
            if final_confidence is not None:
                update_data["final_confidence"] = final_confidence
            
            if tool_used is not None:
                update_data["tool_used"] = tool_used
            
            if total_latency_ms is not None:
                update_data["total_latency_ms"] = total_latency_ms
            
            self.db.client.table("agent_runs").update(update_data).eq("run_id", run_id).execute()
            
            logger.debug(
                "agent_run updated",
                run_id=run_id,
                status=status
            )
            return True
            
        except Exception as e:
            # Don't break agent execution if trace logging fails
            logger.error(
                "Failed to update agent run",
                run_id=run_id,
                error=str(e)
            )
            return False
