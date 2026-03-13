from typing import Optional, List, Dict, Any
import structlog

from services.config import get_supabase_config

logger = structlog.get_logger()


class SupabaseService:

    def __init__(self):
        config = get_supabase_config()
        self.client = config.get_client()
        logger.info("Supabase client initialized", url=config.url)

    def get_pending_jobs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get pending chunk jobs from the database"""
        try:
            response = self.client.table("chunk_jobs")\
                .select("*")\
                .eq("status", "pending")\
                .order("created_at", desc=False)\
                .limit(limit)\
                .execute()
            
            return response.data if response.data else []
        except Exception as e:
            logger.error("Error getting pending jobs", error=str(e))
            return []
    
    def get_failed_jobs(self, limit: int = 100, max_retries: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get failed chunk jobs from the database"""
        try:
            query = self.client.table("chunk_jobs")\
                .select("*")\
                .eq("status", "failed")\
                .order("created_at", desc=False)
            
            if max_retries is not None:
                query = query.lte("retry_count", max_retries)
            
            response = query.limit(limit).execute()
            
            return response.data if response.data else []
        except Exception as e:
            logger.error("Error getting failed jobs", error=str(e))
            return []
    
    def reset_jobs_to_pending(self, job_ids: List[str]) -> int:
        """Reset failed jobs to pending status for retry. Returns number of jobs reset."""
        if not job_ids:
            return 0
        
        try:
            response = self.client.table("chunk_jobs")\
                .update({
                    "status": "pending",
                    "error_message": None,
                    "processed_at": None
                })\
                .in_("id", job_ids)\
                .eq("status", "failed")\
                .execute()
            
            count = len(response.data) if response.data else 0
            logger.info("Jobs reset to pending", count=count, job_ids=job_ids)
            return count
        except Exception as e:
            logger.error("Error resetting jobs to pending", error=str(e), job_ids=job_ids)
            return 0
    
    def reset_all_failed_jobs_to_pending(self, max_retries: Optional[int] = None) -> int:
        """Reset all failed jobs to pending status. Returns number of jobs reset."""
        try:
            query = self.client.table("chunk_jobs")\
                .update({
                    "status": "pending",
                    "error_message": None,
                    "processed_at": None
                })\
                .eq("status", "failed")
            
            if max_retries is not None:
                query = query.lte("retry_count", max_retries)
            
            # Note: Supabase doesn't return count directly, so we need to query first
            failed_jobs = self.get_failed_jobs(limit=10000, max_retries=max_retries)
            if not failed_jobs:
                return 0
            
            job_ids = [job["id"] for job in failed_jobs]
            return self.reset_jobs_to_pending(job_ids)
        except Exception as e:
            logger.error("Error resetting all failed jobs", error=str(e))
            return 0
    
    def update_job_status(self, job_id: str, status: str, error_message: Optional[str] = None) -> bool:
        """Update the status of a chunk job"""
        try:
            update_data = {
                "status": status,
                "processed_at": "now()",
            }
            if error_message:
                # Get current retry count
                current_job = self.client.table("chunk_jobs").select("retry_count").eq("id", job_id).single().execute()

                update_data["error_message"] = error_message
                update_data["retry_count"] = (current_job.data.get("retry_count", 0) + 1)
            
            self.client.table("chunk_jobs").update(update_data).eq("id", job_id).execute()
            logger.info("Job status updated", job_id=job_id, status=status)
            return True
        
        except Exception as e:
            logger.error("Error updating job status", error=str(e), job_id=job_id)
            return False

    def delete_chunks(self, source_type: str, source_id: str) -> bool:
        """Delete all chunks for a given source"""
        try:
            self.client.table("semantic_chunks").delete().eq("source_type", source_type).eq("source_id", source_id).execute()
            logger.info("Chunks deleted", source_type=source_type, source_id=source_id)
            return True
        except Exception as e:
            logger.error("Error deleting chunks", error=str(e), source_type=source_type, source_id=source_id)
            return False
    
    def insert_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Insert a list of chunks into the database"""
        if not chunks:
            return True
        try:
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                self.client.table("semantic_chunks").insert(batch).execute()
                logger.info("Chunks inserted", chunks=batch)
            return True
        except Exception as e:
            logger.error("Error inserting chunks", error=str(e), chunks=chunks)
            return False

    def get_semantic_chunks_page(self, offset: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch a page of semantic chunks (id, content) for re-embedding."""
        try:
            response = self.client.table("semantic_chunks")\
                .select("id, content")\
                .not_.is_("content", "null")\
                .order("id")\
                .range(offset, offset + limit - 1)\
                .execute()
            return response.data if response.data else []
        except Exception as e:
            logger.error("Error fetching semantic chunks page", error=str(e), offset=offset, limit=limit)
            return []

    def update_chunk_embedding(self, chunk_id: str, embedding: List[float]) -> bool:
        """Update a single chunk's embedding (e.g. after switching embedder)."""
        try:
            self.client.table("semantic_chunks").update({"embedding": embedding}).eq("id", chunk_id).execute()
            return True
        except Exception as e:
            logger.error("Error updating chunk embedding", error=str(e), chunk_id=chunk_id)
            return False

    def get_source_display_names(
        self, chunks: List[Dict[str, Any]]
    ) -> Dict[tuple, str]:
        """
        Resolve (source_type, source_id) to human-readable display names.
        Returns a dict keyed by (source_type, source_id) -> display name (e.g. "PCR Protocol", "Vaccine Project").
        """
        out: Dict[tuple, str] = {}
        if not chunks:
            return out
        by_type: Dict[str, List[str]] = {}
        for c in chunks:
            if not isinstance(c, dict):
                continue
            st = c.get("source_type")
            sid = c.get("source_id")
            if not st or not sid:
                continue
            by_type.setdefault(st, []).append(sid)
        for st, ids in by_type.items():
            ids = list(dict.fromkeys(ids))
            if not ids:
                continue
            try:
                if st == "lab_note":
                    r = self.client.table("lab_notes").select("id, title").in_("id", ids).execute()
                    for row in (r.data or []):
                        out[(st, str(row["id"]))] = row.get("title") or "Lab note"
                elif st == "protocol":
                    r = self.client.table("protocols").select("id, name").in_("id", ids).execute()
                    for row in (r.data or []):
                        out[(st, str(row["id"]))] = row.get("name") or "Protocol"
                elif st == "report":
                    r = self.client.table("reports").select("id, title").in_("id", ids).execute()
                    for row in (r.data or []):
                        out[(st, str(row["id"]))] = row.get("title") or "Report"
                elif st == "literature_review":
                    r = self.client.table("literature_reviews").select("id, title").in_("id", ids).execute()
                    for row in (r.data or []):
                        out[(st, str(row["id"]))] = row.get("title") or "Literature review"
                elif st == "experiment_summary" or st == "experiment":
                    r = self.client.table("experiments").select("id, name").in_("id", ids).execute()
                    for row in (r.data or []):
                        out[(st, str(row["id"]))] = row.get("name") or "Experiment"
                else:
                    for sid in ids:
                        out[(st, str(sid))] = st.replace("_", " ").title()
            except Exception as e:
                logger.warning("get_source_display_names failed for type %s", st, error=str(e))
                for sid in ids:
                    out[(st, str(sid))] = st.replace("_", " ").title()
        return out