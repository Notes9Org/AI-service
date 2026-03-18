"""Session/conversation storage for Biomni (S3-backed)."""
import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger()

SESSION_PREFIX = "biomni/sessions"
RUN_PREFIX = "biomni/runs"


def _get_s3_client():
    import boto3
    return boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))


def _bucket() -> Optional[str]:
    return os.getenv("BIOMNI_S3_BUCKET", "").strip() or None


def create_session(user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a new session. Uses S3 to store session metadata.
    Returns session dict with id, user_id, created_at.
    """
    sid = session_id or str(uuid.uuid4())
    session = {
        "id": sid,
        "user_id": user_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "runs": [],
    }
    bucket = _bucket()
    if bucket:
        try:
            key = f"{SESSION_PREFIX}/{user_id}/{sid}.json"
            _get_s3_client().put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(session, indent=2),
                ContentType="application/json",
            )
        except Exception as e:
            logger.warning("biomni_session_create_failed", error=str(e))
    return session


def add_run(
    session_id: str,
    user_id: str,
    query: str,
    result: str,
    steps: List[str],
    clarifications: Optional[List[dict]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Add a run to a session. Returns the run entry or None.
    """
    bucket = _bucket()
    if not bucket:
        return None
    run_entry = {
        "id": str(uuid.uuid4()),
        "query": query,
        "result": result,
        "steps": steps,
        "clarifications": clarifications or [],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    try:
        key = f"{SESSION_PREFIX}/{user_id}/{session_id}.json"
        try:
            resp = _get_s3_client().get_object(Bucket=bucket, Key=key)
            session = json.loads(resp["Body"].read().decode())
        except Exception:
            session = {"id": session_id, "user_id": user_id, "created_at": run_entry["timestamp"], "runs": []}
        session.setdefault("runs", []).append(run_entry)
        _get_s3_client().put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(session, indent=2),
            ContentType="application/json",
        )
        return run_entry
    except Exception as e:
        logger.warning("biomni_session_add_run_failed", error=str(e))
        return None


def get_session(session_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Get a session by ID. Returns None if not found."""
    bucket = _bucket()
    if not bucket:
        return None
    try:
        key = f"{SESSION_PREFIX}/{user_id}/{session_id}.json"
        resp = _get_s3_client().get_object(Bucket=bucket, Key=key)
        return json.loads(resp["Body"].read().decode())
    except Exception:
        return None


def list_sessions(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """List sessions for a user. Returns list of session summaries."""
    bucket = _bucket()
    if not bucket:
        return []
    try:
        prefix = f"{SESSION_PREFIX}/{user_id}/"
        resp = _get_s3_client().list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=limit)
        sessions = []
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            sid = key.split("/")[-1].replace(".json", "")
            try:
                r = _get_s3_client().get_object(Bucket=bucket, Key=key)
                s = json.loads(r["Body"].read().decode())
                sessions.append({"id": s.get("id", sid), "created_at": s.get("created_at"), "runs_count": len(s.get("runs", []))})
            except Exception:
                sessions.append({"id": sid, "created_at": None, "runs_count": 0})
        sessions.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        return sessions
    except Exception as e:
        logger.warning("biomni_session_list_failed", error=str(e))
        return []
