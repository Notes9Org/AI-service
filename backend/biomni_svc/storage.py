"""S3 persistence for Biomni run outputs."""
import json
import os
import tempfile
import uuid
from typing import Optional

import structlog

logger = structlog.get_logger()


def upload_biomni_result_to_s3(
    result: str,
    session_id: str = "",
    user_id: str = "",
    metadata: Optional[dict] = None,
) -> Optional[str]:
    """
    Upload Biomni run result to S3 and return the object URL.

    Returns None if BIOMNI_S3_BUCKET is not configured.
    """
    bucket = os.getenv("BIOMNI_S3_BUCKET", "").strip()
    if not bucket:
        return None

    prefix = os.getenv("BIOMNI_S3_PREFIX", "biomni/results").strip().rstrip("/")
    key = f"{prefix}/{uuid.uuid4().hex}.json"

    try:
        import boto3
        from botocore.exceptions import ClientError

        payload = {
            "result": result,
            "session_id": session_id,
            "user_id": user_id,
            **(metadata or {}),
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(payload, f, indent=2)
            tmp_path = f.name

        try:
            s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
            s3.upload_file(tmp_path, bucket, key, ExtraArgs={"ContentType": "application/json"})
            url = f"s3://{bucket}/{key}"
            logger.info("biomni_result_uploaded_to_s3", bucket=bucket, key=key, url=url)
            return url
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except ClientError as e:
        logger.error("biomni_s3_upload_failed", bucket=bucket, key=key, error=str(e))
        return None
    except Exception as e:
        logger.error("biomni_s3_upload_error", error=str(e), exc_info=True)
        return None
