"""
AWS Transcribe service for dictation/streaming transcription.

Default mode (Option 2 / direct): generates a presigned WebSocket URL
in-process using the existing AWS credentials. No extra Lambda or ECR needed.

Optional overrides:
- AWS_TRANSCRIBE_SERVICE_URL  -> HTTP POST to an external Transcribe service
- AWS_TRANSCRIBE_LAMBDA_FUNCTION_NAME -> boto3 Lambda invoke
"""
import datetime
import json
import os
import re
import uuid
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()
import structlog

from services.config_errors import ConfigurationError
from services.transcribe_presigned_url import generate_transcribe_url

logger = structlog.get_logger()

ENV_SERVICE_URL = "AWS_TRANSCRIBE_SERVICE_URL"
ENV_LAMBDA_FUNCTION = "AWS_TRANSCRIBE_LAMBDA_FUNCTION_NAME"
ENV_REGION = "AWS_TRANSCRIBE_REGION"
ENV_LANGUAGE_CODE = "AWS_TRANSCRIBE_LANGUAGE_CODE"
ENV_SAMPLE_RATE = "AWS_TRANSCRIBE_SAMPLE_RATE_HZ"
ENV_MEDIA_ENCODING = "AWS_TRANSCRIBE_MEDIA_ENCODING"

DEFAULT_REGION = "us-east-1"
DEFAULT_LANGUAGE = "en-US"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MEDIA_ENCODING = "pcm"


def _get_transcribe_config() -> Dict[str, Any]:
    """Read Transcribe-related env vars. Falls back to direct mode."""
    service_url = (os.getenv(ENV_SERVICE_URL) or "").strip()
    lambda_fn = (os.getenv(ENV_LAMBDA_FUNCTION) or "").strip()

    if service_url:
        mode = "http"
    elif lambda_fn:
        mode = "lambda"
    else:
        mode = "direct"

    return {
        "mode": mode,
        "service_url": service_url or None,
        "lambda_function": lambda_fn or None,
        "region": os.getenv(ENV_REGION) or os.getenv("AWS_REGION") or DEFAULT_REGION,
        "language_code": os.getenv(ENV_LANGUAGE_CODE, DEFAULT_LANGUAGE).strip(),
        "sample_rate_hz": int(os.getenv(ENV_SAMPLE_RATE, str(DEFAULT_SAMPLE_RATE))),
        "media_encoding": os.getenv(ENV_MEDIA_ENCODING, DEFAULT_MEDIA_ENCODING).strip(),
    }


def create_transcribe_session(
    user_id: str,
    session_id: str,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a Transcribe streaming session.

    Returns dict with session_id, stream_url (presigned wss://), and expires_at.
    The frontend opens a WebSocket to stream_url and streams raw mic audio.
    """
    options = options or {}
    cfg = _get_transcribe_config()

    language_code = options.get("language_code") or cfg["language_code"]
    sample_rate_hz = options.get("sample_rate_hz") or cfg["sample_rate_hz"]
    media_encoding = options.get("media_encoding") or cfg["media_encoding"]

    if cfg["mode"] == "http":
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "language_code": language_code,
            "sample_rate_hz": sample_rate_hz,
            "media_encoding": media_encoding,
        }
        return _create_session_via_http(cfg["service_url"], payload)

    if cfg["mode"] == "lambda":
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "language_code": language_code,
            "sample_rate_hz": sample_rate_hz,
            "media_encoding": media_encoding,
        }
        return _create_session_via_lambda(cfg, payload)

    return _create_session_direct(
        cfg=cfg,
        session_id=session_id,
        language_code=language_code,
        sample_rate_hz=sample_rate_hz,
        media_encoding=media_encoding,
    )


def _create_session_direct(
    *,
    cfg: Dict[str, Any],
    session_id: str,
    language_code: str,
    sample_rate_hz: int,
    media_encoding: str,
) -> Dict[str, Any]:
    """Generate a presigned WebSocket URL for Transcribe streaming in-process."""
    access_key = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    session_token = os.getenv("AWS_SESSION_TOKEN", "").strip()
    region = cfg["region"]

    if not access_key or not secret_key:
        raise ConfigurationError(
            "AWS Transcribe (direct mode) requires AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY to be set."
        )

    # AWS Transcribe requires session-id to be a UUID
    _UUID_RE = re.compile(r"^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$")
    transcribe_session_id = session_id if _UUID_RE.match(session_id) else str(uuid.uuid4())

    stream_url = generate_transcribe_url(
        access_key=access_key,
        secret_key=secret_key,
        session_token=session_token,
        region=region,
        language_code=language_code,
        media_encoding=media_encoding,
        sample_rate=sample_rate_hz,
        session_id=transcribe_session_id,
    )

    expires_at = (
        datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(seconds=300)
    ).isoformat()

    logger.info(
        "aws_transcribe_session_created",
        mode="direct",
        region=region,
        session_id=session_id,
        language_code=language_code,
        sample_rate_hz=sample_rate_hz,
    )

    return {
        "session_id": session_id,
        "stream_url": stream_url,
        "expires_at": expires_at,
    }


# ---------------------------------------------------------------------------
# Mode A / Mode B helpers (kept for optional overrides)
# ---------------------------------------------------------------------------

def _create_session_via_http(service_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST to existing Transcribe HTTP service."""
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                service_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        logger.error(
            "aws_transcribe_http_error",
            status=e.response.status_code,
            body=(e.response.text or "")[:500],
        )
        raise ConfigurationError(
            f"AWS Transcribe HTTP service error: {e.response.status_code} - "
            f"{e.response.text[:200]}"
        ) from e
    except httpx.RequestError as e:
        logger.error("aws_transcribe_request_error", error=str(e))
        raise ConfigurationError(f"AWS Transcribe request failed: {str(e)}") from e

    return _normalize_session_response(data, payload.get("session_id"))


def _create_session_via_lambda(cfg: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke Lambda via boto3."""
    import boto3

    fn_name = cfg["lambda_function"]
    if not fn_name:
        raise ConfigurationError(
            f"AWS Transcribe Lambda mode requires {ENV_LAMBDA_FUNCTION} to be set."
        )

    try:
        client = boto3.client("lambda", region_name=cfg["region"])
        result = client.invoke(
            FunctionName=fn_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload),
        )
    except Exception as e:
        logger.error("aws_transcribe_lambda_invoke_error", error=str(e))
        raise ConfigurationError(f"AWS Transcribe Lambda invoke failed: {str(e)}") from e

    raw = result.get("Payload")
    if raw is None:
        raise ConfigurationError("AWS Transcribe Lambda returned no payload")

    try:
        body = raw.read().decode("utf-8")
        data = json.loads(body)
    except (AttributeError, json.JSONDecodeError) as e:
        raise ConfigurationError(
            f"AWS Transcribe Lambda returned invalid JSON: {str(e)}"
        ) from e

    if "errorMessage" in data:
        raise ConfigurationError(
            f"AWS Transcribe Lambda error: {data.get('errorMessage', 'Unknown')}"
        )

    return _normalize_session_response(data, payload.get("session_id"))


def _normalize_session_response(data: Dict[str, Any], fallback_session_id: str) -> Dict[str, Any]:
    """Ensure response has session_id and stream_url."""
    session_id = data.get("session_id") or fallback_session_id
    stream_url = data.get("stream_url") or data.get("websocket_url") or data.get("url")
    if not stream_url:
        raise ConfigurationError(
            "AWS Transcribe service did not return stream_url (or websocket_url/url)"
        )
    return {
        "session_id": session_id,
        "stream_url": stream_url,
        "expires_at": data.get("expires_at"),
    }
