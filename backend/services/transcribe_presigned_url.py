# Adapted from https://github.com/aws-samples/amazon-transcribe-streaming-python-websockets
# SPDX-License-Identifier: MIT-0
"""Generate presigned WebSocket URLs for AWS Transcribe streaming (SigV4)."""
import datetime
import hashlib
import hmac
import urllib.parse


def _hmac_sha256(data: str, key: bytes) -> bytes:
    return hmac.new(key, data.encode("utf-8"), hashlib.sha256).digest()


def _get_signature_key(key: str, datestamp: str, region: str, service: str) -> bytes:
    k_secret = ("AWS4" + key).encode("utf-8")
    k_date = _hmac_sha256(datestamp, k_secret)
    k_region = _hmac_sha256(region, k_date)
    k_service = _hmac_sha256(service, k_region)
    return _hmac_sha256("aws4_request", k_service)


def generate_transcribe_url(
    *,
    access_key: str,
    secret_key: str,
    session_token: str = "",
    region: str = "us-east-1",
    language_code: str = "en-US",
    media_encoding: str = "pcm",
    sample_rate: int = 16000,
    session_id: str = "",
) -> str:
    """
    Build a presigned wss:// URL for AWS Transcribe streaming.

    The URL is valid for 300 seconds (5 min). The frontend opens a WebSocket
    to this URL, streams raw mic audio, and receives transcript events.
    """
    service = "transcribe"
    host = f"transcribestreaming.{region}.amazonaws.com:8443"
    endpoint = f"wss://{host}"
    canonical_uri = "/stream-transcription-websocket"

    now = datetime.datetime.now(datetime.timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    datestamp = now.strftime("%Y%m%d")

    credential_scope_encoded = f"{datestamp}%2F{region}%2F{service}%2Faws4_request"
    credential_scope = f"{datestamp}/{region}/{service}/aws4_request"
    algorithm = "AWS4-HMAC-SHA256"
    signed_headers = "host"

    # Query string params must be in sorted order for SigV4
    qs = f"X-Amz-Algorithm={algorithm}"
    qs += f"&X-Amz-Credential={access_key}%2F{credential_scope_encoded}"
    qs += f"&X-Amz-Date={amz_date}"
    qs += "&X-Amz-Expires=300"
    if session_token:
        qs += f"&X-Amz-Security-Token={urllib.parse.quote(session_token, safe='')}"
    qs += f"&X-Amz-SignedHeaders={signed_headers}"
    qs += f"&language-code={language_code}"
    qs += f"&media-encoding={media_encoding}"
    qs += f"&sample-rate={sample_rate}"
    if session_id:
        qs += f"&session-id={session_id}"

    canonical_headers = f"host:{host}\n"
    payload_hash = hashlib.sha256(b"").hexdigest()

    canonical_request = (
        f"GET\n{canonical_uri}\n{qs}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    )
    hashed_request = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    string_to_sign = f"{algorithm}\n{amz_date}\n{credential_scope}\n{hashed_request}"

    signing_key = _get_signature_key(secret_key, datestamp, region, service)
    signature = hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    return f"{endpoint}{canonical_uri}?{qs}&X-Amz-Signature={signature}"
