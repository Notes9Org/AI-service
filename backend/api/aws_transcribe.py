"""
AWS Transcribe dictation session API.

POST /AWS_transcribe: Create a session for streaming dictation.
Returns session_id and stream_url for the frontend to open a WebSocket and stream raw mic audio.
Requires Bearer token (Supabase Auth).
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

import structlog

from services.auth import CurrentUser, get_current_user
from services.aws_transcribe_service import create_transcribe_session
from services.config_errors import ConfigurationError

logger = structlog.get_logger()
router = APIRouter(prefix="/AWS_transcribe", tags=["aws_transcribe"])


class AwsTranscribeRequest(BaseModel):
    """Request schema for POST /AWS_transcribe."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "dictation-session-123",
                "language_code": "en-US",
                "sample_rate_hz": 16000,
                "media_encoding": "pcm",
            }
        }
    )
    session_id: str = Field(..., description="Logical dictation session identifier")
    language_code: Optional[str] = Field(None, description="Language code (e.g. en-US)")
    sample_rate_hz: Optional[int] = Field(None, description="Audio sample rate in Hz")
    media_encoding: Optional[str] = Field(None, description="Media encoding (e.g. pcm, flac)")


class AwsTranscribeSessionResponse(BaseModel):
    """Response schema for POST /AWS_transcribe."""

    session_id: str = Field(..., description="Session identifier")
    stream_url: str = Field(..., description="WebSocket URL for streaming audio and receiving transcripts")
    expires_at: Optional[str] = Field(None, description="ISO timestamp when the session expires")


@router.post("", response_model=AwsTranscribeSessionResponse, summary="Create Transcribe session")
async def create_aws_transcribe_session(
    request: AwsTranscribeRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> AwsTranscribeSessionResponse:
    """
    Create an AWS Transcribe session for streaming dictation.

    Returns a stream_url that the frontend uses to open a WebSocket, stream raw mic audio,
    and receive live transcripts. No frontend changes required if it already supports
    this flow.
    """
    user_id = current_user.user_id
    logger.info(
        "aws_transcribe_session_request",
        user_id=user_id,
        session_id=request.session_id,
    )
    options = {}
    if request.language_code is not None:
        options["language_code"] = request.language_code
    if request.sample_rate_hz is not None:
        options["sample_rate_hz"] = request.sample_rate_hz
    if request.media_encoding is not None:
        options["media_encoding"] = request.media_encoding

    try:
        result = create_transcribe_session(
            user_id=user_id,
            session_id=request.session_id,
            options=options if options else None,
        )
    except ConfigurationError as e:
        logger.warning("aws_transcribe_config_error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("aws_transcribe_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=502, detail=f"AWS Transcribe failed: {str(e)}")

    return AwsTranscribeSessionResponse(
        session_id=result["session_id"],
        stream_url=result["stream_url"],
        expires_at=result.get("expires_at"),
    )
