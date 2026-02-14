"""
Live transcription API using AWS Transcribe Streaming.

WebSocket endpoint: client sends binary PCM audio chunks, receives
JSON transcript events (partial and final). No S3; no storage.
Requires Bearer token via query param: ?token=<access_token>.
"""
import asyncio
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from services.auth import verify_token_for_websocket
from services.transcribe_streaming_service import stream_transcription

router = APIRouter(prefix="/transcribe", tags=["transcribe"])

WS_CLOSE_UNAUTHORIZED = 4401


@router.get("/stream", summary="Live speech-to-text (WebSocket)")
async def transcribe_stream_info() -> dict:
    """
    **WebSocket endpoint** for live transcription. Connect via `WS /transcribe/stream?token=<access_token>&language_code=en-US&sample_rate=16000`.

    - **token** (required): Supabase Auth access_token
    - **language_code**: e.g. en-US
    - **sample_rate**: 8000 or 16000 Hz
    - **media_encoding**: pcm

    Send binary PCM audio frames; receive JSON `{"event":"transcript","is_partial":bool,"text":"..."}`. Send text `end` to stop.
    """
    return {
        "endpoint": "WebSocket /transcribe/stream",
        "description": "Live speech-to-text. Connect with ?token=<access_token>, send PCM audio, receive transcript events.",
        "query_params": ["token", "language_code", "sample_rate", "media_encoding"],
    }


@router.websocket("/stream")
async def stream_transcription_ws(
    websocket: WebSocket,
    token: str = Query(..., description="Supabase Auth access_token (Bearer)"),
    language_code: str = Query("en-US", description="Language code (e.g. en-US)"),
    sample_rate: int = Query(16000, description="Audio sample rate in Hz (8000 or 16000)"),
    media_encoding: str = Query("pcm", description="Encoding (pcm)"),
) -> None:
    """
    **Audio / live transcription endpoint.** Connect via WebSocket with ?token=<access_token>, send raw PCM audio chunks (binary frames), receive real-time transcript events. Send text \"end\" or close the socket to stop.
    """
    await websocket.accept()
    try:
        current_user = verify_token_for_websocket(token)
    except Exception:
        await websocket.close(code=WS_CLOSE_UNAUTHORIZED, reason="Unauthorized")
        return
    await websocket.send_json({"event": "auth_ok", "user_id": current_user.user_id})
    audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=256)

    async def forward_transcripts():
        try:
            async for event in stream_transcription(
                audio_queue,
                language_code=language_code,
                media_sample_rate_hz=sample_rate,
                media_encoding=media_encoding,
            ):
                await websocket.send_json(event)
        except Exception as e:
            try:
                await websocket.send_json({"event": "error", "message": str(e)})
            except Exception:
                pass

    consumer = asyncio.create_task(forward_transcripts())

    try:
        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                break
            if "bytes" in message and message["bytes"]:
                try:
                    audio_queue.put_nowait(message["bytes"])
                except asyncio.QueueFull:
                    pass  # drop chunk if backlog too large
            elif "text" in message:
                if message["text"].strip().lower() == "end":
                    await audio_queue.put(None)
                    break
    finally:
        try:
            audio_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        consumer.cancel()
        try:
            await consumer
        except asyncio.CancelledError:
            pass
