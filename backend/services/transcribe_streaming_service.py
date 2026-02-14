"""
Live transcription via AWS Transcribe Streaming (HTTP/2 event stream).

Uses the amazon-transcribe SDK which calls StartStreamTranscription against
transcribestreaming.{region}.amazonaws.com. No S3; audio is streamed and
transcript events are streamed back.
"""
import asyncio
import os
from typing import AsyncIterator, Optional

import structlog

logger = structlog.get_logger()

# Lazy import to avoid requiring amazon_transcribe when not using transcribe
def _get_client():
    from amazon_transcribe.client import TranscribeStreamingClient
    region = os.getenv("AWS_REGION", "us-east-1")
    return TranscribeStreamingClient(region=region)


async def stream_transcription(
    audio_queue: "asyncio.Queue[Optional[bytes]]",
    *,
    language_code: str = "en-US",
    media_sample_rate_hz: int = 16000,
    media_encoding: str = "pcm",
) -> AsyncIterator[dict]:
    """
    Consume audio chunks from `audio_queue` and yield transcript events.

    The client should put raw PCM bytes into the queue. Put `None` to signal
    end of stream (then stop putting more items).

    Yields dicts: {"event": "transcript", "is_partial": bool, "text": str}
    """
    from amazon_transcribe.model import TranscriptEvent

    client = _get_client()
    stream = await client.start_stream_transcription(
        language_code=language_code,
        media_sample_rate_hz=media_sample_rate_hz,
        media_encoding=media_encoding,
    )

    async def send_audio():
        try:
            while True:
                chunk = await audio_queue.get()
                if chunk is None:
                    await stream.input_stream.end_stream()
                    break
                if chunk:
                    await stream.input_stream.send_audio_event(audio_chunk=chunk)
        except asyncio.CancelledError:
            try:
                await stream.input_stream.end_stream()
            except Exception:
                pass
        except Exception as e:
            logger.warning("transcribe send_audio error", error=str(e))

    sender = asyncio.create_task(send_audio())
    try:
        async for event in stream.output_stream:
            if isinstance(event, TranscriptEvent):
                for result in event.transcript.results:
                    for alt in (result.alternatives or []):
                        text = (alt.transcript or "").strip()
                        if text:
                            yield {
                                "event": "transcript",
                                "is_partial": result.is_partial if result.is_partial is not None else True,
                                "text": text,
                            }
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.warning("transcribe output_stream error", error=str(e))
    finally:
        sender.cancel()
        try:
            await sender
        except asyncio.CancelledError:
            pass
