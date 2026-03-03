"""Test AWS Transcribe streaming with microphone input."""
import asyncio
import hashlib
import hmac
import json
import struct
import sys
import wave
from binascii import crc32

import pyaudio
import websockets

# ---- Paste your stream_url from Step 1 here ----
STREAM_URL = "wss://transcribestreaming.us-east-1.amazonaws.com:8443/stream-transcription-websocket?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIASYJBI62R74EBX2OZ%2F20260303%2Fus-east-1%2Ftranscribe%2Faws4_request&X-Amz-Date=20260303T171505Z&X-Amz-Expires=300&X-Amz-SignedHeaders=host&language-code=en-US&media-encoding=pcm&sample-rate=16000&session-id=6aab2a33-7d04-441c-9409-47d8255a9680&X-Amz-Signature=205386b16bfbca54ae1c0a2e86f024a09eccc4b7321f04d783e81212d00f2db9"

# Audio settings (must match what you passed to POST /AWS_transcribe)
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION_MS = 100  # send 100ms chunks
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
FORMAT = pyaudio.paInt16


def create_audio_event(audio_bytes: bytes) -> bytes:
    """Wrap raw audio in AWS event stream encoding."""
    headers = bytearray()

    def add_header(name: str, value: str):
        name_bytes = name.encode("utf-8")
        value_bytes = value.encode("utf-8")
        headers.extend(len(name_bytes).to_bytes(1, "big"))
        headers.extend(name_bytes)
        headers.append(7)  # string type
        headers.extend(len(value_bytes).to_bytes(2, "big"))
        headers.extend(value_bytes)

    add_header(":content-type", "application/octet-stream")
    add_header(":event-type", "AudioEvent")
    add_header(":message-type", "event")

    headers_len = len(headers)
    payload = audio_bytes
    total_len = 4 + 4 + 4 + headers_len + len(payload) + 4  # prelude + prelude_crc + headers + payload + msg_crc

    prelude = struct.pack(">II", total_len, headers_len)
    prelude_crc = struct.pack(">I", crc32(prelude) & 0xFFFFFFFF)

    message = prelude + prelude_crc + bytes(headers) + payload
    message_crc = struct.pack(">I", crc32(message) & 0xFFFFFFFF)

    return message + message_crc


def decode_event(data: bytes):
    """Decode AWS event stream response."""
    total_len = struct.unpack(">I", data[0:4])[0]
    headers_len = struct.unpack(">I", data[4:8])[0]

    # Parse headers
    headers = {}
    offset = 12  # after prelude (8) + prelude_crc (4)
    end = offset + headers_len
    while offset < end:
        name_len = data[offset]
        offset += 1
        name = data[offset:offset + name_len].decode("utf-8")
        offset += name_len
        value_type = data[offset]
        offset += 1
        if value_type == 7:  # string
            val_len = struct.unpack(">H", data[offset:offset + 2])[0]
            offset += 2
            value = data[offset:offset + val_len].decode("utf-8")
            offset += val_len
        else:
            value = None
            offset += 1
        headers[name] = value

    # Payload is between headers and message CRC
    payload_data = data[12 + headers_len:total_len - 4]
    try:
        payload = json.loads(payload_data) if payload_data else {}
    except json.JSONDecodeError:
        payload = {}

    return headers, payload


async def mic_stream(websocket):
    """Capture mic audio and send to Transcribe via WebSocket."""
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )
    print("🎙️  Speak now... (Ctrl+C to stop)")

    try:
        while True:
            audio_chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            event = create_audio_event(audio_chunk)
            await websocket.send(event)
            await asyncio.sleep(0.001)
    except (KeyboardInterrupt, websockets.exceptions.ConnectionClosed):
        pass
    finally:
        # Send empty event to signal end of stream
        await websocket.send(create_audio_event(b""))
        stream.stop_stream()
        stream.close()
        p.terminate()


async def receive_transcripts(websocket):
    """Receive and print transcription results."""
    try:
        async for message in websocket:
            headers, payload = decode_event(message)

            if headers.get(":message-type") == "event":
                results = payload.get("Transcript", {}).get("Results", [])
                for result in results:
                    is_partial = result.get("IsPartial", True)
                    text = result["Alternatives"][0]["Transcript"]
                    prefix = "..." if is_partial else ">>>"
                    print(f"{prefix} {text}")

            elif headers.get(":message-type") == "exception":
                print(f"ERROR: {payload.get('Message', 'Unknown error')}")
                break
    except websockets.exceptions.ConnectionClosed:
        pass


async def main():
    extra_headers = {
        "Origin": "https://localhost",
    }
    print(f"Connecting to Transcribe...")
    async with websockets.connect(
        STREAM_URL,
        extra_headers=extra_headers,
        ping_timeout=None,
    ) as ws:
        print("Connected!")
        await asyncio.gather(
            mic_stream(ws),
            receive_transcripts(ws),
        )


if __name__ == "__main__":
    asyncio.run(main())