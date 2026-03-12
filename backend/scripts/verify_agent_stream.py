#!/usr/bin/env python3
"""
Verify that /agent/stream emits SSE events incrementally (not buffered until the end).

Usage:
  export AGENT_STREAM_TOKEN="your-supabase-jwt"
  export AGENT_BASE_URL="http://localhost:8000"  # optional, default
  python -m scripts.verify_agent_stream

If events print with increasing timestamps as they arrive, streaming works.
If all events print at once at the end, something is buffering the response.
"""
import json
import os
import sys
import time

try:
    import httpx
except ImportError:
    print("Install httpx: pip install httpx", file=sys.stderr)
    sys.exit(1)

BASE_URL = os.environ.get("AGENT_BASE_URL", "http://localhost:8000")
TOKEN = os.environ.get("AGENT_STREAM_TOKEN", "")


def main():
    if not TOKEN:
        print(
            "Set AGENT_STREAM_TOKEN to a valid Supabase JWT.\n"
            "Example: export AGENT_STREAM_TOKEN='eyJ...'",
            file=sys.stderr,
        )
        sys.exit(1)

    url = f"{BASE_URL.rstrip('/')}/agent/stream"
    payload = {
        "query": "How many experiments were completed last month?",
        "session_id": "verify-stream-test",
        "history": [],
    }

    print(f"POST {url}")
    print("Expecting events to arrive incrementally (timestamps should increase as they print).\n")

    start = time.time()
    event_count = 0

    with httpx.Client(timeout=120.0) as client:
        with client.stream(
            "POST",
            url,
            json=payload,
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "application/json",
            },
        ) as response:
            if response.status_code != 200:
                print(f"Error: {response.status_code} {response.reason_phrase}", file=sys.stderr)
                print(response.text[:500], file=sys.stderr)
                sys.exit(1)

            buffer = ""
            for chunk in response.iter_text():
                if not chunk:
                    continue
                buffer += chunk
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    event_type = None
                    data_str = None
                    for line in block.split("\n"):
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:"):
                            data_str = line[5:].strip()
                    if event_type and data_str:
                        event_count += 1
                        elapsed = time.time() - start
                        try:
                            data = json.loads(data_str)
                            preview = str(data)[:80] + "..." if len(str(data)) > 80 else str(data)
                        except json.JSONDecodeError:
                            preview = data_str[:80]
                        print(f"[{elapsed:.2f}s] event: {event_type} | {preview}")
                        if event_type == "done":
                            print(f"\nDone. Total events: {event_count}")
                            return
                        if event_type == "error":
                            print(f"Error: {data.get('error', data)}", file=sys.stderr)
                            sys.exit(1)

    print(f"\nStream ended. Total events: {event_count}")


if __name__ == "__main__":
    main()
