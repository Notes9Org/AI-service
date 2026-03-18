"""
Backfill Zep memory from existing chat_sessions and chat_messages.

Migrates existing conversations to Zep so context is available when Zep is enabled.
Idempotent: re-running is safe (Zep appends messages).

Run from project root (backend):
  python -m scripts.backfill_zep
  python -m scripts.backfill_zep --dry-run
  python -m scripts.backfill_zep --limit 10

Requires: ZEP_API_KEY, DATABASE_URL or Supabase config.
"""
import argparse
import asyncio
import os
import sys

# Ensure project root (backend) is on path when run as script from scripts/
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dotenv import load_dotenv

load_dotenv(os.path.join(_project_root, ".env"))


async def main():
    parser = argparse.ArgumentParser(description="Backfill Zep memory from chat_messages")
    parser.add_argument("--dry-run", action="store_true", help="Only report what would be migrated")
    parser.add_argument("--limit", type=int, default=None, help="Max sessions to process (default: all)")
    args = parser.parse_args()

    if not (os.getenv("ZEP_API_KEY") or "").strip():
        print("ZEP_API_KEY is not set. Cannot backfill.", file=sys.stderr)
        sys.exit(1)

    from services.config import get_database_config
    from services.zep_memory import add_messages as zep_add_messages

    try:
        db_config = get_database_config()
        conn = db_config.get_connection()
    except Exception as e:
        print(f"Failed to connect to database: {e}", file=sys.stderr)
        sys.exit(1)

    # Fetch chat_sessions with user_id
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, user_id FROM chat_sessions ORDER BY created_at"
            )
            sessions = [{"id": row[0], "user_id": row[1]} for row in cur.fetchall()]
    except Exception as e:
        print(f"Failed to fetch chat_sessions: {e}", file=sys.stderr)
        conn.close()
        sys.exit(1)

    if args.limit:
        sessions = sessions[: args.limit]

    print(f"Found {len(sessions)} chat sessions")
    if args.dry_run:
        with conn.cursor() as cur:
            for s in sessions:
                cur.execute(
                    "SELECT id, role, content FROM chat_messages WHERE session_id = %s ORDER BY created_at",
                    (s["id"],),
                )
                msgs = [{"id": r[0], "role": r[1], "content": r[2]} for r in cur.fetchall()]
                pairs = _pair_messages(msgs)
                print(f"  Session {s['id']}: {len(msgs)} messages, {len(pairs)} turns")
        conn.close()
        print("Dry run complete. No data written.")
        return

    total_added = 0
    total_errors = 0

    with conn.cursor() as cur:
        for s in sessions:
            session_id = str(s["id"])
            user_id = str(s.get("user_id") or "")
            if not user_id:
                print(f"  Skip session {session_id}: no user_id")
                continue

            try:
                cur.execute(
                    "SELECT id, role, content FROM chat_messages WHERE session_id = %s ORDER BY created_at",
                    (s["id"],),
                )
                msgs = [{"id": r[0], "role": r[1], "content": r[2]} for r in cur.fetchall()]
            except Exception as e:
                print(f"  Error fetching messages for {session_id}: {e}")
                total_errors += 1
                continue

        pairs = _pair_messages(msgs)
        for user_content, assistant_content in pairs:
            try:
                await zep_add_messages(
                    session_id=session_id,
                    user_id=user_id,
                    user_content=user_content,
                    assistant_content=assistant_content,
                )
                total_added += 1
            except Exception as e:
                print(f"  Error adding to Zep for {session_id}: {e}")
                total_errors += 1

    conn.close()
    print(f"Backfill complete. Added {total_added} turns, {total_errors} errors.")


def _pair_messages(msgs: list) -> list:
    """Convert ordered messages into (user_content, assistant_content) pairs.
    Skips system messages. Only includes complete user+assistant pairs.
    """
    pairs = []
    i = 0
    while i < len(msgs):
        m = msgs[i]
        role = (m.get("role") or "").lower()
        content = (m.get("content") or "").strip()

        if role == "system":
            i += 1
            continue

        if role == "user":
            # Look for next assistant
            if i + 1 < len(msgs):
                next_m = msgs[i + 1]
                next_role = (next_m.get("role") or "").lower()
                next_content = (next_m.get("content") or "").strip()
                if next_role == "assistant":
                    pairs.append((content, next_content))
                    i += 2
                    continue
            # Orphan user: skip (incomplete turn)
            i += 1
        elif role == "assistant":
            # Orphan assistant: skip (incomplete turn)
            i += 1
        else:
            i += 1

    return pairs


if __name__ == "__main__":
    asyncio.run(main())
