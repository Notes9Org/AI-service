"""
Re-index semantic_chunks embeddings with the current embedder (Azure or Bedrock per LLM_PROVIDER).

Use this after switching embedding models (e.g. Azure text-embedding-3-small → Bedrock Cohere).
Query and index embeddings must come from the same model; otherwise RAG finds 0 chunks.

Run from project root (python-service):
  python -m scripts.reindex_embeddings
  python -m scripts.reindex_embeddings --dry-run
  python -m scripts.reindex_embeddings --limit 100

Or from scripts/ directory:
  python reindex_embeddings.py
"""
import argparse
import os
import sys

# Ensure project root (python-service) is on path when run as script from scripts/
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dotenv import load_dotenv
load_dotenv(os.path.join(_project_root, ".env"))


def main():
    parser = argparse.ArgumentParser(description="Re-embed all semantic chunks with current embedder")
    parser.add_argument("--dry-run", action="store_true", help="Only report how many chunks would be updated")
    parser.add_argument("--limit", type=int, default=None, help="Max chunks to process (default: all)")
    parser.add_argument("--batch", type=int, default=20, help="Chunk batch size for embed API (default: 20)")
    args = parser.parse_args()

    from services.db import SupabaseService
    from services.embedder import EmbeddingService
    from services.config import get_llm_provider

    provider = get_llm_provider()
    embedder = EmbeddingService()
    db = SupabaseService()
    model = getattr(embedder, "model", "unknown")
    print(f"Using embedder: provider={provider}, model={model}")
    if args.dry_run:
        print("Dry run: no updates will be written.")

    offset = 0
    batch_size = args.batch
    total_updated = 0
    total_failed = 0

    while True:
        page = db.get_semantic_chunks_page(offset=offset, limit=batch_size)
        if not page:
            break
        if args.limit is not None and total_updated + total_failed >= args.limit:
            break

        ids = [c["id"] for c in page]
        contents = [c.get("content") or "" for c in page]
        if not contents:
            offset += batch_size
            continue

        if args.dry_run:
            total_updated += len(page)
            print(f"  Would re-embed {len(page)} chunks (offset {offset})")
            offset += batch_size
            if args.limit and total_updated >= args.limit:
                break
            continue

        embeddings = embedder.embed_batch(contents)
        for i, chunk in enumerate(page):
            if args.limit and total_updated + total_failed >= args.limit:
                break
            if i >= len(embeddings) or embeddings[i] is None:
                total_failed += 1
                continue
            if db.update_chunk_embedding(chunk["id"], embeddings[i]):
                total_updated += 1
            else:
                total_failed += 1

        print(f"  Processed offset {offset}: updated {total_updated}, failed {total_failed}")
        offset += batch_size
        if len(page) < batch_size:
            break

    print(f"Done. Updated={total_updated}, failed={total_failed}")
    if args.dry_run and total_updated > 0:
        print("Run without --dry-run to apply re-embedding.")
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
