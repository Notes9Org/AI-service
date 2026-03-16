"""CLI entry point for BioMni agent. Run with: python -m biomni_runner.run (from biomni/)"""
import matplotlib
matplotlib.use("Agg")

import argparse
import sys
from pathlib import Path

# Ensure backend root is on path
_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

# Load .env from biomni root
try:
    from dotenv import load_dotenv
    load_dotenv(_backend_root / ".env")
except ImportError:
    pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run BioMni biomedical AI agent with AWS Bedrock"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to execute (e.g. 'Predict ADMET properties for aspirin')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Bedrock model ID (default: from BEDROCK_CHAT_MODEL_ID or BIOMNI_LLM_MODEL)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to parent of biomni_data/ (BioMni uses path/biomni_data/). Default: biomni/data/biomni",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (prompt for queries until exit)",
    )
    parser.add_argument(
        "--no-load-datalake",
        action="store_true",
        help="Skip loading datalake (faster init, but some tools won't work). Default: load from local path.",
    )
    args = parser.parse_args()

    if not args.query and not args.interactive:
        parser.error("Provide --query or --interactive")

    from biomni_runner.agent import BioMniAgent

    agent = BioMniAgent(
        path=args.data_path,
        llm=args.model,
        load_datalake=not args.no_load_datalake,
    )

    if args.interactive:
        print("BioMni agent ready (Bedrock). Type your query and press Enter. 'quit' or 'exit' to stop.")
        while True:
            try:
                query = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                break
            try:
                _log, answer = agent.go(query)
                print(answer)
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
        return 0

    try:
        _log, answer = agent.go(args.query)
        print(answer)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
