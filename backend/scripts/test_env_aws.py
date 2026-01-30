#!/usr/bin/env python3
"""Test AWS Bedrock .env: embedding + chat. Run: python -m scripts.test_env_aws"""
import os
import sys
import json
from pathlib import Path

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_env_path = Path(_project_root) / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)
else:
    print("No .env found. Set LLM_PROVIDER=bedrock, AWS_REGION, BEDROCK_EMBEDDING_MODEL_ID, BEDROCK_CHAT_MODEL_ID.")
    sys.exit(1)


def _invoke_embed(client, model_id: str, dimensions: int, is_v4: bool = True) -> list:
    body = {"texts": ["Hello, world"], "input_type": "search_document", "embedding_types": ["float"]}
    if is_v4:
        body["output_dimension"] = dimensions
    response = client.invoke_model(
        modelId=model_id, body=json.dumps(body),
        accept="application/json", contentType="application/json",
    )
    result = json.loads(response["body"].read())
    raw = result.get("embeddings")
    if isinstance(raw, dict) and "float" in raw:
        return raw["float"][0]
    if isinstance(raw, list) and raw:
        return raw[0]
    raise RuntimeError("Unexpected embedding response")


def test_bedrock_embedding(client, model_id: str, dimensions: int) -> None:
    ids = [model_id]
    if ":" not in model_id:
        ids.append(f"{model_id}:0")
    for mid in ids:
        try:
            emb = _invoke_embed(client, mid, dimensions, is_v4=True)
            if len(emb) != dimensions:
                raise RuntimeError(f"Expected dim {dimensions}, got {len(emb)}")
            print(f"  Embedding OK (model={mid}, dim={len(emb)})")
            return
        except Exception as e:
            if "ValidationException" in type(e).__name__ and "model identifier" in str(e).lower():
                continue
            raise
    print(f"  Error: Enable Cohere Embed v4 in Bedrock or set BEDROCK_EMBEDDING_MODEL_ID=cohere.embed-v4:0")
    raise SystemExit(1)


CLAUDE_V1 = "anthropic.claude-3-5-sonnet-20240620-v1:0"
CLAUDE_V2 = "anthropic.claude-3-5-sonnet-20241022-v2:0"


def test_bedrock_chat(client, model_id: str) -> None:
    msg = [{"role": "user", "content": [{"text": "Say hello in one word."}]}]
    fallback = CLAUDE_V1 if model_id == CLAUDE_V2 else (CLAUDE_V2 if model_id == CLAUDE_V1 else None)
    ids = [model_id] + ([fallback] if fallback else [])
    last_err = None
    for mid in ids:
        try:
            response = client.converse(modelId=mid, messages=msg)
            out = response.get("output", {}).get("message", {}).get("content", [])
            if not out or "text" not in out[0]:
                raise RuntimeError("Unexpected response")
            print(f"  Chat OK (model={mid}): {out[0]['text'][:60]!r}")
            return
        except Exception as e:
            last_err = e
            if "on-demand throughput" in str(e).lower() or "inference profile" in str(e).lower():
                continue
            raise
    print(f"  Error: {last_err}")
    print("  Set BEDROCK_CHAT_MODEL_ID to an inference profile ID from Bedrock → Model catalog, or use v1 model.")
    raise SystemExit(1)


def list_inference_profiles(region: str) -> None:
    import boto3
    client = boto3.client("bedrock", region_name=region)
    if not hasattr(client, "list_inference_profiles"):
        print("  Upgrade boto3: pip install -U boto3. Or Bedrock → Model catalog → copy inference profile ID.")
        return
    try:
        response = client.list_inference_profiles()
        seen = set()
        while True:
            for p in response.get("inferenceProfileSummaries", []):
                pid = p.get("inferenceProfileId") or p.get("inferenceProfileArn") or ""
                if not pid or pid in seen:
                    continue
                seen.add(pid)
                name = p.get("inferenceProfileName", "")
                is_claude = "claude" in (name + " ".join(str(m) for m in (p.get("models") or [])[:2])).lower()
                if is_claude:
                    print(f"  BEDROCK_CHAT_MODEL_ID={pid}  # {name}")
                else:
                    print(f"  # {name} -> {pid}")
            if not response.get("nextToken"):
                break
            response = client.list_inference_profiles(nextToken=response["nextToken"])
        if not seen:
            print("  No profiles. Bedrock → Model catalog → open Claude → copy inference profile ID.")
    except Exception as e:
        print(f"  {e}. Bedrock → Model catalog → copy inference profile ID for Claude.")


def main() -> int:
    from services.config import get_llm_provider, get_bedrock_config, ConfigurationError

    if "--list-profiles" in sys.argv:
        config = get_bedrock_config()
        print("Inference profiles (set BEDROCK_CHAT_MODEL_ID in .env):")
        list_inference_profiles(config.region)
        return 0

    if get_llm_provider() != "bedrock":
        print("Set LLM_PROVIDER=bedrock in .env to test AWS.")
        return 0

    try:
        config = get_bedrock_config()
        client = config.create_bedrock_runtime_client()
    except ConfigurationError as e:
        print(f"Config: {e}")
        return 1

    try:
        print("1. Embedding...")
        test_bedrock_embedding(client, config.get_embedding_model(), config.get_dimensions())
        print("2. Chat...")
        test_bedrock_chat(client, config.get_chat_model_id())
    except SystemExit:
        raise
    except Exception as e:
        print(f"  Error: {e}")
        return 1

    print("All AWS Bedrock checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
