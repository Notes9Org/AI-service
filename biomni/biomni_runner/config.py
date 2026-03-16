"""BioMni-specific configuration. Reuses Bedrock config from services.aws_config."""
import os
from pathlib import Path

# Ensure biomni root is on path when run as module
_backend_root = Path(__file__).resolve().parent.parent  # biomni/


def get_data_path() -> str:
    """Return BioMni data path. BioMni expects path to the parent of biomni_data/ (it joins path/biomni_data/data_lake)."""
    path = os.getenv("BIOMNI_DATA_PATH")
    if path:
        return str(Path(path).resolve())
    # Default: biomni/data/biomni (parent of biomni_data/; BioMni will use path/biomni_data/data_lake)
    default = (_backend_root / "data" / "biomni").resolve()
    return str(default)


def get_llm_model() -> str:
    """Return Bedrock model ID for BioMni. Reuses existing backend config when available."""
    model = os.getenv("BIOMNI_LLM_MODEL")
    if model:
        return model
    try:
        from services.aws_config import get_bedrock_config
        cfg = get_bedrock_config()
        return cfg.get_chat_model_id()
    except Exception:
        return os.getenv("BEDROCK_CHAT_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")


def get_timeout_seconds() -> int:
    """Return BioMni timeout in seconds."""
    return int(os.getenv("BIOMNI_TIMEOUT_SECONDS", "600"))


def ensure_bedrock_env() -> None:
    """Set BioMni env vars for Bedrock. Uses existing AWS credentials from boto3 chain."""
    os.environ.setdefault("LLM_SOURCE", "Bedrock")
    if "AWS_REGION" not in os.environ:
        try:
            from services.aws_config import get_bedrock_config
            cfg = get_bedrock_config()
            os.environ["AWS_REGION"] = cfg.region
        except Exception:
            os.environ.setdefault("AWS_REGION", "us-east-1")
