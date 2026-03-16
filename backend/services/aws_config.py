"""AWS Bedrock configuration. All AWS/LLM provider=bedrock settings live here.

To change the chat model: set these in backend/.env (copy from .env.example):
  BEDROCK_CHAT_MODEL_ID          — agent pipeline (normalize, router, judge, retry, RAG)
  BEDROCK_CHAT_MODEL_ID_SQL     — optional: SQL generation only (defaults to main)
  BEDROCK_CHAT_MODEL_ID_SUMMARY — summarizer + general chat /chat endpoint (defaults to main)
"""
import os
from typing import Optional

from services.config_errors import ConfigurationError


_bedrock_config: Optional["BedrockConfig"] = None


def get_bedrock_config() -> "BedrockConfig":
    """Get or create AWS Bedrock configuration."""
    global _bedrock_config
    if _bedrock_config is None:
        _bedrock_config = BedrockConfig()
    return _bedrock_config


class BedrockConfig:
    """AWS Bedrock configuration for LLM and embeddings."""

    def __init__(self):
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.embedding_model_id = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "cohere.embed-v4:0")
        self.dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
        self.chat_model_id = os.getenv("BEDROCK_CHAT_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
        self.chat_model_id_sql = os.getenv("BEDROCK_CHAT_MODEL_ID_SQL") or self.chat_model_id
        self.chat_model_id_summary = os.getenv("BEDROCK_CHAT_MODEL_ID_SUMMARY") or self.chat_model_id
        self.max_completion_tokens = int(os.getenv("BEDROCK_MAX_COMPLETION_TOKENS", "4096"))
        self.default_temperature = float(os.getenv("NORMALIZE_TEMPERATURE", os.getenv("AZURE_OPENAI_DEFAULT_TEMPERATURE", "0.0")))
        self._validate()

    def _validate(self):
        if not self.region:
            raise ConfigurationError("AWS Bedrock: AWS_REGION must be set (e.g. us-east-1).")
        if not self.embedding_model_id:
            raise ConfigurationError("AWS Bedrock: BEDROCK_EMBEDDING_MODEL_ID must be set (e.g. cohere.embed-v4:0).")
        if not self.chat_model_id:
            raise ConfigurationError(
                "AWS Bedrock: BEDROCK_CHAT_MODEL_ID must be set (e.g. anthropic.claude-3-5-sonnet-20240620-v1:0 or inference profile ID)."
            )

    def get_embedding_model(self) -> str:
        return self.embedding_model_id

    def get_dimensions(self) -> int:
        return self.dimensions

    def get_chat_model_id(self) -> str:
        return self.chat_model_id

    def get_chat_model_id_sql(self) -> str:
        return self.chat_model_id_sql

    def get_chat_model_id_summary(self) -> str:
        return self.chat_model_id_summary

    def create_bedrock_runtime_client(self):
        try:
            import boto3
            return boto3.client("bedrock-runtime", region_name=self.region)
        except Exception as e:
            raise ConfigurationError(
                f"AWS Bedrock: Failed to create bedrock-runtime client. Error: {str(e)}"
            ) from e
