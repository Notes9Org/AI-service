"""Azure OpenAI configuration. All Azure/LLM provider=azure settings live here."""
import os
from typing import Optional

from services.config_errors import ConfigurationError

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = None


_azure_openai_config: Optional["AzureOpenAIConfig"] = None


def get_azure_openai_config() -> "AzureOpenAIConfig":
    """Get or create Azure OpenAI configuration."""
    global _azure_openai_config
    if _azure_openai_config is None:
        _azure_openai_config = AzureOpenAIConfig()
    return _azure_openai_config


class AzureOpenAIConfig:
    """Azure OpenAI configuration."""

    def __init__(self):
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        self.model_name = os.getenv("AZURE_OPENAI_MODEL_NAME", "text-embedding-3-small")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "text-embedding-3-small")
        self.dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
        self.chat_deployment = (
            os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
            or os.getenv("AZURE_OPENAI_CHAT_MODEL")
            or "gpt-5.2-chat"
        )
        self.chat_model_id_sql = self.chat_deployment
        self.chat_model_id_summary = self.chat_deployment
        self.max_completion_tokens = int(os.getenv("AZURE_OPENAI_MAX_COMPLETION_TOKENS", "16384"))
        self.default_temperature = float(os.getenv("AZURE_OPENAI_DEFAULT_TEMPERATURE", "0.0"))
        self._validate()

    def _validate(self):
        if not self.endpoint:
            raise ConfigurationError(
                "Azure OpenAI: AZURE_OPENAI_ENDPOINT must be set (e.g. https://<resource>.openai.azure.com)."
            )
        if not self.api_key:
            raise ConfigurationError("Azure OpenAI: AZURE_OPENAI_API_KEY must be set.")
        if not self.chat_deployment:
            raise ConfigurationError(
                "Azure OpenAI: AZURE_OPENAI_CHAT_DEPLOYMENT or AZURE_OPENAI_CHAT_MODEL must be set."
            )

    def get_embedding_model(self) -> str:
        return self.deployment or self.model_name

    def get_dimensions(self) -> int:
        return self.dimensions

    def get_chat_model_id(self) -> str:
        return self.chat_deployment

    def get_chat_model_id_sql(self) -> str:
        return self.chat_model_id_sql

    def get_chat_model_id_summary(self) -> str:
        return self.chat_model_id_summary

    def create_client(self):
        try:
            from openai import AzureOpenAI
            endpoint = self.endpoint.rstrip("/")
            if not endpoint.startswith("https://"):
                raise ConfigurationError(
                    f"Azure OpenAI: Invalid endpoint format: {endpoint}. Must start with https://"
                )
            return AzureOpenAI(
                api_version=self.api_version,
                azure_endpoint=endpoint,
                api_key=self.api_key,
            )
        except Exception as e:
            raise ConfigurationError(
                f"Azure OpenAI: Failed to create client. Error: {str(e)}"
            ) from e
