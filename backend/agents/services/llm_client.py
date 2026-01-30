"""LLM client abstraction with retry and JSON validation (Azure OpenAI or AWS Bedrock)."""
import json
import time
import re
from typing import Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

from services.config import get_llm_provider, get_azure_openai_config, get_bedrock_config

logger = structlog.get_logger()


class LLMError(Exception):
    """Custom exception for LLM errors."""
    pass


class LLMClient:
    """Abstracted LLM client with structured output support (Azure or Bedrock)."""

    def __init__(self):
        self._provider = get_llm_provider()
        if self._provider == "bedrock":
            self.config = get_bedrock_config()
            self.client = self.config.create_bedrock_runtime_client()
            self.default_deployment = self.config.get_chat_model_id()
            self.default_model = self.default_deployment
            self.chat_model_id_sql = self.config.get_chat_model_id_sql()
            self.chat_model_id_summary = self.config.get_chat_model_id_summary()
            self.max_completion_tokens = self.config.max_completion_tokens
            self.default_temperature = self.config.default_temperature
            logger.info(
                "LLM client initialized",
                model=self.default_model,
                provider="AWS Bedrock",
                max_completion_tokens=self.max_completion_tokens,
                default_temperature=self.default_temperature,
            )
        else:
            self.config = get_azure_openai_config()
            self.client = self.config.create_client()
            self.default_deployment = self.config.chat_deployment
            self.default_model = self.default_deployment
            self.chat_model_id_sql = self.config.get_chat_model_id_sql()
            self.chat_model_id_summary = self.config.get_chat_model_id_summary()
            self.max_completion_tokens = self.config.max_completion_tokens
            self.default_temperature = self.config.default_temperature
            logger.info(
                "LLM client initialized",
                model=self.default_model,
                deployment=self.default_deployment,
                endpoint=self.config.endpoint,
                api_version=self.config.api_version,
                max_completion_tokens=self.max_completion_tokens,
                default_temperature=self.default_temperature,
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def complete_json(
        self,
        prompt: str,
        schema: Dict[str, Any],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output from prompt.
        """
        model = model or self.default_deployment
        use_temperature = temperature if temperature is not None else self.default_temperature
        start_time = time.time()

        try:
            if self._provider == "bedrock":
                content = self._converse_bedrock(
                    model=model,
                    system="You are a helpful assistant that returns valid JSON. Always respond with JSON only, no markdown, no explanations.",
                    user=f"{prompt}\n\nIMPORTANT: Return ONLY valid JSON, no markdown code blocks, no explanations.",
                    temperature=use_temperature,
                )
            else:
                content = self._complete_json_azure(model, prompt, use_temperature, start_time)

            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
                content = re.sub(r"\s*```$", "", content)
            content = content.strip()

            result = json.loads(content)
            latency_ms = (time.time() - start_time) * 1000
            logger.info(
                "LLM JSON completion",
                model=model,
                latency_ms=round(latency_ms, 2),
                response_length=len(content),
            )
            return result
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON from LLM", error=str(e), content=content[:200])
            raise LLMError(f"Invalid JSON response: {str(e)}")
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_str = str(e)
            logger.error(
                "LLM JSON completion failed",
                error=error_str,
                model=model,
                provider=self._provider,
                latency_ms=round(latency_ms, 2),
            )
            if self._provider == "azure" and "'input' is a required property" in error_str:
                raise LLMError(
                    f"API error: The deployment '{model}' might not exist or is not a chat model. "
                    f"Check your Azure Portal → Deployments."
                )
            if self._provider == "azure" and "deployment" in error_str.lower() and "not found" in error_str.lower():
                raise LLMError(
                    f"Deployment '{model}' not found. Set AZURE_OPENAI_CHAT_DEPLOYMENT to the correct deployment name."
                )
            if self._provider == "bedrock":
                raise LLMError(f"Bedrock LLM failed (model={model}): {error_str}") from e
            raise LLMError(f"LLM completion failed: {error_str}") from e

    def _complete_json_azure(
        self, model: str, prompt: str, use_temperature: float, start_time: float
    ) -> str:
        system_message = "You are a helpful assistant that returns valid JSON. Always respond with JSON only, no markdown, no explanations."
        user_message = f"{prompt}\n\nIMPORTANT: Return ONLY valid JSON, no markdown code blocks, no explanations."
        request_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "timeout": 30.0,
        }
        if use_temperature != 1.0:
            request_params["temperature"] = use_temperature
        try:
            request_params["max_completion_tokens"] = self.max_completion_tokens
        except Exception:
            pass
        try:
            response = self.client.chat.completions.create(**request_params)
        except Exception as temp_error:
            if "temperature" in str(temp_error).lower() and "unsupported" in str(temp_error).lower():
                if "temperature" in request_params:
                    del request_params["temperature"]
                response = self.client.chat.completions.create(**request_params)
            else:
                raise
        content = response.choices[0].message.content
        if not content:
            raise LLMError("Empty response from LLM")
        return content

    def _converse_bedrock(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.0,
    ) -> str:
        """Call Bedrock Converse API and return assistant text."""
        messages = [{"role": "user", "content": [{"text": user}]}]
        system_content = [{"text": system}]
        inference_config = {"maxTokens": self.max_completion_tokens}
        if temperature != 1.0:
            inference_config["temperature"] = temperature
        response = self.client.converse(
            modelId=model,
            messages=messages,
            system=system_content,
            inferenceConfig=inference_config,
        )
        out = response.get("output", {}).get("message", {}).get("content", [])
        if not out or "text" not in out[0]:
            raise LLMError("Empty or invalid response from Bedrock")
        return out[0]["text"]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def complete_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate text output from prompt.
        """
        model = model or self.default_deployment
        start_time = time.time()

        try:
            if self._provider == "bedrock":
                content = self._converse_bedrock(
                    model=model,
                    system="You are a helpful assistant.",
                    user=prompt,
                    temperature=temperature,
                )
            else:
                request_params = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    "timeout": 30.0,
                }
                if temperature != 1.0:
                    request_params["temperature"] = temperature
                try:
                    request_params["max_completion_tokens"] = self.max_completion_tokens
                except Exception:
                    pass
                try:
                    response = self.client.chat.completions.create(**request_params)
                except Exception as temp_error:
                    if "temperature" in str(temp_error).lower() and "unsupported" in str(temp_error).lower():
                        if "temperature" in request_params:
                            del request_params["temperature"]
                        response = self.client.chat.completions.create(**request_params)
                    else:
                        raise
                content = response.choices[0].message.content
                if not content:
                    raise LLMError("Empty response from LLM")

            latency_ms = (time.time() - start_time) * 1000
            logger.info(
                "LLM text completion",
                model=model,
                latency_ms=round(latency_ms, 2),
                response_length=len(content),
            )
            return content.strip()
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(
                "LLM text completion failed",
                error=str(e),
                model=model,
                latency_ms=round(latency_ms, 2),
            )
            raise LLMError(f"LLM completion failed: {str(e)}")
