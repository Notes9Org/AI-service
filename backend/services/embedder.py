"""Embedding generation service using AWS Bedrock.

Query and index embeddings must use the same model; after changing Bedrock models,
re-index existing embeddings to keep them consistent.
"""
import json
from typing import List, Optional
from dotenv import load_dotenv
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from services.config import get_bedrock_config

load_dotenv()

logger = structlog.get_logger()


def _parse_bedrock_embeddings_response(result: dict, expected_count: int) -> List[List[float]]:
    """Parse Cohere embed-v4 response: embeddings.float or embeddings list."""
    if "embeddings" not in result:
        raise ValueError(f"Unexpected embedding response: {result}")
    raw = result["embeddings"]
    if isinstance(raw, dict) and "float" in raw:
        return raw["float"]
    if isinstance(raw, list) and len(raw) >= expected_count:
        return raw
    raise ValueError(f"Unexpected embeddings shape: {result}")


class EmbeddingService:
    """Service for generating text embeddings via AWS Bedrock."""

    def __init__(self):
        self.config = get_bedrock_config()
        self.client = self.config.create_bedrock_runtime_client()
        self.model = self.config.get_embedding_model()
        self.dimensions = self.config.get_dimensions()
        logger.info(
            "Embedding service initialized",
            model=self.model,
            dimensions=self.dimensions,
            provider="AWS Bedrock",
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        return self._embed_text_bedrock(text.strip())

    def _invoke_bedrock_embed(self, body: str, model_id: str) -> dict:
        """Invoke Bedrock embedding; retry with model_id:0 if model identifier invalid."""
        try:
            response = self.client.invoke_model(
                modelId=model_id, body=body,
                accept="application/json", contentType="application/json",
            )
            return json.loads(response["body"].read())
        except Exception as e:
            if "ValidationException" in type(e).__name__ and "model identifier" in str(e).lower() and ":" not in model_id:
                response = self.client.invoke_model(
                    modelId=f"{model_id}:0", body=body,
                    accept="application/json", contentType="application/json",
                )
                return json.loads(response["body"].read())
            raise

    def _embed_text_bedrock(self, text: str) -> List[float]:
        body = json.dumps({
            "texts": [text],
            "input_type": "search_document",
            "embedding_types": ["float"],
            "output_dimension": self.dimensions,
        })
        result = self._invoke_bedrock_embed(body, self.model)
        embeddings = _parse_bedrock_embeddings_response(result, 1)
        embedding = embeddings[0]
        if len(embedding) != self.dimensions:
            raise ValueError(
                f"Invalid embedding dimensions: expected {self.dimensions}, got {len(embedding)}"
            )
        logger.debug(
            "Embedding generated successfully",
            model=self.model,
            dimensions=len(embedding),
        )
        return embedding


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts (batch processing).
        Returns list of embeddings, with None for failed texts.
        """
        if not texts:
            return []

        valid_texts = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not valid_texts:
            return [None] * len(texts)

        return self._embed_batch_bedrock(texts, valid_texts)

    def _embed_batch_bedrock(
        self, texts: List[str], valid_texts: List[tuple]
    ) -> List[Optional[List[float]]]:
        text_values = [t for _, t in valid_texts]
        body = json.dumps({
            "texts": text_values,
            "input_type": "search_document",
            "embedding_types": ["float"],
            "output_dimension": self.dimensions,
        })
        try:
            result = self._invoke_bedrock_embed(body, self.model)
            embedding_list = _parse_bedrock_embeddings_response(result, len(text_values))
        except Exception as e:
            logger.error("Error generating batch embeddings", error=str(e), count=len(texts))
            return [None] * len(texts)

        result = []
        valid_idx = 0
        for i, text in enumerate(texts):
            if text and text.strip():
                if valid_idx < len(embedding_list):
                    emb = embedding_list[valid_idx]
                    if emb and len(emb) == self.dimensions:
                        result.append(emb)
                    else:
                        result.append(None)
                else:
                    result.append(None)
                valid_idx += 1
            else:
                result.append(None)
        logger.info(
            "Batch embeddings generated",
            total=len(texts),
            successful=len([e for e in result if e]),
        )
        return result

    # Azure embedding paths have been removed; only Bedrock is supported.
