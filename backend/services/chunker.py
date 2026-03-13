"""Text chunking utilities for backend-only use.

Note: The worker owns the full semantic chunking and metadata logic. This
backend module is kept minimal and should not be used for indexing into
semantic_chunks; it is safe to delete entirely if no backend code imports it.
"""
import re
import json
from typing import List
from bs4 import BeautifulSoup
import html2text
import structlog

logger = structlog.get_logger()


def extract_plain_text(content: str) -> str:
    """
    Extract plain text from various formats.
    Handles HTML, TipTap JSON, and plain text.
    """
    if not content:
        return ""

    # Try to parse as TipTap JSON
    try:
        tiptap_data = json.loads(content)
        if isinstance(tiptap_data, dict) and "type" in tiptap_data:
            return extract_from_tiptap(tiptap_data)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Try to parse as HTML
    if "<" in content and ">" in content:
        try:
            soup = BeautifulSoup(content, "html.parser")
            # Remove script and style elements
            for script in soup(["script", "style", "meta", "link"]):
                script.decompose()

            # Convert to plain text
            text = soup.get_text(separator=" ", strip=True)
            # Clean up whitespace
            text = re.sub(r"\s+", " ", text)
            return text.strip()
        except Exception as e:
            logger.warning("Error parsing HTML", error=str(e))
            # Fallback to html2text
            h = html2text.HTML2Text()
            h.ignore_links = True
            h.ignore_images = True
            return h.handle(content).strip()

    # Plain text - just clean up
    return content.strip()


def extract_from_tiptap(doc: dict) -> str:
    """Extract text from TipTap JSON structure."""
    text_parts: List[str] = []

    def traverse(node: dict) -> None:
        if isinstance(node, dict):
            if "text" in node:
                text_parts.append(node["text"])
            if "content" in node and isinstance(node["content"], list):
                for child in node["content"]:
                    traverse(child)

    traverse(doc)
    return " ".join(text_parts)


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    Simple sentence-based chunking with overlap.

    This is provided only for any backend features that need ad-hoc chunking;
    it is not used for writing to semantic_chunks (the worker handles that).
    """
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []

    # Split into sentences (try to preserve sentence boundaries)
    sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])"
    sentences = re.split(sentence_pattern, text)

    # If no sentence boundaries found, split by paragraphs/newlines
    if len(sentences) == 1:
        sentences = text.split("\n\n")
        if len(sentences) == 1:
            sentences = text.split("\n")

    current_chunk: List[str] = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_length = len(sentence)

        # If single sentence is larger than chunk_size, split it by words
        if sentence_length > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            words = sentence.split()
            word_chunk: List[str] = []
            word_length = 0

            for word in words:
                word_len = len(word) + 1  # +1 for space
                if word_length + word_len > chunk_size:
                    if word_chunk:
                        chunks.append(" ".join(word_chunk))
                    word_chunk = [word]
                    word_length = word_len
                else:
                    word_chunk.append(word)
                    word_length += word_len

            if word_chunk:
                current_chunk = word_chunk
                current_length = word_length
            continue

        # Normal case: add sentence to current chunk
        if current_length + sentence_length + 1 <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length + 1
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            if chunks and chunk_overlap > 0:
                overlap_text = " ".join(current_chunk[-3:])
                if len(overlap_text) > chunk_overlap:
                    overlap_text = overlap_text[-chunk_overlap:]
                current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                current_length = len(overlap_text) + sentence_length + 1 if overlap_text else sentence_length
            else:
                current_chunk = [sentence]
                current_length = sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [c.strip() for c in chunks if c.strip()]