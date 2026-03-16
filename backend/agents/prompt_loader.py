"""Load prompts from markdown files. Keeps prompts editable by domain experts without touching code."""

import re
from pathlib import Path

_PROMPTS_ROOT = Path(__file__).resolve().parent.parent / "prompts"

_NOTES_SEPARATOR = re.compile(r"^\s*---\s*$", re.MULTILINE)


def load(name: str, *, strip_notes: bool = True) -> str:
    """
    Load a prompt from prompts/{name}.md.

    Args:
        name: File name without .md (e.g. "classify_query", "evaluate_response")
        strip_notes: If True, remove the "---" line separator and everything after (domain expert notes).

    Returns:
        Prompt text. Use .format() to fill placeholders like {query_text}.
    """
    if not name.endswith(".md"):
        name = f"{name}.md"
    file_path = _PROMPTS_ROOT / name
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt not found: {file_path}")
    text = file_path.read_text(encoding="utf-8")
    if strip_notes:
        match = _NOTES_SEPARATOR.search(text)
        if match:
            text = text[:match.start()].rstrip()
    return text


def load_prompt(agent: str, name: str, *, strip_notes: bool = True) -> str:
    """
    Load a prompt from prompts/{name}.md. Agent is ignored (kept for API compatibility).
    """
    return load(name, strip_notes=strip_notes)


def get_root() -> Path:
    """Return the prompts directory path."""
    return _PROMPTS_ROOT
