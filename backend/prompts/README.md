# Prompts

All agent prompts live here as markdown files. Domain experts can edit them without touching Python.

## Files

| File | Used by |
|------|---------|
| classify_query.md | normalize |
| out_of_scope_response.md | normalize |
| generate_query.md | sql |
| summarize_results.md | summarizer |
| evaluate_response.md | judge |
| anchor_expander.md | enrichment |
| rewrite_query.md | retry |

## Format

Each `.md` file has:
1. **Content** — Role, Input, Task, Output Format. Use `{placeholder}` for dynamic values.
2. **`---`** — Separator (everything after is stripped before sending to the LLM).
3. **Notes** — Instructions for domain experts (optional).

## Loading in Code

```python
from agents.prompt_loader import load, load_prompt

# By name
prompt = load("evaluate_response")

# By agent + name (agent ignored, kept for compatibility)
prompt = load_prompt("judge", "evaluate_response")

# Fill placeholders
text = prompt.format(original_query="...", answer="...", ...)
```

## Editing Guidelines

- Edit Role, Task, and Instructions freely.
- Do not change Output Format (JSON keys, structure) without coordinating with engineering.
- Use Git to track changes.
