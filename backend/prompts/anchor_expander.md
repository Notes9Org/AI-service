# Anchor Expansion

SQL returned structured results with project and experiment IDs. Generate targeted follow-up queries to fetch the rich textual content (lab notes, protocols, experiment narratives) that will give the summarizer the full picture.

## Input

User query: "{user_query}"
Project IDs: {project_ids}
Experiment IDs: {experiment_ids}

## Guidelines

- Generate 1–{max_queries} short, specific retrieval queries (e.g. "lab notes for experiment X", "protocol details for project Y").
- Each query should target a single entity when possible.
- Prefer experiment_id scoping when both project and experiment apply.

## Output

Return ONLY a JSON array:
[
  {{"query_text": "...", "experiment_id": "uuid or null", "project_id": "uuid or null"}}
]

---
<!-- Edit Guidelines freely. -->
