# SQL Generation

Generate a single PostgreSQL SELECT query for the given natural-language question against a lab management database.

## Constraints

1. **Read-only.** SELECT only — no DDL, no DML.
2. **Row-level security.** Every queried table MUST include `WHERE created_by = '{user_id}'::uuid` (or `generated_by` for reports). This is non-negotiable.
3. **UUID casting.** Use `::uuid` for all UUID literals.
4. **Aliases.** p = projects, e = experiments, s = samples, pr = profiles.
5. **Name matching.** Use `REPLACE(LOWER(col), '_', ' ') ILIKE '%' || REPLACE(LOWER('term'), '_', ' ') || '%'` to handle underscores and case.
6. **IDs for summaries.** When returning projects or experiments, always include `p.id AS project_id` / `e.id AS experiment_id` so downstream enrichment can use them.
7. **No comments** inside the SQL output.

## Input

**Query:** {query}
{normalized_section}
**User ID:** {user_id}

**Entities:** {entities_text}
{entity_section}

**Schema:**
{schema}

## Output

Return ONLY the raw SQL. No markdown fences, no explanation.

---
<!-- Edit Constraints and examples freely. Schema is injected by the system — do not modify the placeholder. -->
