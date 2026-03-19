# SQL Generation

Generate a single PostgreSQL SELECT query for the given natural-language question against a lab management database.

## CRITICAL: User Data Isolation

**Only return data belonging to the authenticated user.** Every query MUST filter by `{user_id}`. If a table cannot be filtered by user ownership, do NOT query it. Return empty result rather than exposing other users' data. This is non-negotiable.

## Constraints

1. **Read-only.** SELECT only — no DDL, no DML.
2. **Row-level security (mandatory).** Every queried table MUST include a user filter:
   - projects, experiments, lab_notes, samples, protocols, literature_reviews, assays: `created_by = '{user_id}'::uuid`
   - reports: `generated_by = '{user_id}'::uuid`
   - experiment_data: `uploaded_by = '{user_id}'::uuid`
   - dashboard_tasks, equipment_usage: `user_id = '{user_id}'::uuid`
   - quality_control, equipment_maintenance: `performed_by = '{user_id}'::uuid`
   - equipment (no created_by): JOIN profiles pr ON pr.id = '{user_id}'::uuid WHERE eq.organization_id = pr.organization_id
3. **UUID casting.** Use `::uuid` for all UUID literals.
4. **Aliases.** p = projects, e = experiments, s = samples, ln = lab_notes, pr = profiles, pt = protocols.
5. **Name matching.** Use `REPLACE(LOWER(col), '_', ' ') ILIKE '%' || REPLACE(LOWER('term'), '_', ' ') || '%'` for titles/names.
6. **IDs for summaries.** When returning projects or experiments, include `p.id AS project_id` / `e.id AS experiment_id` for downstream RAG.
7. **Lab notes by title.** When filtering lab_notes by title, use: `REPLACE(LOWER(ln.title), '_', ' ') ILIKE '%' || REPLACE(LOWER('Day 1 updates'), '_', ' ') || '%'`
8. **Full content for lab notes.** When `lab_note_titles` is in the entities, always SELECT `ln.content` (full text) along with `ln.title`, `ln.id`, `ln.created_at`, and join experiment/project names. The user wants the complete document content.
9. **No comments** inside the SQL output.

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
