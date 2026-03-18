"""
Centralized constants for the agent graph. Reduces hardcoded logic and magic numbers.
Tool names, verdicts, and routes match contracts (router, response, normalized).
"""

# --- Tool names (must match RouterDecision.tools and FinalResponse.tool_used) ---
TOOL_SQL = "sql"
TOOL_RAG = "rag"
TOOL_HYBRID = "hybrid"
TOOL_NONE = "none"

TOOLS_ALL = (TOOL_SQL, TOOL_RAG)

# --- Entity keys used for SQL fallback / RAG-weak routing (single source of truth) ---
ENTITY_KEYS_FOR_SQL_FALLBACK = (
    "experiment_ids",
    "project_ids",
    "experiment_names",
    "project_names",
    "lab_note_titles",
    "protocol_names",
    "dates",
    "statuses",
    "person_names",
)

# --- Route kinds (must match RouterDecision.route) ---
ROUTE_IN_SCOPE = "in_scope"
ROUTE_OUT_OF_SCOPE = "out_of_scope"

# --- Judge verdicts ---
VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"

# --- Retry / graph ---
DEFAULT_MAX_RETRIES = 2
RECURSION_LIMIT = 50

# --- Confidence defaults (when judge not used or fallback) ---
CONFIDENCE_JUDGE_PASS_DEFAULT = 0.7
CONFIDENCE_HYBRID_BONUS = 0.1
CONFIDENCE_SQL_ONLY = 0.9
CONFIDENCE_RAG_ONLY = 0.6
CONFIDENCE_ERROR_OR_MISSING = 0.0
CONFIDENCE_OUT_OF_SCOPE = 0.0
CONFIDENCE_ROUTER_FALLBACK = 0.4
CONFIDENCE_ROUTER_OUT_OF_SCOPE = 0.95

# --- Anchor expander / enrichment ---
MAX_ENRICHMENT_QUERIES = 5
MAX_ANCHOR_QUERIES_TOTAL = 14
ENRICHMENT_MATCH_COUNT = 6
ENRICHMENT_THRESHOLD = 0.20
ENRICHMENT_CONTENT_TRUNCATE = 1500

# --- RAG node: weak-content gate (min avg content length to not mark rag_weak) ---
RAG_WEAK_MIN_AVG_CONTENT_LEN = 80
RAG_CHUNK_CONTENT_TRUNCATE = 1500
RAG_TOP_CHUNKS = 6
# When we have SQL IDs, fetch this many chunks per experiment/project (UUID filter)
RAG_TOP_CHUNKS_PER_ENTITY = 3
RAG_MAX_ENTITIES_FOR_ID_FETCH = 10
RAG_MAX_CHUNKS_PER_SECTION = 10
# Deep fetch: when user asks to extract/pull/fetch content from a named document
RAG_DEEP_FETCH_PER_ENTITY = 12   # Chunks per entity in deep fetch mode (vs 3 normal)
RAG_DEEP_FETCH_TOP = 16          # Final chunks to keep in deep fetch mode (vs 6 normal)

# --- Summarizer (keep prompt bounded so LLM can handle it) ---
SUMMARIZER_TEMPERATURE = 0.3
SUMMARIZER_ENRICHED_PREVIEW_COUNT = 10
SUMMARIZER_ENRICHED_CONTENT_LEN = 1000
SUMMARIZER_RAG_CONTENT_LEN = 1500
SUMMARIZER_ANSWER_PREVIEW_LEN = 200
# Caps to avoid context overflow and LLM errors (RetryError/LLMError)
SUMMARIZER_SQL_MAX_ROWS = 50
SUMMARIZER_SQL_MAX_CELL_LEN = 400
SUMMARIZER_SQL_MAX_CELL_LEN_FULL_CONTENT = 15_000  # For full lab note content retrieval
SUMMARIZER_RAG_MAX_CHUNKS = 16
SUMMARIZER_PROMPT_MAX_CHARS = 120_000  # ~30k tokens; stay under typical 32k context

# --- SQL analyzer ---
SQL_ANALYZER_MAX_IDS = 50

# --- Truncation for logs/traces (shared) ---
LOG_QUERY_PREVIEW_LEN = 100
LOG_QUERY_LEN = 200
LOG_SUGGESTED_REVISION_LEN = 200
