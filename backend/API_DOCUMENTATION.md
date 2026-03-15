# Notes9 AI Service API Documentation


### Root Endpoint

Get service information and available endpoints.

**Endpoint:** `GET /`

**Description:** Root endpoint with service information and available API endpoints.

**Request:**
```http
GET / HTTP/1.1
Host: your-domain.com
```

**Response:**
```json
{
  "service": "Notes9 Agent Chat Service",
  "version": "1.0.0",
  "status": "operational",
  "endpoints": {
    "notes9": {
      "run": "/notes9/run",
      "stream": "/notes9/stream"
    },
    "chat": {
      "post": "/chat",
      "stream": "/chat/stream"
    },
    "monitoring": {
      "health": "/health",
      "readiness": "/health/ready"
    },
    "documentation": {
      "swagger": "/docs",
      "redoc": "/redoc"
    }
  }
}
```

**Status Codes:**
- `200 OK` - Success

---

## Notes9 Agent Endpoints

### 1. Run Agent

Execute the full agent pipeline to answer a user query.

**Endpoint:** `POST /notes9/run`

**Description:** Executes the complete agent graph: normalize → router → tools (SQL/RAG) → summarizer → judge → final response. The agent intelligently routes queries to SQL (for structured data queries) or RAG (for semantic search), or both.

**Request Body:**
```json
{
  "query": "How many experiments were completed last month?",
  "user_id": "user-123",
  "session_id": "session-456",
  "history": [
    {
      "role": "user",
      "content": "What experiments are in progress?"
    },
    {
      "role": "assistant",
      "content": "There are 5 experiments currently in progress."
    }
  ],
  "scope": {
    "organization_id": "cedbb951-4b9f-440a-96ad-0373fe059a1b",
    "project_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "experiment_id": "f1e2d3c4-b5a6-9876-5432-10fedcba9876"
  },
  "options": {
    "debug": false,
    "max_retries": 2
  }
}
```

**Request Schema:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | User query to process |
| `user_id` | string | Yes | User ID for tracking and context |
| `session_id` | string | Yes | Session ID for tracking and context |
| `history` | array | No | Previous messages in the conversation (default: `[]`) |
| `scope` | object | No | Access scope (deprecated, not used for filtering) |
| `options` | object | No | Options: `debug` (bool), `max_retries` (int), etc. |

**History Item Schema:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | string | Yes | Message role: `"user"` or `"assistant"` |
| `content` | string | Yes | Message content |

**Options Schema:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `debug` | boolean | No | Include debug trace in response (default: `false`) |
| `max_retries` | integer | No | Maximum retry attempts if judge fails (default: `2`) |

**Response:**
```json
{
  "answer": "Based on the database records, 12 experiments were completed last month (January 2024).",
  "citations": [
    {
      "display_label": "Lab note: PCR Protocol",
      "source_type": "lab_note",
      "source_name": "PCR Protocol",
      "relevance": 0.95,
      "excerpt": "Experiment completed on January 15, 2024..."
    }
  ],
  "confidence": 0.92,
  "tool_used": "sql",
  "debug": {
    "normalize": {
      "intent": "aggregate",
      "normalized_query": "count experiments completed in last month",
      "latency_ms": 450
    },
    "router": {
      "tools": ["sql"],
      "confidence": 0.95,
      "latency_ms": 320
    },
    "sql": {
      "generated_sql": "SELECT COUNT(*) FROM experiments WHERE ...",
      "row_count": 12,
      "latency_ms": 1200
    }
  }
}
```

**Response Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `answer` | string | Generated answer to the user query |
| `citations` | array | List of source citations (empty if SQL-only) |
| `confidence` | float | Confidence score (0.0 to 1.0) |
| `tool_used` | string | Tool(s) used: `"sql"`, `"rag"`, or `"hybrid"` |
| `debug` | object | Debug trace (only if `options.debug = true`) |

**Citation Schema (response omits source_id and chunk_id; those are stored server-side for reference):**

| Field | Type | Description |
|-------|------|-------------|
| `display_label` | string | Human-readable label for UI (e.g. "Lab note: PCR Protocol"). |
| `source_type` | string | Source type: `"lab_note"`, `"protocol"`, `"report"`, etc. |
| `source_name` | string | Document name (e.g. lab note title). |
| `relevance` | float | Relevance score (0.0 to 1.0) |
| `excerpt` | string | Relevant excerpt from source (optional) |

**Status Codes:**
- `200 OK` - Agent execution completed successfully
- `400 Bad Request` - Invalid request (missing required fields, invalid query)
- `500 Internal Server Error` - Agent execution failed

**Example cURL:**
```bash
curl -X POST "https://your-domain.com/notes9/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT" \
  -d '{
    "query": "How many experiments were completed last month?",
    "session_id": "session-456",
    "history": [],
    "options": {
      "debug": false,
      "max_retries": 2
    }
  }'
```

---

### 2. Stream Agent (SSE)

Execute the agent with Server-Sent Events for live, Cursor-style streaming (thinking steps, SQL, RAG chunks, answer tokens).

**Endpoint:** `POST /notes9/stream`

**Description:** Same as `/notes9/run` but returns `text/event-stream`. Events arrive incrementally: `thinking`, `sql`, `rag_chunks`, `token`, `done`, `error`, `ping`.

**Authentication:** Bearer token required (`Authorization: Bearer <access_token>`).

**Request Body:** Same as `/notes9/run` (query, session_id, history, options).

**Response:** `text/event-stream` with SSE events. See `frontend-integration/AGENT_STREAM_CLIENT.md` for full event schema and a ready-to-use React hook.

**Verify streaming:**
```bash
export AGENT_STREAM_TOKEN="your-supabase-jwt"
python -m scripts.verify_agent_stream
```

---

## Chat Endpoints

Direct Claude/LLM chat (no agent pipeline). Uses same input pattern as notes9: content, session_id, history.

### 1. Chat (non-streaming)

**Endpoint:** `POST /chat`

**Description:** Send user message and optional history; receive full assistant reply.

**Request Body:** `{ "content": "...", "session_id": "...", "history": [{ "role": "user"|"assistant", "content": "..." }] }`

**Response:** `{ "content": "...", "role": "assistant" }`

**Authentication:** Bearer token required.

### 2. Chat Stream (SSE)

**Endpoint:** `POST /chat/stream`

**Description:** Same as `/chat` but returns `text/event-stream`. Events: `token` (incremental text), `done` (full content, role), `error`.

**Request Body:** Same as `/chat`.

**Response:** SSE stream. Events: `token` → `{"text": "..."}`, `done` → `{"content": "...", "role": "assistant"}`, `error` → `{"error": "..."}`.

**Authentication:** Bearer token required.

---

## Biomni Endpoints

Biomedical AI agent powered by [Stanford Biomni](https://github.com/snap-stanford/Biomni). All mutation endpoints require Bearer token (Supabase Auth).

**Aligned with agent/run:** Biomni uses the same input pattern: `query`, `session_id`, `history`, `options`. `user_id` is always from the Bearer token (never from request body).

### 1. Run Biomni Task

**Endpoint:** `POST /biomni/run`

**Description:** Execute a biomedical research task. Supports clarification questions, PDF generation, and session persistence.

**Request Body:**
```json
{
  "query": "Predict ADMET properties for aspirin",
  "session_id": "session-456",
  "max_retries": 3,
  "history": [],
  "options": {
    "skip_clarify": false,
    "max_clarify_rounds": 2,
    "generate_pdf": false
  }
}
```

Use `query` (same as agent/run) or `prompt` (backwards compat). Either is required. `user_id` comes from Bearer token only.

**Response:** `result`, `success`, `error`, `steps`, `artifact_url`, `pdf_url`, `clarify_question`, `clarify_options`

**Status Codes:** `200 OK`, `503` (Biomni not installed)

---

### 2. Stream Biomni Task (SSE)

**Endpoint:** `POST /biomni/stream`

**Description:** Same as `/biomni/run` but returns `text/event-stream`. Events: `started`, `step`, `clarify`, `result`, `error`, `ping`, `done`.

**Request Body:** Same as `/biomni/run`.

**Response:** SSE stream. See `frontend-integration/BIOMNI_STREAM_CLIENT.md` for event schema and React hooks.

---

### 3. Biomni WebSocket

**Endpoint:** `WS /biomni/ws`

**Description:** Bidirectional streaming. Connect with `?token=JWT` or send `{"type": "auth", "token": "..."}` first. Send `{"type": "run", "query": "...", "session_id": "..."}` to execute (or use `prompt`). `user_id` from token. Supports clarifying questions: server sends `{"type": "clarify", "question": "...", "options": [...]}`, client responds with `{"type": "clarify_response", "answer": "..."}`.

---

### 4. Biomni Health

**Endpoint:** `GET /biomni/health`

**Description:** Lightweight check that the agent can be initialized. No auth required.

---

### 5. Sessions

**Endpoints:**
- `GET /biomni/sessions` — List user's sessions
- `GET /biomni/sessions/{session_id}` — Get session with all runs
- `GET /biomni/sessions/{session_id}/pdf` — Download session PDF report

---

### 6. MCP

**Endpoints:**
- `GET /biomni/mcp/servers` — List registered MCP servers and tools
- `GET /biomni/mcp/health` — Test MCP connections

---

## Error Responses

All endpoints may return standard HTTP error responses with the following format:

### 400 Bad Request
```json
{
  "error": "Validation error",
  "details": [
    {
      "loc": ["body", "query"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 422 Unprocessable Entity
```json
{
  "error": "Validation error",
  "details": [
    {
      "loc": ["body", "query"],
      "msg": "ensure this value has at least 1 characters",
      "type": "value_error.any_str.min_length"
    }
  ]
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal server error"
}
```

Or with specific error message:
```json
{
  "error": "Agent execution failed: Database connection timeout"
}
```

---

## Interactive API Documentation

The service provides interactive API documentation:

- **Swagger UI:** `GET /docs`
- **ReDoc:** `GET /redoc`
- **OpenAPI JSON:** `GET /openapi.json`

Visit these endpoints in your browser to explore the API interactively.

---

## Rate Limiting

Currently, there are no rate limits implemented. In production, consider implementing:
- Rate limiting per user/IP
- Request throttling
- Quota management

---

## Best Practices

1. **Always include `user_id` and `session_id`** in agent requests for proper tracking
2. **Use conversation history** for context-aware responses
3. **Set `debug: true`** during development to see detailed execution traces
4. **Handle errors gracefully** - check status codes and error messages
5. **Monitor `/health/ready`** before sending production traffic

---

## Support

For issues or questions:
- Check CloudWatch logs for detailed error messages
- Review the interactive API documentation at `/docs`
- Verify environment variables are correctly set in ECS Task Definition


