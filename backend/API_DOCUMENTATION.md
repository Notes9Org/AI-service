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
    "agent": {
      "run": "/agent/run",
      "normalize_test": "/agent/normalize/test"
    },
    "literature": {
      "search": "/literature/search"
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

## Agent Endpoints

### 1. Run Agent

Execute the full agent pipeline to answer a user query.

**Endpoint:** `POST /agent/run`

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
      "source_type": "lab_note",
      "source_id": "123e4567-e89b-12d3-a456-426614174000",
      "chunk_id": "chunk-123",
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

**Citation Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `source_type` | string | Source type: `"lab_note"`, `"protocol"`, `"report"`, etc. |
| `source_id` | string | Source ID (UUID) |
| `chunk_id` | string | Chunk ID if from RAG (optional) |
| `relevance` | float | Relevance score (0.0 to 1.0) |
| `excerpt` | string | Relevant excerpt from source (optional) |

**Status Codes:**
- `200 OK` - Agent execution completed successfully
- `400 Bad Request` - Invalid request (missing required fields, invalid query)
- `500 Internal Server Error` - Agent execution failed

**Example cURL:**
```bash
curl -X POST "https://your-domain.com/agent/run" \
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

**Endpoint:** `POST /agent/stream`

**Description:** Same as `/agent/run` but returns `text/event-stream`. Events arrive incrementally: `thinking`, `sql`, `rag_chunks`, `token`, `done`, `error`, `ping`.

**Authentication:** Bearer token required (`Authorization: Bearer <access_token>`).

**Request Body:** Same as `/agent/run` (query, session_id, history, options).

**Response:** `text/event-stream` with SSE events. See `frontend-integration/AGENT_STREAM_CLIENT.md` for full event schema and a ready-to-use React hook.

**Verify streaming:**
```bash
export AGENT_STREAM_TOKEN="your-supabase-jwt"
python -m scripts.verify_agent_stream
```

---

### 3. Test Normalize Node

Test the normalization node directly without running the full agent pipeline.

**Endpoint:** `POST /agent/normalize/test`

**Description:** Tests the query normalization step independently. Useful for debugging query understanding and entity extraction.

**Request Body:**
```json
{
  "query": "Find notes about PCR experiments from last week",
  "user_id": "test-user",
  "session_id": "test-session",
  "history": [
    {
      "role": "user",
      "content": "What experiments are running?"
    }
  ]
}
```

**Request Schema:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Query to normalize |
| `user_id` | string | No | User ID (default: `"test-user"`) |
| `session_id` | string | No | Session ID (default: `"test-session"`) |
| `history` | array | No | Conversation history (optional) |

**Response (Success):**
```json
{
  "success": true,
  "input": {
    "query": "Find notes about PCR experiments from last week"
  },
  "output": {
    "intent": "search",
    "normalized_query": "find lab notes about PCR experiments from last week",
    "entities": {
      "keywords": ["PCR", "experiments"],
      "time_range": "last week",
      "source_type": "lab_note"
    },
    "context": {
      "requires_semantic_search": true
    },
    "history_summary": "User previously asked about running experiments"
  }
}
```

**Response (Error):**
```json
{
  "success": false,
  "error": "Failed to normalize query: ..."
}
```

**Response Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether normalization succeeded |
| `input` | object | Original input query |
| `output` | object | Normalized output (only if `success = true`) |
| `error` | string | Error message (only if `success = false`) |

**Output Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `intent` | string | Query intent: `"aggregate"`, `"search"`, or `"hybrid"` |
| `normalized_query` | string | Cleaned and normalized query text |
| `entities` | object | Extracted entities (dates, IDs, keywords, etc.) |
| `context` | object | Conversation context and metadata |
| `history_summary` | string | Summary of relevant conversation history (optional) |

**Status Codes:**
- `200 OK` - Request processed (check `success` field)

**Example cURL:**
```bash
curl -X POST "https://your-domain.com/agent/normalize/test" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find notes about PCR experiments from last week",
    "user_id": "test-user",
    "session_id": "test-session"
  }'
```

---

## Literature Search Endpoints

### 1. Search Literature

Search for scientific literature using Brave Search API.

**Endpoint:** `GET /literature/search`

**Description:** Searches for scientific papers and literature related to a topic. Returns structured results with metadata including title, authors, year, journal, abstract, DOI, PMID, and PDF URLs.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `q` | string | Yes | Search query for literature topic (min length: 1) |
| `limit` | integer | No | Maximum number of results (1-50, default: 20) |

**Request:**
```http
GET /literature/search?q=CRISPR gene editing&limit=10 HTTP/1.1
Host: your-domain.com
```

**Response:**
```json
{
  "papers": [
    {
      "id": "paper-1",
      "title": "CRISPR-Cas9: A Revolutionary Gene Editing Tool",
      "authors": ["Jennifer Doudna", "Emmanuelle Charpentier"],
      "year": 2020,
      "journal": "Nature",
      "abstract": "CRISPR-Cas9 has revolutionized the field of gene editing...",
      "isOpenAccess": true,
      "doi": "10.1038/s41586-020-2442-0",
      "pmid": "32709965",
      "pdfUrl": "https://example.com/paper.pdf",
      "url": "https://www.nature.com/articles/s41586-020-2442-0",
      "source": "brave"
    },
    {
      "id": "paper-2",
      "title": "Applications of CRISPR in Biomedical Research",
      "authors": ["Feng Zhang"],
      "year": 2021,
      "journal": "Science",
      "abstract": "This review discusses the applications of CRISPR...",
      "isOpenAccess": false,
      "doi": "10.1126/science.abc1234",
      "pmid": null,
      "pdfUrl": null,
      "url": "https://www.science.org/doi/10.1126/science.abc1234",
      "source": "brave"
    }
  ],
  "totalCount": 10,
  "source": "brave"
}
```

**Response Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `papers` | array | List of paper objects |
| `totalCount` | integer | Number of results returned |
| `source` | string | Search source identifier (always `"brave"`) |

**Paper Object Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique paper identifier |
| `title` | string | Paper title |
| `authors` | array | List of author names |
| `year` | integer | Publication year |
| `journal` | string | Journal name |
| `abstract` | string | Paper abstract |
| `isOpenAccess` | boolean | Whether paper is open access |
| `doi` | string | Digital Object Identifier (optional) |
| `pmid` | string | PubMed ID (optional) |
| `pdfUrl` | string | Direct PDF URL (optional) |
| `url` | string | Paper URL |
| `source` | string | Source identifier |

**Status Codes:**
- `200 OK` - Search completed successfully
- `400 Bad Request` - Invalid query (empty or too short)
- `500 Internal Server Error` - Search failed

**Example cURL:**
```bash
curl "https://your-domain.com/literature/search?q=CRISPR%20gene%20editing&limit=10"
```

**Example with URL encoding:**
```bash
curl "https://your-domain.com/literature/search?q=machine%20learning%20in%20biology&limit=20"
```

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
5. **Use appropriate `limit` values** for literature search (1-50)
6. **Monitor `/health/ready`** before sending production traffic

---

## Support

For issues or questions:
- Check CloudWatch logs for detailed error messages
- Review the interactive API documentation at `/docs`
- Verify environment variables are correctly set in ECS Task Definition
