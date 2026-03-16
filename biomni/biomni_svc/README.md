# Biomni Agent Integration

Biomedical AI agent integration for Notes9, powered by [Stanford Biomni](https://github.com/snap-stanford/Biomni).

## Quick Start (Local, No Config)

To run Biomni locally with internal storage and no external config:

1. **Install**: `pip install -r requirements.txt` (from `backend/`)
2. **No env vars required** — uses existing AWS Bedrock config (`BEDROCK_CHAT_MODEL_ID`, `AWS_REGION`, etc.)
3. **Optional**: Add `BIOMNI_SKIP_DATALAKE=true` to `.env` for faster startup (skips ~11GB datalake; some tools won't work)
4. **Data path**: Defaults to `backend/data/biomni` — create with `mkdir -p backend/data/biomni` if needed
5. **Start**: `uvicorn main:app` — endpoints at `POST /biomni/run`, `POST /biomni/stream`, `WS /biomni/ws`, `GET /biomni/health`

MCP and session storage are optional; they are disabled when not configured.

## Additional API Keys (Biomni Tools)

Some Biomni tools need extra API keys beyond the main LLM (Bedrock/Anthropic):

| Env Variable | Tool | Purpose |
|--------------|------|---------|
| **ANTHROPIC_API_KEY** | `advanced_web_search_claude` | Web search via Anthropic API. **Required** when the agent uses web search, even if the main LLM is Bedrock. |
| NCBI_EMAIL | PubMed, NCBI Entrez | Email for NCBI API (recommended). |
| SYNAPSE_AUTH_TOKEN | Synapse downloads | Access to Synapse biomedical datasets. |

**Note:** When using Bedrock, the main agent uses AWS. But the literature tool `advanced_web_search_claude` calls the Anthropic API directly, so you must set `ANTHROPIC_API_KEY` to avoid "Set your api_key explicitly" errors during web search.

## Runtime Requirements

- **Python**: >= 3.11 (Dockerfile uses `python:3.11-slim`)
- **Storage**: Local path or EFS (see below)

## Local Storage / EFS

Use `BIOMNI_PATH` with a local directory or EFS mount (~20 GB for the datalake):

```yaml
# Example: ECS task definition volume
volumes:
  - name: biomni-data
    host: {}
volumeMounts:
  - name: biomni-data
    containerPath: /data/biomni
```

For local development: `mkdir -p ./data/biomni`

## Lambda + EFS

When deploying to AWS Lambda with EFS, set `BIOMNI_PATH` to the parent of `biomni_data/`.

**Path structure** (after DataSync from S3 to EFS):

```
/mnt/biomni/                    (Lambda EFS mount path)
└── biomni/
    └── datalake/
        └── biomni_data/
            └── data_lake/
                ├── *.parquet
                ├── *.pkl
                └── *.csv
```

**Lambda configuration:**

| Setting | Value |
|---------|-------|
| `BIOMNI_PATH` | `/mnt/biomni/biomni/datalake` |
| EFS file system | Attach to Lambda (e.g. fs-0c3cd01b44ecffdf0) |
| EFS access point | Use access point with root `/biomni_data` or `/` |
| Local mount path | `/mnt/biomni` |
| VPC | Same VPC as EFS mount targets |

## Environment Variables

Copy `backend/.env.example` to `backend/.env`. Set `BIOMNI_PATH` (or `BIOMNI_DATA_PATH`) to your EFS mount path (e.g. `/mnt/biomni/biomni/datalake` for Lambda). Also set `BIOMNI_LLM` and AWS credentials for Bedrock.

## Security and Hardening

Biomni executes LLM-generated code with full privileges. For production:

1. **Container isolation**: Run the backend in a locked-down container:
   - Non-root user (Dockerfile already uses `appuser`)
   - Minimal filesystem access (only `BIOMNI_PATH` and required temp dirs)
   - Restricted outbound network (whitelist Bedrock, Supabase)

2. **Timeouts**: `BIOMNI_TIMEOUT_SECONDS` limits task duration. Align with API gateway/ALB timeouts.

3. **Secrets**: Store AWS credentials and API keys in AWS Secrets Manager or SSM Parameter Store. Never commit `.env` with production secrets.

4. **Logging**: Structured logs (`biomni_task_started`, `biomni_task_completed`, `biomni_task_failed`, `biomni_task_timeout`) include user_id, session_id, latency, and error details for monitoring.
