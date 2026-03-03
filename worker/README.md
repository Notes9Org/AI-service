# Chunk Worker

Processes pending `chunk_jobs` from Supabase into semantic chunks (chunking + embeddings).

## Local run

```bash
cd worker
cp .env.example .env   # edit with your credentials
pip install -r requirements.txt
python worker.py
```

## Options

- `--once` - Process one batch and exit
- `--batch-size N` - Jobs per batch (default: 10)
- `--poll-interval N` - Seconds between polls when idle (default: 5)
- `--retry-failed` - Reset failed jobs to pending
- `--retry-limit N` - Limit jobs to retry
- `--max-retries N` - Only retry jobs with retry_count <= N

## Docker (local)

Ensure `worker/.env` exists (copy from backend or create from `.env.example`):

```bash
cd worker
cp .env.example .env   # or: cp ../backend/.env .env
# Edit .env with your credentials
docker-compose -f docker-compose.worker.yml up
```

Or run directly with env file:

```bash
docker run --rm --env-file worker/.env notes9-worker:local
```

---

## Lambda Deployment Plan

### AWS Setup

#### 1. ECR Repository
- Go to **ECR** → Create repository
- Name: `notes9-worker` (or use existing)
- Note the repository URI for the deploy workflow

#### 2. Lambda Function
- Go to **Lambda** → Create function
- **Author from scratch**
- Function name: `notes9-chunk-worker`
- Runtime: **Container image**
- Create container image (or "Use an existing image" after first push)
- **Architecture**: x86_64
- **Timeout**: 300 seconds (5 min) or 900 (15 min) for large batches
- **Memory**: 2048 MB
- **Ephemeral storage**: 512 MB (or more if needed)

#### 3. Lambda Environment Variables
Set in Lambda → Configuration → Environment variables:

| Variable | Description |
|----------|-------------|
| `NEXT_PUBLIC_SUPABASE_URL` | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key |
| `LLM_PROVIDER` | `bedrock` or `azure` |
| `AWS_REGION` | e.g. `us-east-1` |
| `BEDROCK_EMBEDDING_MODEL_ID` | e.g. `cohere.embed-v4:0` |
| `EMBEDDING_DIMENSIONS` | e.g. `1536` |
| `CHUNK_SIZE` | Optional, default `1000` |
| `CHUNK_OVERLAP` | Optional, default `200` |

For Azure: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`, etc.

#### 4. Lambda Execution Role (IAM)
Ensure the role has:
- `AWSLambdaBasicExecutionRole` (CloudWatch Logs)
- Bedrock: `bedrock:InvokeModel` on the embedding model
- ECR: `ecr:GetDownloadUrlForLayer`, `ecr:BatchGetImage`, `ecr:BatchCheckLayerAvailability`

#### 5. Trigger Options

**Option A: EventBridge Schedule (polling)**
- **EventBridge** → Rules → Create rule
- Schedule: `rate(2 minutes)` or `cron(0/2 * * * ? *)`
- Target: your Lambda function
- Payload: `{}`

**Option B: Supabase Webhook (event-driven)**
- **API Gateway** → Create HTTP API
- Create route: `POST /process-jobs` → integrate with Lambda
- Use the API URL in Supabase Database Webhooks (see below)

---

### Supabase Setup

#### If using EventBridge (polling)
- No Supabase changes needed
- Lambda polls `chunk_jobs` on schedule

#### If using Supabase Webhook (event-driven)
1. **Database** → **Webhooks** → Create new webhook
2. **Name**: `trigger-chunk-worker`
3. **Table**: `chunk_jobs`
4. **Events**: `Insert`
5. **Type**: `HTTP Request`
6. **URL**: API Gateway URL (e.g. `https://xxx.execute-api.us-east-1.amazonaws.com/process-jobs`)
7. **HTTP method**: `POST`
8. **Headers**: add auth header (e.g. `x-webhook-secret`) if you add auth in Lambda

---

### Lambda Handler

The worker needs a Lambda entrypoint. Add `handler.py`:

```python
# worker/handler.py
from worker import ChunkWorker

def handler(event, context):
    worker = ChunkWorker()
    processed = worker.run_once()
    return {"processed": processed}
```

Set Lambda container **CMD** to `handler.handler` instead of `python worker.py`.

---

### Deploy Workflow

Add to `.github/workflows/deploy-worker.yml`:
1. Update Lambda function code with new ECR image
2. Optionally sync environment variables

---

### Summary Checklist

| Where | Action |
|-------|--------|
| **AWS ECR** | Create `notes9-worker` repo |
| **AWS Lambda** | Create function from container image, set timeout/memory and env vars |
| **AWS IAM** | Ensure Lambda role has Bedrock access |
| **AWS EventBridge** | Create rule to invoke Lambda on schedule (if polling) |
| **AWS API Gateway** | Create HTTP API + Lambda integration (if webhook) |
| **Supabase** | Create webhook on `chunk_jobs` INSERT → API Gateway URL (if webhook) |
| **GitHub** | Add `ECR_REPOSITORY_WORKER` and `LAMBDA_FUNCTION_NAME_WORKER` secrets |

---

## CI/CD

Push to `worker` branch to build and push the Docker image to ECR.

Configure `ECR_REPOSITORY_WORKER` (or `ECR_REPOSITORY`) and `LAMBDA_FUNCTION_NAME_WORKER` secrets for deployment.
