# Biomni Biomedical AI Agent

Standalone Biomni agent service. Deployed separately via `deploy-biomni` workflow.

## Structure

- `biomni_svc/` - Biomni service (agent, stream, session, storage, MCP)
- `biomni_runner/` - CLI runner (deprecated for API; use `POST /biomni/run` etc.)
- `api/` - FastAPI routes
- `services/` - Auth, config, AWS (minimal for Biomni)

## Deployment

- **Branch**: `biomni`
- **Paths**: `biomni/**` or `.github/workflows/deploy-biomni.yml`
- **Environment**: `biomni-production` (ECR_REPOSITORY, LAMBDA_FUNCTION_NAME)
- **Lambda**: Requires VPC and EFS (BIOMNI_PATH on EFS mount)

## Local Run

```bash
cd biomni
pip install -r requirements.txt
uvicorn main:app --reload
```

## Env Vars

See `.env.example`. Required: `SUPABASE_JWT_SECRET`, `AWS_REGION`, `BIOMNI_PATH` (EFS for Lambda).
