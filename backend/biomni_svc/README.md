# Biomni Agent Integration

Biomedical AI agent integration for Notes9, powered by [Stanford Biomni](https://github.com/snap-stanford/Biomni).

## Runtime Requirements

- **Python**: >= 3.11 (Dockerfile uses `python:3.11-slim`)
- **Storage**: Either S3 (direct) or local path (see below)

## S3 Datalake (Direct Access, No Download)

When `BIOMNI_S3_DATALAKE_BUCKET` is set, the agent path becomes `s3://bucket/prefix`. Biomni reads datalake files directly from S3 on demand via s3fs—no local copy or download.

**IAM**: `s3:GetObject`, `s3:ListBucket` on the bucket (e.g. `arn:aws:s3:::biomnidatalake`).

**S3 structure** (must match): `s3://bucket/prefix/biomni_data/data_lake/*.parquet`, etc.

If Biomni fails on non-parquet/CSV files (e.g. pickle), use the FUSE mount fallback: mount S3 to a directory and set `BIOMNI_PATH` to that mount. Requires `--device /dev/fuse --cap-add SYS_ADMIN` in Docker.

## Local Storage (Alternative)

If `BIOMNI_S3_DATALAKE_BUCKET` is not set, use `BIOMNI_PATH` with a local or mounted volume (~20 GB for the datalake):

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

## Environment Variables

Copy `backend/.env.example` to `backend/.env`. For S3: set `BIOMNI_S3_DATALAKE_BUCKET=biomnidatalake` and `BIOMNI_S3_DATALAKE_PREFIX=biomni/datalake`. Otherwise set `BIOMNI_PATH`, `BIOMNI_LLM`, and AWS credentials for Bedrock.

## Security and Hardening

Biomni executes LLM-generated code with full privileges. For production:

1. **Container isolation**: Run the backend in a locked-down container:
   - Non-root user (Dockerfile already uses `appuser`)
   - Minimal filesystem access (only `BIOMNI_PATH` and required temp dirs)
   - Restricted outbound network (whitelist Bedrock, S3, Supabase)

2. **Timeouts**: `BIOMNI_TIMEOUT_SECONDS` limits task duration. Align with API gateway/ALB timeouts.

3. **Secrets**: Store AWS credentials and API keys in AWS Secrets Manager or SSM Parameter Store. Never commit `.env` with production secrets.

4. **Logging**: Structured logs (`biomni_task_started`, `biomni_task_completed`, `biomni_task_failed`, `biomni_task_timeout`) include user_id, session_id, latency, and error details for monitoring.
