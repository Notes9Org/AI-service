# Deployment Secrets Setup

Each deployment uses its own **GitHub Environment** so workflows only use their respective secrets.

## Environments


| Environment          | Workflow          | Branch | Purpose                                    |
| -------------------- | ----------------- | ------ | ------------------------------------------ |
| `backend-production` | deploy.yml        | main   | Backend (FastAPI) → Lambda notes9-agent    |
| `worker-production`  | deploy-worker.yml | worker | Chunk worker → Lambda notes9-worker-lambda |


## Setup

### 1. Create environments

**Settings** → **Environments** → **New environment**

- Create `backend-production`
- Create `worker-production`

### 2. Add repository secrets (shared)

**Settings** → **Secrets and variables** → **Actions** → **Repository secrets**


| Secret                  | Value                                           |
| ----------------------- | ----------------------------------------------- |
| `AWS_ACCESS_KEY_ID`     | Your AWS access key                             |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret key                             |
| `AWS_REGION`            | e.g. `us-east-1` (optional, default: us-east-1) |


### 3. Add environment secrets

**Settings** → **Environments** → select environment → **Add secret**

**backend-production:**


| Secret                 | Value             |
| ---------------------- | ----------------- |
| `ECR_REPOSITORY`       | `notes9-catalyst` |
| `LAMBDA_FUNCTION_NAME` | `notes9-agent`    |


**worker-production:**


| Secret                 | Value                  |
| ---------------------- | ---------------------- |
| `ECR_REPOSITORY`       | `notes9-worker`        |
| `LAMBDA_FUNCTION_NAME` | `notes9-worker-lambda` |


## Migrating from "AWS connection"

If you already use the `AWS connection` environment:

1. Create `backend-production` and add the same secrets as in `AWS connection`.
2. Create `worker-production` and add worker-specific values.
3. The workflows now reference these environments.

## Result

- **deploy.yml** (main): uses only `backend-production` secrets.
- **deploy-worker.yml** (worker): uses only `worker-production` secrets.
- No cross-use of ECR or Lambda names between deployments.

