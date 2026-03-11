# BioMni Agent Runner

Standalone BioMni biomedical AI agent integration using AWS Bedrock and the local data lake at `backend/data/biomni/biomni_data/`.

## API Endpoint (Swagger UI)

Start the backend server and open **[http://localhost:8000/docs](http://localhost:8000/docs)** to test via Swagger UI.


| Method | Path             | Description                                     |
| ------ | ---------------- | ----------------------------------------------- |
| POST   | `/biomni/query`  | Run a biomedical query through the BioMni agent |
| GET    | `/biomni/health` | Check agent config (data path, model)           |


**Example request** (`POST /biomni/query`):

```json
{
  "query": "Predict ADMET properties for aspirin",
  "max_retries": 3
}
```

**Example response**:

```json
{
  "answer": "Based on the analysis ...",
  "steps": [
    { "content": "Step 1 log ..." },
    { "content": "Step 2 log ..." }
  ]
}
```

## CLI Usage

From the `backend/` directory:

```bash
# Single query
python -m biomni_runner.run --query "Predict ADMET properties for aspirin"

# Interactive mode
python -m biomni_runner.run --interactive

# Custom model or data path
python -m biomni_runner.run --query "..." --model "anthropic.claude-sonnet-4-20250514-v1:0" --data-path /path/to/data

# Skip loading datalake (faster startup, some tools won't work)
python -m biomni_runner.run --query "..." --no-load-datalake
```

## Environment

- Uses existing backend `.env` (AWS credentials, Bedrock config).
- Optional: `BIOMNI_DATA_PATH` (parent of biomni_data/, e.g. backend/data/biomni), `BIOMNI_LLM_MODEL`, `BIOMNI_TIMEOUT_SECONDS`.

## Troubleshooting

`**convert_to_openai_data_block` import error:** BioMni requires `langchain-core>=0.3.56`. Upgrade:

```bash
pip install 'langchain-core>=0.3.56' 'pydantic>=2.7.4'
```

`**langchain-aws` required for Bedrock:** BioMni needs `langchain-aws` for Bedrock models:

```bash
pip install langchain-aws
```

**Dependency conflicts (Python 3.12.4+):** langchain-core 0.3.x requires `pydantic>=2.7.4` on Python 3.12.4+. Reinstall from requirements:

```bash
pip install -r requirements.txt --upgrade
```

## Note

The package is named `biomni_runner` (not `biomni`) to avoid shadowing the installed `biomni` PyPI package.