# News Topic Classification & Summarization (MVP)

This repository hosts a FastAPI microservice that:

- Classifies news article text into topics (AG News baseline to start)
- Abstractive summarization via DistilBART (`sshleifer/distilbart-cnn-12-6`) with configurable generation params
- Provides clean JSON endpoints for easy integration

## Current Status (MVP Scaffold)

- [x] Directory structure
- [x] Baseline training script (`scripts/train_baseline.py`) for TF-IDF + Logistic Regression on AG News
- [x] Pydantic schemas
- [x] FastAPI endpoints: `/health`, `/classify_news`, `/summarize` (HF DistilBART)
- [ ] Trend aggregation + persistence (future)

## Quickstart (Windows PowerShell)

```powershell
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# Train baseline (downloads AG News via Hugging Face datasets)
python scripts/train_baseline.py --output-dir models/classifier --limit 5000

# Run API
uvicorn app.main:app --reload --port 8000
```

### Database Migrations & Seed Data

After installing dependencies, bring your SQLite schema up to date and load demo
rows for the feedback/review queues:

```powershell
python -m alembic -c alembic.ini upgrade head
python scripts/seed_db.py  # add --overwrite to refresh demo content
```

### Run with Docker

Build the image and start the API (migrations run automatically on boot):

```powershell
docker compose up --build
```

This mounts the local `data/` and `models/` directories into the container so the
SQLite database and model artifacts persist between restarts.

### Unified CLI (Training, Evaluation, Ops)

The Typer-based CLI consolidates the training/evaluation utilities and safe DB
seeding behind a single entry point:

```powershell
# populate demo data (creates tables if needed)
python scripts/manage.py seed-db --overwrite

# train the lightweight TF-IDF + logistic regression model
python scripts/manage.py train-baseline --limit 2000 --output-dir models/classifier_huffpost

# fine-tune a transformer (AG News)
python scripts/manage.py train-transformer --model-name distilbert-base-uncased

# evaluate a trained transformer against the HuffPost dataset
python scripts/manage.py eval-transformer models/transformer_huffpost data/raw/huffpost/News_Category_Dataset_v3.json

# bundle local model artifacts into a timestamped zip
python scripts/manage.py bundle-artifacts --sources models,classifier --label nightly

# push a bundle to remote storage (reads ARTIFACT_* env vars)
python scripts/manage.py bundle-artifacts --push --remote-name nightly.zip
```

### Artifact publishing to object storage

The `bundle-artifacts` command can ship the generated archive to an S3-compatible
bucket. Configure the target via `.env`:

```env
ARTIFACT_STORE_TYPE=s3
ARTIFACT_S3_BUCKET=news-artifacts
# Optional
ARTIFACT_S3_REGION=us-east-1
ARTIFACT_S3_ENDPOINT=http://localhost:9000  # for MinIO or other custom endpoints
ARTIFACT_S3_PREFIX=nightly
ARTIFACT_S3_PATH_STYLE=1
ARTIFACT_PUSH_DEFAULT=1  # make --push the default behaviour
```

When `ARTIFACT_PUSH_DEFAULT` is `1`, running `bundle-artifacts` without flags will
upload automatically. Override per invocation with `--push/--no-push`, or provide
`--remote-name` to change the object key.

Every bundle also carries an `artifact-manifest.json` at the archive root that
records the timestamp, label, resolved sources, push preference, and key
artifact-related settings.

### Authentication & Request IDs

If you set `API_KEY` (or `API_KEYS`) in your `.env`, all write-sensitive endpoints (batch classify, review mutations, feedback submission, dataset export, `/metrics/reset`) enforce an API key header. The default header name is `x-api-key`.

```powershell
$headers = @{ 'x-api-key' = $env:API_KEY }
Invoke-RestMethod -Uri http://127.0.0.1:8000/review/enqueue -Headers $headers -Method Post -Body (@{
  text = 'low confidence sample'
  predicted_label = 'TECH'
  confidence_score = 0.42
  confidence_margin = 0.07
} | ConvertTo-Json) -ContentType 'application/json'
```

Every response also returns an `x-request-id` header so logs and client traces can be correlated.

### Example Request (Classification)

```powershell
$body = @{ title = "New Satellite Launch"; text = "A private aerospace firm launched a new satellite..." } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/classify_news -Method POST -Body $body -ContentType 'application/json'
```

### Response (Sample)

```json
{
  "top_category": "3",
  "categories": [
    { "name": "3", "prob": 0.85 },
    { "name": "1", "prob": 0.08 },
    { "name": "2", "prob": 0.05 }
  ],
  "model_version": "clf_v1",
  "latency_ms": 12.4
}
```

(AG News labels: 0 = World, 1 = Sports, 2 = Business, 3 = Sci/Tech)

## Project Layout

```
app/
  api/routes/*.py
  services/
  core/config.py
scripts/train_baseline.py
models/classifier/*.pkl
```

## Next Steps

1. Promote dataset/model artifacts to remote storage (S3/Azure blob) and wire
   configurable paths.
2. Package model training + evaluation into reproducible CLI workflows.
3. Add health/metrics dashboards (Grafana or Lightdash) atop the existing metrics.
4. Introduce optional MLflow tracking once deployment workflow is stable.

## Environment Variables

See `.env.example`. Notable flags:

- `API_KEY` / `API_KEYS` — comma-separated list of keys required for protected routes.
- `API_KEY_HEADER` — override the header name (default `x-api-key`).
- `REQUEST_ID_HEADER` — change the request identifier header (default `x-request-id`).
- `LOG_JSON` — set to `0` to emit human-readable request logs instead of JSON.
- Summarizer controls: `SUMMARIZER_MAX_NEW_TOKENS`, `SUMMARIZER_MIN_NEW_TOKENS`, `SUMMARIZER_NUM_BEAMS`, `SUMMARIZER_TRUNCATE_TOKENS`.
- Artifact publishing: `ARTIFACT_STORE_TYPE`, `ARTIFACT_S3_BUCKET`,
  `ARTIFACT_S3_REGION`, `ARTIFACT_S3_ENDPOINT`, `ARTIFACT_S3_PREFIX`,
  `ARTIFACT_S3_PATH_STYLE`, `ARTIFACT_PUSH_DEFAULT`.

## Notes

- Lint warnings (line length) intentionally ignored for speed; can add `ruff`/`flake8` later.
- Summarization now uses DistilBART (CPU). Tune via env vars: `SUMMARIZER_MAX_NEW_TOKENS`, `SUMMARIZER_MIN_NEW_TOKENS`, `SUMMARIZER_NUM_BEAMS`, `SUMMARIZER_TRUNCATE_TOKENS`.

## License / Dataset Attribution

Educational use. Cite AG News dataset source. Will add more explicit licensing notes later.

---

Reach out for the next iteration plan (transformer fine-tune + trends) when ready.
