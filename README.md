docker compose up --build
# News Topic Intelligence Service

> Production-ready FastAPI platform for newsroom classification & summarization with an automated review loop and reproducible ML workflow.

## ⚡ Spotlight: What I Delivered

- **Hardened the public API** with API-key enforcement, structured logging, latency metrics, and configurable JSON output so analysts and ops teams can trust every response.
- **Built an active learning safety net** that funnels low-confidence predictions into a review queue and promotes human-label feedback back into the training pipeline.
- **Automated the model lifecycle** through a Typer CLI that seeds demo data, trains TF-IDF and transformer models, evaluates them, and packages artifacts for S3-compatible storage.
- **Stood up a Streamlit command center** showcasing classifier, summarizer, and live trends—perfect for stakeholder demos and recruiter-ready screenshots.
- **Locked down the supply chain** by upgrading vulnerable dependencies and wiring CI to run Ruff, pytest, Bandit, and pip-audit on every push.

## 🔑 Design Decisions & Impact

### Active Learning Review Loop
Low-confidence predictions from `/classify_news` (and the batch endpoint) are automatically queued in the database. Reviewers can label them, and the CLI can merge that feedback into future fine-tuning runs—one tight feedback loop instead of ad-hoc spreadsheets.

### Transparent Operations From Day One
Every request receives an `x-request-id`, metrics are exposed at `/metrics`, and logs are JSON-formatted by default. Turning on `LOG_JSON=0` flips back to console-friendly logs when you just need to prototype.

### Reproducible Training Runs
`scripts/manage.py` is the single entry point for seeding, baseline training, transformer fine-tuning, evaluation, and artifact bundling. Each command supports explicit seeds, dataset limits, and MLflow instrumentation so you can rerun experiments without surprise drift.

### Security & CI Discipline
Dependency upgrades removed historical CVEs, GitHub Actions enforces lint + tests + security scans, and `.env.example` documents every secret toggle (API keys, artifact pushes, MLflow). Recruiters love seeing end-to-end ownership, not just a model notebook.

## � Architecture Snapshot

```
        +---------------------------+
        |      FastAPI service      |
        |   (app/main.py routes)    |
        +----+----------------+-----+
             |                |
   Classifier service    Summarizer service
 (TF-IDF / Transformer)     (DistilBART)
             |                |
        Model artifacts    Hugging Face Hub
             |
        Typer CLI (scripts/manage.py)
     ┌──────────┴───────────┐
 bundle-artifacts     train/eval commands
             |
     Artifact store (local zip / S3)

Optional: MLflow experiment tracking (params, metrics, artifacts)
```

## 🚀 Run It Locally (Windows PowerShell)

1. Copy `.env.example` to `.env`, set `API_KEY`, and point to your datasets/models.
2. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

3. Prepare the database and demo content:

```powershell
python -m alembic -c alembic.ini upgrade head
python scripts/manage.py seed-db --overwrite
```

4. (Optional) Fine-tune or retrain models:

```powershell
# TF-IDF baseline on AG News
python scripts/manage.py train-baseline --limit 5000

# HuffPost transformer (place dataset under data/raw/huffpost first)
python scripts/manage.py train-transformer --data-path data/raw/huffpost/News_Category_Dataset_v3.json
```

5. Launch the API:

```powershell
uvicorn app.main:app --reload --port 8000
```

6. (Optional) Bring up the entire stack with Docker:

```powershell

```

## 📊 Ops & Tooling Quick Reference

- **Review queue triage**: `python scripts/manage.py active-finetune --dry-run` shows what will feed the next training pass.
- **Bundle deployable artifacts**: `python scripts/manage.py bundle-artifacts --label nightly --push` zips configs + models and pushes to S3-compatible storage when `ARTIFACT_STORE_TYPE=s3`.
- **Security & quality**: `ruff check .`, `bandit -r app scripts dashboard -ll`, `pip-audit`, and `pytest` match the CI pipeline.
- Docs for deeper dives live under `docs/DEPLOYMENT.md` and `docs/RUNBOOK.md`.

## 👀 Demo Dashboard

```powershell
streamlit run dashboard/streamlit_app.py
```

The Streamlit app mirrors real API calls, showcases confidence scores, and surfaces trend charts from `/trends`—ideal for walking a recruiter through the product without hitting the raw JSON endpoints.

## � Project Layout

```
app/
  api/routes/            # FastAPI routers (classify, summarize, review, metrics, ...)
  core/                  # Config, logging, security, metrics globals
  services/              # Classifier, summarizer, artifact store, MLflow helpers
  db/                    # SQLAlchemy models & session helpers
scripts/                 # Training, evaluation, artifact management commands
dashboard/               # Streamlit demo application
tests/                   # pytest suite (API, CLI, metrics, MLflow, ...)
docs/                    # Deployment + runbook guidance
```

## 🌟 Next Horizons

- Bias and drift analysis for the review queue.
- Scheduled retraining with GitHub Actions + artifact promotion.
- Container image hardening and SBOM generation.

---

Have questions or want a live walk-through? Open an issue or reach out—happy to demo how each decision keeps the pipeline production-ready.
- **API keys** – Set `API_KEY` or `API_KEYS` to lock down batch classification, review mutations, feedback ingestion, dataset export, and `/metrics/reset`.
