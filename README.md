docker compose up --build

# News Topic Intelligence Service

> 📽️ **Demo Video:** [Watch the end-to-end product walk-through](./DEMO_VIDEO.mkv)

> Production-ready FastAPI platform for newsroom classification & summarization with an automated review loop and reproducible ML workflow.

## ⚡ Spotlight: What I Delivered

- **Hardened the public API** with API-key enforcement, structured logging, latency metrics, and configurable JSON output so analysts and ops teams can trust every response.
- **Built an active learning safety net** that funnels low-confidence predictions into a review queue and promotes human-label feedback back into the training pipeline.
- **Implemented A/B testing infrastructure** for safe model rollouts, comparing ensemble vs transformer performance with traffic splitting, consistent user assignment, and automated winner determination.
- **Automated the model lifecycle** through a Typer CLI that seeds demo data, trains TF-IDF and transformer models, evaluates them, and packages artifacts for S3-compatible storage.
- **Stood up a Streamlit command center** showcasing classifier, summarizer, and live trends—perfect for stakeholder demos and recruiter-ready screenshots.
- **Added AI-powered image generation** with RTX 4060 GPU acceleration, generating news article visualizations in 3-5 seconds using Stable Diffusion 1.5.
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

### A/B Testing for Safe Model Rollouts

Implemented traffic splitting between ensemble and transformer models with hash-based user assignment for consistency. Tracks latency and accuracy metrics per variant, enabling data-driven model selection. Prevents deploying broken models while measuring real-world performance impact.

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
        A/B Testing Service   Model artifacts
     (Traffic splitting,       Hugging Face Hub
      metrics tracking)
             |                |
        Typer CLI (scripts/manage.py)
     ┌──────────┴───────────┐
 bundle-artifacts     train/eval commands
             |
     Artifact store (local zip / S3)

   +---------------------------+
   |    AI Image Generation    |
   |   (RTX 4060 GPU accel)    |
   +---------------------------+
           |
    Stable Diffusion 1.5
   (Hugging Face diffusers)

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
  api/routes/            # FastAPI routers (classify, summarize, review, metrics, images, ...)
  core/                  # Config, logging, security, metrics globals
  services/              # Classifier, summarizer, A/B testing, artifact store, image_generator, MLflow helpers
  db/                    # SQLAlchemy models & session helpers
scripts/                 # Training, evaluation, artifact management commands
dashboard/               # Streamlit demo application (classifier, summarizer, trends, images)
tests/                   # pytest suite (API, CLI, metrics, MLflow, image generation, ...)
docs/                    # Deployment + runbook guidance
generated_images/        # AI-generated images (gitignored)
artifacts/               # Model artifacts and bundles
models/                  # Trained model files
```

## 🌟 Next Horizons

- Bias and drift analysis for the review queue.
- Scheduled retraining with GitHub Actions + artifact promotion.
- Container image hardening and SBOM generation.

## 🗺️ **Complete Project Roadmap**

| Version  | Phase                          | Status           | Key Deliverables                                                                                                                                                                                         |
| -------- | ------------------------------ | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **v1.0** | Baseline ML                    | ✅ Completed     | • TF-IDF + Logistic Regression classifier<br>• AG News dataset (4 categories)<br>• Basic model training pipeline<br>• Initial evaluation metrics                                                         |
| **v2.0** | FastAPI Service                | ✅ Completed     | • RESTful API with `/classify_news`<br>• API key authentication<br>• Request/response logging<br>• Basic error handling<br>• Uvicorn deployment                                                          |
| **v3.0** | Ensemble + Tuning              | ✅ Completed     | • Ensemble classifier (sklearn + transformer)<br>• HuffPost dataset (42 categories)<br>• Active learning review queue<br>• Model fine-tuning pipeline<br>• Streamlit dashboard<br>• Database integration |
| **v4.0** | ML Ops Deployment & Monitoring | ✅ **COMPLETED** | • **BentoML production serving**<br>• **Evidently drift detection**<br>• **GitHub Actions CI/CD**<br>• **Automated retraining pipeline**<br>• **Kubernetes deployment**<br>• **Production monitoring**   |
| **v5.0** | A/B Testing & Model Comparison | ✅ **COMPLETED** | • **Traffic splitting infrastructure**<br>• **Hash-based user assignment**<br>• **Real-time metrics tracking**<br>• **Automated winner determination**<br>• **Production-safe model rollouts**           |
| **v6.0** | AI Image Generation & Visual Content | ✅ **COMPLETED** | • **RTX 4060 GPU acceleration**<br>• **Stable Diffusion 1.5 integration**<br>• **News article visualization**<br>• **FastAPI image endpoints**<br>• **Streamlit image generation UI** |

## 🎯 v4.0: ML Ops Deployment & Monitoring ✅ COMPLETED

**Status: ✅ Production-Ready with Automated Retraining**

### What Was Delivered

- **BentoML Model Serving**: Production-grade model deployment with ensemble/sklearn/transformer backends, automatic drift detection, and confidence scoring.
- **Evidently Drift Detection**: Real-time data drift monitoring that triggers retraining when distribution shifts exceed thresholds.
- **GitHub Actions CI/CD**: Automated pipeline for drift detection, model retraining, testing, and deployment with artifact promotion.
- **Kubernetes Deployment**: Containerized deployment with health checks, resource limits, and service discovery.

### Key Features

#### BentoML Service (`scripts/serve.py`)

```python
# Ensemble classification with drift monitoring
response = service.classify(ClassificationRequest(
    text="Apple announces new iPhone",
    backend="ensemble"
))
# Returns: category, confidence, drift_detected, drift_report
```

#### Drift Detection (`scripts/drift_detection.py`)

- Monitors category distribution shifts
- Triggers retraining when drift_score > 0.5
- Updates reference dataset automatically

#### CI/CD Pipeline (`.github/workflows/retrain.yml`)

- **Triggers**: Manual, scheduled (weekly), or drift detection
- **Jobs**: Check drift → Retrain models → Test → Deploy
- **Artifacts**: Model bundles with timestamps

#### Deployment (`scripts/deploy.sh`)

- Docker image building and Kubernetes deployment
- Health checks and resource management
- Staging → Production promotion

### Quick Start v4.0

1. **Test BentoML Service**:

```bash
python test_service_drift.py
```

2. **Run Drift Detection**:

```bash
python scripts/drift_detection.py
```

3. **Deploy to Production**:

```bash
./scripts/deploy.sh
```

4. **Monitor CI/CD**: Push to main or trigger manual workflow

### Architecture v4.0

```
Production Traffic
       ↓
   BentoML Service (Drift Detection)
       ↓
   Drift Detected? → GitHub Actions
       ↓              (Retrain Pipeline)
   Serve Response     ↓
                     Model Retraining
                     → Artifact Bundle
                     → Deploy to K8s
```

## 🎯 v5.0: A/B Testing & Model Comparison ✅ COMPLETED

**Status: ✅ Production-Ready A/B Testing Infrastructure**

### What Was Delivered

- **Traffic Splitting Service**: Hash-based user assignment ensuring consistent variant exposure
- **Real-time Metrics Tracking**: Latency and accuracy monitoring per model variant
- **Automated Winner Determination**: Statistical comparison with configurable thresholds
- **API Integration**: Seamless A/B testing endpoints with experiment management

### Key Features

#### A/B Testing Service (`app/services/ab_testing.py`)

```python
# Traffic splitting with consistent user assignment
ab_service = get_ab_testing_service()
variant = ab_service.assign_variant("ensemble_vs_transformer", user_id)
result = classify_text(text, backend=variant)
ab_service.record_result(experiment_name, variant, latency, confidence)
```

#### FastAPI A/B Endpoints (`app/api/routes/ab_test.py`)

- **POST `/ab_test`**: Automatic variant assignment and classification
- **GET `/ab_test/results/{experiment}`**: Real-time experiment metrics
- **POST `/ab_test/complete/{experiment}`**: Winner determination and experiment completion

#### Experiment Management

- **Traffic Split**: Configurable 50/50 or custom ratios
- **Consistent Assignment**: Hash-based user bucketing prevents result contamination
- **Metrics Tracking**: Request counts, latency averages, accuracy monitoring
- **Winner Logic**: Better accuracy wins, latency as tiebreaker

### Quick Start v5.0

1. **Test A/B Classification**:

```bash
python test_ab_testing.py
```

2. **Start FastAPI Server**:

```bash
uvicorn app.main:app --reload --port 8000
```

3. **Test A/B Endpoint**:

```bash
curl -X POST "http://localhost:8000/ab_test" \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple announces new iPhone", "user_id": "user123"}'
```

4. **View Experiment Results**:

```bash
curl "http://localhost:8000/ab_test/results/ensemble_vs_transformer" \
  -H "X-API-Key: your-key"
```

### Architecture v5.0

```
User Request
     ↓
Traffic Splitter (Hash-based)
     ↓
   Control Group ──── Ensemble Model
     ↓                   ↓
   Treatment Group ── Transformer Model
     ↓                   ↓
  Metrics Collection   Metrics Collection
     ↓                   ↓
  Experiment Results   Experiment Results
     ↓                   ↓
   Winner Determination
```

## 🎯 v6.0: AI Image Generation & Visual Content ✅ COMPLETED

**Status: ✅ GPU-Accelerated Image Generation for News Articles**

### What Was Delivered

- **RTX 4060 GPU Acceleration**: High-performance image generation with 3-5 second response times for 512x512 images
- **Stable Diffusion 1.5 Integration**: Professional-grade image generation using Hugging Face diffusers
- **News-Specific Image Generation**: Context-aware prompts that create relevant visualizations for news articles
- **Streamlit Dashboard Integration**: Interactive image generation interface with multiple generation modes
- **FastAPI Endpoints**: RESTful API for programmatic image generation and service health monitoring

### Key Features

#### Image Generation Service (`app/services/image_generator.py`)

```python
# GPU-accelerated news image generation
from app.services.image_generator import NewsImageGenerator

generator = NewsImageGenerator()
image_path = generator.generate_news_image(
    title="Apple Announces New iPhone",
    category="Technology",
    summary="Apple unveiled the latest iPhone with revolutionary features..."
)
```

#### FastAPI Image Endpoints (`app/api/routes/images.py`)

- **GET `/images/status`**: Service health check and GPU availability
- **POST `/images/generate-image`**: Custom prompt-based image generation
- **POST `/images/generate-news-image`**: News article visualization with automatic prompt engineering

#### Streamlit Images Tab (`dashboard/streamlit_app.py`)

- **Custom Generation**: Free-form prompt input with real-time generation
- **News Article Mode**: Automatic prompt generation from article content
- **Category-Based Mode**: Pre-configured prompts for different news categories
- **Progress Tracking**: Real-time generation status with estimated completion times

### Quick Start v6.0

1. **Check GPU Availability**:

```bash
curl "http://localhost:8000/images/status" \
  -H "X-API-Key: your-key"
```

2. **Generate News Article Image**:

```bash
curl -X POST "http://localhost:8000/images/generate-news-image" \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Apple Announces New iPhone",
    "category": "Technology",
    "summary": "Apple unveiled the latest iPhone with revolutionary features including advanced AI capabilities and improved camera system."
  }'
```

3. **Launch Streamlit Dashboard**:

```bash
streamlit run dashboard/streamlit_app.py
```

4. **Access Images Tab**: Navigate to the "Images" tab for interactive generation

### Architecture v6.0

```
News Article / Custom Prompt
              ↓
     Prompt Engineering
     (Category + Content Analysis)
              ↓
   RTX 4060 GPU Acceleration
   (Stable Diffusion 1.5)
              ↓
   Image Post-Processing
   (Resize, Format, Save)
              ↓
   File Path Response
   (generated_images/*.png)
```

### Performance Metrics

- **Generation Time**: 3-5 seconds per 512x512 image
- **GPU Memory**: ~4GB VRAM usage during generation
- **Image Quality**: Professional-grade outputs suitable for news media
- **Concurrent Requests**: Single GPU instance (can be scaled with multiple GPUs)

## 🎯 **PFE Internship Preparation Guide**

### Your Project Demonstrates These Key Engineering Skills:

**1. Full-Stack Development**

- Modern FastAPI backend with async processing
- RESTful API design with proper error handling
- Database integration with SQLAlchemy and Alembic migrations

**2. Machine Learning Engineering**

- End-to-end ML pipeline from data to production
- Model training, evaluation, and deployment
- A/B testing for model comparison and safe rollouts
- AI image generation with GPU acceleration and Stable Diffusion

**3. DevOps & Automation**

- CI/CD pipeline with automated testing and deployment
- Containerization with Docker and docker-compose
- Automated model retraining and artifact management

**4. Production-Ready Architecture**

- Service-oriented architecture with dependency injection
- Comprehensive testing (unit, integration, end-to-end)
- Monitoring, logging, and security best practices
- GPU-accelerated AI services integration

### PFE Interview Talking Points:

**"Décrivez votre projet de PFE" (Describe your PFE project)**

> "J'ai développé un service complet de classification d'articles de presse utilisant l'intelligence artificielle. Le système comprend un pipeline ML automatisé, des tests A/B pour la comparaison de modèles, et un déploiement en production avec monitoring continu."

**"Quelles technologies avez-vous utilisées?" (What technologies did you use?)**

> "J'ai utilisé FastAPI pour l'API REST, scikit-learn et transformers pour les modèles ML, PostgreSQL pour la base de données, et Docker pour la containerisation. Le projet inclut également des tests automatisés et un pipeline CI/CD."

**"Comment avez-vous géré la complexité?" (How did you handle complexity?)**

> "J'ai structuré le projet en couches (services, API, base de données) avec une architecture modulaire. J'ai implémenté des tests A/B pour valider les performances des modèles en production, et ajouté un système de monitoring pour suivre les métriques en temps réel."

**"Quels défis avez-vous rencontrés?" (What challenges did you face?)**

> "Le défi principal était d'intégrer les tests A/B avec le système de classification existant. J'ai dû modifier le service de classification pour supporter plusieurs backends de modèles tout en maintenant la cohérence des assignations utilisateurs via un système de hachage."

### Technical Skills Demonstrated:

- **Python & FastAPI**: Modern web development with async/await
- **Machine Learning**: Model training, evaluation, A/B testing, AI image generation
- **GPU Computing**: RTX 4060 acceleration with CUDA and Hugging Face diffusers
- **Database Design**: SQLAlchemy ORM, migrations, data modeling
- **DevOps**: Docker, CI/CD, automated testing, monitoring
- **Software Architecture**: Service layer, dependency injection, clean code
- **Testing**: Unit tests, integration tests, end-to-end testing
- **AI/ML Integration**: Stable Diffusion, prompt engineering, computer vision

### Your Competitive Advantages for Tunisian PFE:

✅ **Complete Project**: From concept to production deployment
✅ **Modern Technologies**: FastAPI, transformers, Docker, GPU acceleration (industry standards)
✅ **Production Mindset**: Monitoring, testing, security, scalability
✅ **AI Innovation**: Cutting-edge AI image generation with Stable Diffusion
✅ **Documentation**: Comprehensive README, API docs, architecture diagrams
✅ **Real-World Application**: News classification with business impact and visual content generation

### PFE Evaluation Criteria Alignment:

**Technical Excellence (40%)**: Advanced ML implementation, clean architecture, GPU computing
**Innovation (20%)**: A/B testing, automated pipelines, modern tech stack, AI image generation
**Documentation (15%)**: Detailed README, code comments, architecture docs
**Presentation (15%)**: Clear explanations, demo capabilities
**Autonomy (10%)**: Independent project execution from start to finish

**Remember**: Tunisian PFE evaluators look for practical engineering skills, project completeness, and the ability to explain technical decisions. Your project demonstrates all of these!

**Pro Tip**: Prepare a 10-minute demo showing the API working, A/B testing in action, and the automated pipeline. Focus on explaining _why_ you made each architectural decision! 🚀
