docker compose up --build

# News Topic Intelligence Service

> 📽️ **Demo Video:** [Watch the end-to-end product walk-through](./DEMO_VIDEO.mkv)

> Production-ready FastAPI platform for **multimodal news classification** & summarization with an automated review loop and reproducible ML workflow.

## ⚡ Spotlight: What I Delivered

- **Built an intelligent RAG chatbot** with multi-source retrieval across 200K+ news articles, platform documentation, and real-time analytics, featuring intent classification, conversation memory, and source citations for comprehensive news intelligence.
- **Hardened the public API** with API-key enforcement, structured logging, latency metrics, and configurable JSON output so analysts and ops teams can trust every response.
- **Built an active learning safety net** that funnels low-confidence predictions into a review queue and promotes human-label feedback back into the training pipeline.
- **Implemented Stream Review system** for real-time human-in-the-loop labeling of streaming data, automatically detecting low-confidence predictions and anomalies during live news processing with dedicated review queues and dashboard integration.
- **Implemented A/B testing infrastructure** for safe model rollouts, comparing ensemble vs transformer performance with traffic splitting, consistent user assignment, and automated winner determination.
- **Automated the model lifecycle** through a Typer CLI that seeds demo data, trains TF-IDF and transformer models, evaluates them, and packages artifacts for S3-compatible storage.
- **Stood up a Streamlit command center** showcasing classifier, summarizer, live trends, and chatbot interface—perfect for stakeholder demos and recruiter-ready screenshots.
- **Implemented multimodal news classification** using CLIP + BLIP vision models, combining text and image analysis for enhanced accuracy with intelligent confidence scoring.
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
        +----+Chatbot Service |
        |    | (RAG + Intent   |
        |    |  Classification)|
        |    +----------------+
        |            |
        |    Multi-Source RAG
        |    (News + Docs + Analytics)
        |
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
# TF-IDF baseline on HuffPost (default dataset with dates)
python scripts/manage.py train-baseline --limit 5000

# HuffPost transformer (dataset already available locally)
python scripts/manage.py train-transformer --data-path data/raw/huffpost/News_Category_Dataset_v3.json
```

5. Launch the API:

```powershell
uvicorn app.main:app --reload --port 8001
```

6. (Optional) Bring up the entire stack with Docker:

```powershell

```

## 📊 Ops & Tooling Quick Reference

- **Review queue triage**: `python scripts/manage.py active-finetune --dry-run` shows what will feed the next training pass.
- **Bundle deployable artifacts**: `python scripts/manage.py bundle-artifacts --label nightly --push` zips configs + models and pushes to S3-compatible storage when `ARTIFACT_STORE_TYPE=s3`.
- **Security & quality**: `ruff check .`, `bandit -r app scripts dashboard -ll`, `pip-audit`, and `pytest` match the CI pipeline.
- Docs for deeper dives live under `docs/DEPLOYMENT.md`, `docs/RUNBOOK.md`, and `docs/STREAM_REVIEW.md`.

## 👀 Demo Dashboard

```powershell
streamlit run dashboard/streamlit_app.py
```

The Streamlit app mirrors real API calls, showcases confidence scores, and surfaces trend charts from `/trends`—ideal for walking a recruiter through the product without hitting the raw JSON endpoints.

## 🤖 Intelligent RAG Chatbot

The platform includes a sophisticated **Retrieval-Augmented Generation (RAG) chatbot** that provides intelligent responses about news articles, platform documentation, and real-time analytics.

### Features

- **Multi-Source Intelligence**: Searches across 200K+ news articles, platform documentation, and live analytics
- **Intent Classification**: Automatically routes queries to appropriate data sources (news, platform help, analytics)
- **Source Citations**: Every response includes verifiable sources with article metadata
- **Conversation Memory**: Maintains context across multi-turn conversations
- **Production-Ready**: REST API endpoints with comprehensive error handling and metrics

### Usage Examples

```bash
# Start the chatbot service
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

# Query examples:
# "Show me recent articles about climate change"
# "How do I use the streaming API?"
# "What are the latest trends in politics?"
# "Explain how the classifier works"
```

### API Endpoints

- `POST /chatbot/chat` - Send a message and get intelligent response
- `GET /chatbot/health` - Service health check
- `GET /chatbot/stats` - Usage statistics and metrics

### Architecture

The chatbot uses a **hybrid RAG approach**:
- **News Articles**: FAISS vector search over 200K+ articles with metadata filtering
- **Documentation**: Semantic search across platform docs with section-aware retrieval
- **Analytics**: Real-time data integration for trend analysis and insights
- **Intent Router**: Rule-based classification directing queries to optimal sources

## � Project Layout

```
app/
  api/routes/            # FastAPI routers (classify, summarize, review, metrics, images, chatbot)
  core/                  # Config, logging, security, metrics globals
  services/              # Classifier, summarizer, A/B testing, artifact store, image_generator, chatbot
    chatbot/             # RAG chatbot with intent classification and multi-source retrieval
  db/                    # SQLAlchemy models & session helpers
scripts/                 # Training, evaluation, artifact management, data ingestion commands
dashboard/               # Streamlit demo application (classifier, summarizer, trends, chatbot)
tests/                   # pytest suite (API, CLI, metrics, MLflow, image generation, chatbot)
docs/                    # Deployment + runbook guidance, chatbot roadmap
data/
  vectorstores/          # FAISS indexes for news articles and documentation
  processed/             # Cleaned datasets ready for ML
  raw/                   # Original datasets (HuffPost, images, etc.)
generated_images/        # AI-generated images (gitignored)
artifacts/               # Model artifacts and bundles
models/                  # Trained model files
```

## 🌟 Next Horizons

- Bias and drift analysis for the review queue.
- Scheduled retraining with GitHub Actions + artifact promotion.
- Container image hardening and SBOM generation.

## 🗺️ **Complete Project Roadmap**

| Version  | Phase                                | Status           | Key Deliverables                                                                                                                                                                                         |
| -------- | ------------------------------------ | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **v1.0** | Baseline ML                          | ✅ Completed     | • TF-IDF + Logistic Regression classifier<br>• HuffPost dataset (42 categories with dates)<br>• Basic model training pipeline<br>• Initial evaluation metrics                                            |
| **v2.0** | FastAPI Service                      | ✅ Completed     | • RESTful API with `/classify_news`<br>• API key authentication<br>• Request/response logging<br>• Basic error handling<br>• Uvicorn deployment                                                          |
| **v3.0** | Ensemble + Tuning                    | ✅ Completed     | • Ensemble classifier (sklearn + transformer)<br>• HuffPost dataset (42 categories)<br>• Active learning review queue<br>• Model fine-tuning pipeline<br>• Streamlit dashboard<br>• Database integration |
| **v4.0** | ML Ops Deployment & Monitoring       | ✅ **COMPLETED** | • **BentoML production serving**<br>• **Evidently drift detection**<br>• **GitHub Actions CI/CD**<br>• **Automated retraining pipeline**<br>• **Kubernetes deployment**<br>• **Production monitoring**   |
| **v5.0** | A/B Testing & Model Comparison       | ✅ **COMPLETED** | • **Traffic splitting infrastructure**<br>• **Hash-based user assignment**<br>• **Real-time metrics tracking**<br>• **Automated winner determination**<br>• **Production-safe model rollouts**           |
| **v6.0** | AI Image Generation & Visual Content | ✅ **COMPLETED** | • **RTX 4060 GPU acceleration**<br>• **Stable Diffusion 1.5 integration**<br>• **News article visualization**<br>• **FastAPI image endpoints**<br>• **Streamlit image generation UI**                    |
| **v7.0** | Multimodal News Classification       | ✅ **COMPLETED** | • **CLIP + BLIP vision models**<br>• **Intelligent image relevance analysis**<br>• **Text-image fusion architecture**<br>• **Context-aware confidence scoring**<br>• **Production multimodal API**       |
| **v8.0** | Time Series Forecasting              | ✅ **COMPLETED** | • **Hybrid ML/DL forecasting**<br>• **Prophet + XGBoost + LSTM models**<br>• **News category trend prediction**<br>• **MLflow experiment tracking**<br>• **RESTful forecasting API**                     |
| **v9.0** | Intelligent RAG Chatbot              | ✅ **COMPLETED** | • **Multi-source RAG across 200K+ articles**<br>• **Intent classification & routing**<br>• **Conversation memory & citations**<br>• **REST API integration**<br>• **Streamlit chatbot UI**              |

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
uvicorn app.main:app --reload --port 8001
```

3. **Test A/B Endpoint**:

```bash
curl -X POST "http://localhost:8001/ab_test" \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple announces new iPhone", "user_id": "user123"}'
```

4. **View Experiment Results**:

```bash
curl "http://localhost:8001/ab_test/results/ensemble_vs_transformer" \
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
curl "http://localhost:8001/images/status" \
  -H "X-API-Key: your-key"
```

2. **Generate News Article Image**:

```bash
curl -X POST "http://localhost:8001/images/generate-news-image" \
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

## 🎯 v7.0: Multimodal News Classification ✅ COMPLETED

**Status: ✅ Intelligent Text + Image Classification with Vision-Language Models**

### What Was Delivered

- **CLIP + BLIP Vision Models**: Dual vision architecture combining OpenAI CLIP for embeddings and Salesforce BLIP for intelligent image captioning
- **Smart Image Relevance Analysis**: BLIP-powered content understanding that analyzes actual image content rather than just URLs
- **Text-Image Fusion Architecture**: Neural network that learns to combine text and image features for improved classification accuracy
- **Context-Aware Confidence Scoring**: Dynamic confidence scores based on image content analysis (charts: 0.9, financial content: 0.8, business images: 0.7)
- **Production Multimodal API**: Enhanced `/classify_news` endpoint supporting both text-only and text+image classification

### Key Features

#### Multimodal Classifier Service (`app/services/multimodal_classifier.py`)

```python
# Intelligent multimodal classification
from app.services.multimodal_classifier import classify_multimodal_news

result = classify_multimodal_news(
    title="Stock Market Update",
    text="The stock market showed significant gains today...",
    image_url="https://example.com/chart.png"
)

# Returns enhanced classification with:
# - modalities: ["text", "image"]
# - fusion_used: true
# - image_confidence: 0.9 (for charts)
```

#### Vision Models Integration

- **CLIP (OpenAI)**: Provides image embeddings for fusion with text features
- **BLIP (Salesforce)**: Generates natural language captions for intelligent content analysis
- **Fusion Model**: PyTorch neural network combining 768D text + 512D image embeddings

#### Smart Confidence Scoring

```python
# BLIP analyzes image content and assigns confidence:
# 📊 Charts/graphs → 0.9 confidence
# 💰 Financial/stock images → 0.8 confidence
# 🏢 Business content → 0.7 confidence
# 📝 Text documents → 0.6 confidence
# 👤 People/photos → 0.2 confidence (low relevance)
```

#### FastAPI Multimodal Endpoints (`app/api/routes/classify.py`)

- **POST `/classify_news`**: Enhanced endpoint supporting `image_url` and `image_base64` parameters
- **Automatic Fallback**: Gracefully falls back to text-only classification if image processing fails
- **Fusion Indicators**: Returns `modalities` and `fusion_used` flags for transparency

### Quick Start v7.0

1. **Test Multimodal Classification**:

```bash
# Text + image classification
curl -X POST "http://localhost:8001/classify_news" \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Stock Market Analysis",
    "text": "This chart shows the performance of major indices...",
    "image_url": "https://example.com/stock-chart.png"
  }'

# Response includes multimodal metadata:
# {
#   "top_category": "BUSINESS",
#   "modalities": ["text", "image"],
#   "fusion_used": true,
#   "image_confidence": 0.9
# }
```

2. **Text-Only Classification** (automatic fallback):

```bash
curl -X POST "http://localhost:8001/classify_news" \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple announces new iPhone"}'
```

### Architecture v7.0

```
News Article + Image URL
              ↓
     Text Classification (RoBERTa)
              ↓
   Image Processing Pipeline
   ├── CLIP: Image Embeddings (512D)
   └── BLIP: Content Captioning
              ↓
     Multimodal Fusion
     (Text 768D + Image 512D → 1280D)
              ↓
   Classification Head (1280D → 16 categories)
              ↓
   Enhanced Response
   (modalities, fusion_used, image_confidence)
```

### Performance Metrics

- **Classification Accuracy**: Improved accuracy for news with visual content
- **Image Processing**: ~2-3 seconds per image (CLIP + BLIP)
- **Fusion Latency**: Minimal overhead (< 100ms) for trained model
- **Confidence Intelligence**: Context-aware scoring based on actual image content
- **Fallback Robustness**: Automatic text-only mode when images unavailable

## 🎯 v8.0: Time Series Forecasting ✅ COMPLETED

**Status: ✅ Hybrid ML/DL Forecasting with MLOps Tracking**

### What Was Delivered

- **Hybrid Forecasting Models**: Ensemble of Prophet (statistical), XGBoost (ML), and LSTM (DL) for robust predictions
- **News Category Trends**: Forecast article volume trends for 42 HuffPost categories using historical data
- **MLflow Experiment Tracking**: Full MLOps pipeline with metrics, artifacts, and model versioning
- **RESTful Forecasting API**: `/trends/forecast/{category}` endpoint with configurable prediction horizons

### Key Features

#### Forecasting Service (`app/services/time_series_forecaster.py`)

- **Prophet Models**: Seasonal decomposition for news trends with holiday awareness
- **XGBoost Models**: Feature engineering with lag variables, rolling statistics, and date features
- **LSTM Networks**: Deep learning for complex sequential patterns in PyTorch
- **Ensemble Predictions**: Weighted combination of all three models for robust forecasts

#### API Endpoints (`app/api/routes/trends.py`)

```python
# Forecast POLITICS category trends for next 7 days
GET /trends/forecast/POLITICS?days_ahead=7

# Response includes forecast values, confidence intervals, and model metadata
{
  "category": "POLITICS",
  "dates": ["2025-10-12", "2025-10-13", ...],
  "forecast": [145.2, 152.8, ...],
  "confidence_lower": [130.7, 137.5, ...],
  "confidence_upper": [159.7, 168.1, ...],
  "model_info": {
    "prophet_weight": 0.7,
    "xgb_weight": 0.3,
    "method": "ensemble"
  }
}
```

#### Training Pipeline (`scripts/train_forecasting.py`)

```powershell
# Train forecasting models for top 10 categories
python scripts/manage.py train-forecasting --max-categories 10

# Models saved to models/forecasting/ with MLflow tracking
```

### Quick Start v8.0

```powershell
# 1. Train forecasting models (one-time setup)
python scripts/manage.py train-forecasting

# 2. Start API server
uvicorn app.main:app --reload --port 8001

# 3. Forecast category trends
curl "http://localhost:8001/trends/forecast/POLITICS?days_ahead=7"
```

### Architecture v8.0

```
HuffPost Dataset (209K articles, 2012-2022)
              ↓
     Time Series Aggregation
     ├── Daily article counts per category
     └── Date range filling (missing days = 0)
              ↓
   Model Training Pipeline
   ├── Prophet: Seasonal decomposition
   ├── XGBoost: Feature engineering
   └── LSTM: Sequential learning
              ↓
    Ensemble Forecasting
    ├── Weighted predictions
    └── Confidence intervals
              ↓
   REST API + MLflow Tracking
```

## 📄 Resume-ready project summary

Use one of these variants directly in your resume.

### Experience entry (recommended)

News Topic Intelligence Platform — ML Engineer (2024–2025)

- Built a production-ready FastAPI platform for news classification and summarization with 11+ REST endpoints, SQLAlchemy/Alembic, and structured logging/metrics.
- Implemented multimodal news classification using CLIP + BLIP vision models, combining text and image analysis for enhanced accuracy with intelligent confidence scoring.
- Added GPU-accelerated AI image generation using Stable Diffusion 1.5 (RTX 4060 + diffusers), producing 512×512 visuals in ~3–5s via FastAPI and Streamlit UI.
- Delivered A/B testing infrastructure with hash-based user assignment and real-time metrics to compare ensemble vs transformer models for safe rollouts.
- Added active learning loop: routed low-confidence predictions into a review queue and merged human feedback into retraining pipelines.
- Implemented Stream Review system for real-time human-in-the-loop labeling of streaming data, automatically detecting low-confidence predictions and anomalies during live news processing with dedicated review queues and dashboard integration.
- Implemented hybrid time series forecasting with Prophet/XGBoost/LSTM ensemble models for predicting news category trends using 10+ years of historical data.
- Automated ML lifecycle with Typer CLI (train/eval/bundle), GitHub Actions CI (ruff/pytest/bandit), and artifact publishing to S3-compatible storage.
- Shipped an interactive Streamlit dashboard (classification, summarization, trends, forecasting, images) for stakeholder demos and analysis.

Tech: Python 3.11, FastAPI, SQLAlchemy, Pydantic, PyTorch, transformers, diffusers, scikit-learn, CUDA (RTX 4060), Hugging Face, CLIP, BLIP, Streamlit, Alembic, PostgreSQL, MLflow, BentoML, Evidently, Docker, GitHub Actions, DVC, pytest, ruff, bandit, Prophet, XGBoost.

### Project section (concise)

- Full-stack AI platform for news classification/summarization with multimodal text+image analysis, A/B testing, active learning, Stream Review system for real-time human-in-the-loop labeling, GPU image generation (Stable Diffusion 1.5 on RTX 4060), and hybrid time series forecasting (Prophet/XGBoost/LSTM).
- Production API (FastAPI) with auth, JSON logging, latency metrics; Streamlit dashboard for live demos; automated training/eval/artifact bundling.
- CI/CD with GitHub Actions, security scanning (bandit), linting (ruff), tests (pytest); ML monitoring and drift detection (Evidently, MLflow/BentoML).

### One-liner

Production-grade AI platform for news intelligence with multimodal text+image classification, transformer-based analysis, Stream Review system for real-time human-in-the-loop labeling, GPU-accelerated image generation, hybrid time series forecasting (Prophet/XGBoost/LSTM), A/B testing, and automated MLOps (FastAPI + CUDA + diffusers + CLIP/BLIP + GitHub Actions).
