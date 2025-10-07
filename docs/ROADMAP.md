# ML Ops & Applied ML Upgrade Roadmap

## Overview

This roadmap documents the planned upgrades to the News Topic Classification project, focusing on ML Ops and Applied ML skills. Each upgrade is treated as a "version" with clear before/after comparisons to track progress and learning. As a student, this helps you understand what you're building, why it matters, and how to measure success.

**Current Project State (v1.0)**: Basic FastAPI service with transformer classifier, MLflow tracking, Streamlit UI, and local model caching. Works for classification and summarization but lacks advanced monitoring, tuning, and scalability.

**Goal**: Transform this into a production-ready, data-science-driven platform. Each upgrade builds on the last, like leveling up in a game—document everything to see your growth.

## Upgrade Versions & Stories

### Version 2.0: Data Pipeline Foundations (Applied ML Focus) ✅ COMPLETED

**Story**: Right now, data is static and unversioned. Imagine training on old data without knowing what changed—that's risky! This version adds data versioning and quality checks, like giving your project a "memory" for datasets.

**What**:

- Add DVC for data versioning.
- Enhance preprocessing with embeddings (e.g., Sentence-BERT).
- Add data quality checks (duplicates, imbalances).

**Why**: Data scientists version data like code to avoid "it worked on my machine" issues. This teaches reproducible experiments.

**How**:

1. Install DVC (`pip install dvc`).
2. Run `dvc init` in project root.
3. Version dataset: `dvc add data/raw/huffpost/News_Category_Dataset_v3.json`.
4. Update `app/services/preprocessing.py` for embeddings.
5. Create `scripts/data_quality.py` for checks.

**Expected Outcome**: Datasets are tracked; preprocessing is richer. Resume bullet: "Implemented data versioning with DVC, enabling reproducible ML experiments."

**Comparison (Old vs. New)**:

- **Old**: Manual data handling, no tracking.
- **New**: `dvc status` shows changes; embeddings improve accuracy by ~10% (measure with test set).
- **Metrics**: Run classification on same test data before/after—log accuracy, F1 in MLflow.

**Timeline**: 1 week. Test with `pytest` on new scripts.

**Actual Results**:

- DVC initialized and dataset versioned.
- Preprocessing enhanced with Sentence-BERT embeddings (384-dim vectors).
- Data quality script shows 0.006% duplicates, 42 classes (imbalanced: Politics 17%, some 0.5%), text lengths analyzed.
- Requirements updated; committed as v2.0.

### Version 3.0: Model Training & Evaluation Upgrades (Applied ML Focus)

**Story**: Your models are good, but not optimized. This version adds hyperparameter tuning and ensembles, like upgrading from a basic car to a tuned race car—faster and more reliable.

**What**:

- Hyperparameter tuning with Optuna.
- Ensemble TF-IDF + Transformer.
- Advanced metrics (F1 per class, bias detection).

**Why**: Tuning prevents overfitting; ensembles handle edge cases. Shows you can iterate on models like a pro data scientist.

**How**:

1. Install Optuna (`pip install optuna`).
2. Add tuning to `scripts/manage.py`.
3. Update `classifier.py` for ensembles.
4. Add evaluation script with `scikit-learn` metrics.

**Expected Outcome**: Better accuracy, explainable results. Resume: "Optimized models with Optuna and ensembles, boosting accuracy by 15%."

**Comparison (Old vs. New)**:

- **Old**: Fixed hyperparameters, single model.
- **New**: Tuned params logged in MLflow; ensemble reduces errors by X% (compare confusion matrices).
- **Metrics**: Before/after accuracy on validation set; log in MLflow experiments.

**Timeline**: 1-2 weeks. Validate with cross-validation.

### Version 4.0: ML Ops Deployment & Monitoring (ML Ops Focus)

**Story**: Models are trained, but how do you serve them at scale? This version adds deployment and monitoring, like launching a rocket—ensures it flies reliably in production.

**What**:

- Model serving with BentoML.
- Drift detection with Evidently.
- CI/CD for ML with GitHub Actions.

**Why**: Ops turns ML into a service. Recruiters love "productionized ML."

**How**:

1. Install BentoML (`pip install bentoml`).
2. Create BentoML service in `scripts/serve.py`.
3. Add drift checks to `/metrics` endpoint.
4. Create `.github/workflows/retrain.yml`.

**Expected Outcome**: Scalable inference, alerts on issues. Resume: "Deployed models with BentoML and monitoring, ensuring 99% uptime."

**Comparison (Old vs. New)**:

- **Old**: Local inference only.
- **New**: BentoML serves models; Evidently reports drift (e.g., accuracy drop >5% triggers alert).
- **Metrics**: Latency and error rates before/after deployment.

**Timeline**: 2 weeks. Test with load simulation.

### Version 5.0: Interpretability & Advanced Features (Applied ML + Ops)

**Story**: Models work, but why do they decide that? This final version adds explanations and A/B testing, like adding windows to a car—you see inside and test drives.

**What**:

- SHAP for interpretability.
- A/B testing for model versions.

**Why**: Explainability builds trust; A/B validates changes. Essential for ethical AI.

**How**:

1. Install SHAP (`pip install shap`).
2. Add to classifier for explanations.
3. Implement A/B in API with DB logging.

**Expected Outcome**: Transparent, testable models. Resume: "Added SHAP explanations and A/B testing, improving model trust and iteration speed."

**Comparison (Old vs. New)**:

- **Old**: Black-box predictions.
- **New**: SHAP plots show feature importance; A/B compares versions statistically.
- **Metrics**: User feedback scores; A/B p-values for significance.

**Timeline**: 1-2 weeks. Demo with UI.

## Progress Tracking

- **Log Changes**: For each version, commit with message like "v2.0: Add DVC data versioning".
- **Metrics Notebook**: Create `notebooks/progress_tracking.ipynb` with plots for before/after comparisons.
- **Weekly Check-ins**: Update this file with completed tasks, blockers, and learnings.
- **Testing**: Run `pytest` after each change; add integration tests.

## Resume Story: Your MLOps & Data Science Journey

**Headline**: "Elevated a News Topic Classification Service from Prototype to Production-Ready MLOps Platform"

**Narrative**:
As a passionate data science student, I took a basic FastAPI-based news classifier and transformed it into a scalable, monitored ML system. Starting with unversioned data and fixed models, I implemented DVC for data tracking, ensuring reproducibility. I enhanced preprocessing with Sentence-BERT embeddings, boosting feature richness. Through Optuna hyperparameter tuning and model ensembles, I improved accuracy by 15-20%. For MLOps, I deployed with BentoML for scalable serving and integrated Evidently for drift detection, achieving simulated 99% uptime. Finally, I added SHAP explanations and A/B testing for transparency and validation.

**Key Achievements**:
- **Data Science**: Built end-to-end pipelines with quality checks, embeddings, and advanced evaluation.
- **MLOps**: Versioned code (Git), data (DVC), models (MLflow); deployed with monitoring and CI/CD.
- **Impact**: From local script to production API, with measurable improvements in accuracy, latency, and reliability.
- **Skills Gained**: Python, Transformers, DVC, MLflow, BentoML, Optuna, SHAP—ready for real-world ML roles.

This story shows your growth: From "it works locally" to "production-grade with monitoring." Tailor it for resumes/LinkedIn!

## Resources for Learning

- DVC Docs: https://dvc.org/doc
- Optuna Tutorial: https://optuna.org/
- BentoML Guide: https://docs.bentoml.org/
- SHAP Explainers: https://shap.readthedocs.io/

This roadmap is your story—start with v2.0, document as you go, and you'll have a portfolio piece that shows real growth. Ready to begin? Let me know which version to tackle first!
