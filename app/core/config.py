from pydantic import BaseModel, Field
from functools import lru_cache
import os
import re
from typing import Dict
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseModel):
    model_config = {"protected_namespaces": ()}

    app_name: str = os.getenv("APP_NAME", "news-topic-intel")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    model_dir: str = os.getenv("MODEL_DIR", "models")
    classifier_backend: str = os.getenv("CLASSIFIER_BACKEND", "sklearn")
    classifier_version: str = os.getenv("CLASSIFIER_VERSION", "clf_v1")
    summarizer_version: str = os.getenv("SUMMARIZER_VERSION", "sum_v1")
    transformer_model_dir: str = os.getenv(
        "TRANSFORMER_MODEL_DIR", "models/transformer_classifier"
    )
    summarizer_model: str = os.getenv(
        "SUMMARIZER_MODEL", "sshleifer/distilbart-cnn-12-6"
    )
    summarizer_max_new_tokens: int = int(os.getenv("SUMMARIZER_MAX_NEW_TOKENS", "120"))
    summarizer_min_new_tokens: int = int(os.getenv("SUMMARIZER_MIN_NEW_TOKENS", "25"))
    summarizer_num_beams: int = int(os.getenv("SUMMARIZER_NUM_BEAMS", "4"))
    summarizer_truncate_tokens: int = int(
        os.getenv("SUMMARIZER_TRUNCATE_TOKENS", "512")
    )
    # Persistence / Ops
    db_url: str = Field(
        default_factory=lambda: os.getenv(
            "DB_URL",
            "sqlite:///data/feedback.db",
        )
    )
    predict_batch_size: int = int(os.getenv("PREDICT_BATCH_SIZE", "16"))
    artifact_root: str = Field(
        default_factory=lambda: os.getenv("ARTIFACT_ROOT", "artifacts")
    )
    artifact_default_sources: str = Field(
        default_factory=lambda: os.getenv("ARTIFACT_SOURCES", "models,config")
    )
    artifact_push_default: bool = Field(
        default_factory=lambda: (
            os.getenv("ARTIFACT_PUSH_DEFAULT", "0").lower() not in {"0", "false"}
        )
    )
    artifact_store_type: str = Field(
        default_factory=lambda: os.getenv("ARTIFACT_STORE_TYPE", "local")
    )
    artifact_s3_bucket: str | None = Field(
        default_factory=lambda: os.getenv("ARTIFACT_S3_BUCKET")
    )
    artifact_s3_region: str | None = Field(
        default_factory=lambda: os.getenv("ARTIFACT_S3_REGION")
    )
    artifact_s3_endpoint: str | None = Field(
        default_factory=lambda: os.getenv("ARTIFACT_S3_ENDPOINT")
    )
    artifact_s3_prefix: str = Field(
        default_factory=lambda: os.getenv("ARTIFACT_S3_PREFIX", "")
    )
    artifact_s3_use_path_style: bool = Field(
        default_factory=lambda: (
            os.getenv("ARTIFACT_S3_PATH_STYLE", "0").lower() not in {"0", "false"}
        )
    )
    # MLflow tracking
    mlflow_enabled: bool = Field(
        default_factory=lambda: (
            os.getenv("MLFLOW_ENABLED", "0").lower() not in {"0", "false"}
        )
    )
    mlflow_tracking_uri: str = Field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    )
    mlflow_experiment: str = Field(
        default_factory=lambda: os.getenv(
            "MLFLOW_EXPERIMENT",
            "news-topic-intel",
        )
    )
    mlflow_run_template: str = Field(
        default_factory=lambda: os.getenv(
            "MLFLOW_RUN_TEMPLATE",
            "{prefix}-{timestamp}",
        )
    )
    mlflow_tags_spec: str = Field(default_factory=lambda: os.getenv("MLFLOW_TAGS", ""))
    # Active learning review thresholds
    review_conf_threshold: float = float(os.getenv("REVIEW_CONF_THRESHOLD", "0.6"))
    review_margin_threshold: float = float(os.getenv("REVIEW_MARGIN_THRESHOLD", "0.1"))
    # API security
    api_key_header: str = os.getenv("API_KEY_HEADER", "x-api-key")
    api_keys_csv: str = os.getenv("API_KEYS", os.getenv("API_KEY", ""))
    # Observability
    request_id_header: str = os.getenv("REQUEST_ID_HEADER", "x-request-id")
    log_json: bool = os.getenv("LOG_JSON", "1") not in {"0", "false", "False"}

    def allowed_api_keys(self) -> list[str]:
        if not self.api_keys_csv:
            return []
        return [key.strip() for key in self.api_keys_csv.split(",") if key.strip()]

    def mlflow_tags(self) -> Dict[str, str]:
        if not self.mlflow_tags_spec:
            return {}
        tags: Dict[str, str] = {}
        for token in re.split(r"[;,]", self.mlflow_tags_spec):
            if not token.strip():
                continue
            if "=" in token:
                key, value = token.split("=", 1)
                tags[key.strip()] = value.strip()
            else:
                tags[token.strip()] = "true"
        return tags


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
