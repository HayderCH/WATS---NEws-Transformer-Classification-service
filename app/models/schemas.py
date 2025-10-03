from datetime import datetime

from pydantic import BaseModel, Field

try:
    # Pydantic v2 ConfigDict (preferred)
    from pydantic import ConfigDict  # type: ignore
except ImportError:  # pragma: no cover - fallback for older versions
    ConfigDict = None  # type: ignore
from typing import List, Optional


class HealthResponse(BaseModel):
    status: str = "ok"


class ClassificationRequest(BaseModel):
    title: Optional[str] = None
    text: str = Field(..., min_length=10, description="Full article text")


class CategoryProbability(BaseModel):
    name: str
    prob: float


class ClassificationResponse(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(protected_namespaces=())
    top_category: str
    categories: List[CategoryProbability]
    confidence_level: str
    confidence_score: float
    confidence_margin: float
    model_version: str
    latency_ms: float
    suggestion: Optional[str] = None


class SummarizationRequest(BaseModel):
    text: str = Field(..., min_length=10)
    max_len: int = Field(120, ge=16, le=512)
    min_len: int = Field(25, ge=4, le=400)


class SummarizationResponse(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(protected_namespaces=())
    summary: str
    model_version: str
    latency_ms: float
    cached: bool | None = False


# Batch Summarization
class SummarizationBatchItem(BaseModel):
    text: str
    max_len: int | None = None
    min_len: int | None = None


class SummarizationBatchRequest(BaseModel):
    items: List[SummarizationBatchItem]


class SummarizationBatchItemResponse(BaseModel):
    summary: str
    model_version: str
    latency_ms: float
    cached: bool | None = False


class SummarizationBatchResponse(BaseModel):
    results: List[SummarizationBatchItemResponse]


# Batch Classification
class BatchItem(BaseModel):
    title: Optional[str] = None
    text: str = Field(..., min_length=10)


class BatchClassificationRequest(BaseModel):
    items: List[BatchItem]
    top_k: int = Field(5, ge=1, le=42)


class BatchClassificationResponse(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(protected_namespaces=())
    results: List[ClassificationResponse]


# Feedback
class FeedbackIn(BaseModel):
    text: str = Field(..., min_length=10)
    predicted_label: str
    true_label: Optional[str] = None
    model_version: Optional[str] = None
    confidence_score: Optional[float] = None
    confidence_margin: Optional[float] = None


class FeedbackAck(BaseModel):
    status: str = "ok"
    id: int


class FeedbackStats(BaseModel):
    by_predicted: dict
    by_true: dict


# Review API models
class ReviewEnqueueIn(BaseModel):
    text: str
    predicted_label: str
    confidence_score: float
    confidence_margin: float
    model_version: Optional[str] = None
    top_labels: Optional[List[CategoryProbability]] = None


class ReviewQueuedAck(BaseModel):
    queued: bool
    id: Optional[int] = None


class ReviewLabelIn(BaseModel):
    item_id: int
    true_label: str


class TrendBucket(BaseModel):
    date: str
    label: str
    count: int


class TrendTotal(BaseModel):
    label: str
    count: int


class TrendsResponse(BaseModel):
    window_days: int = Field(..., ge=1)
    generated_at: datetime
    buckets: List[TrendBucket]
    totals: List[TrendTotal]
