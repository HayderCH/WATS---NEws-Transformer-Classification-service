from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.models.schemas import ClassificationRequest, ClassificationResponse
from sqlalchemy.exc import SQLAlchemyError
from app.services.classifier import classify_text
from app.core.config import get_settings
from app.db.session import SessionLocal
from app.db.models import ReviewItem

router = APIRouter()
_settings = get_settings()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/classify_news", response_model=ClassificationResponse)
def classify(req: ClassificationRequest, db: Session = Depends(get_db)):
    text = req.text if req.title is None else f"{req.title}. {req.text}"
    result = classify_text(text)
    # Auto-enqueue if below thresholds
    try:
        if (
            result.get("confidence_score", 1.0) < _settings.review_conf_threshold
            or result.get("confidence_margin", 1.0) < _settings.review_margin_threshold
        ):
            rec = ReviewItem(
                text=text,
                predicted_label=result.get("top_category", ""),
                confidence_score=float(result.get("confidence_score", 0.0)),
                confidence_margin=float(result.get("confidence_margin", 0.0)),
                model_version=result.get("model_version"),
            )
            db.add(rec)
            db.commit()
    except SQLAlchemyError:
        # Do not fail the request if queuing encounters an error
        pass
    return ClassificationResponse(**result)
