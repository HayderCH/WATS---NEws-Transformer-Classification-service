from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.schemas import TrendsResponse
from app.services.trends import compute_trend_snapshot

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/trends", response_model=TrendsResponse)
def read_trends(
    window: int = Query(7, ge=1, le=90),
    db: Session = Depends(get_db),
):
    snapshot = compute_trend_snapshot(db, window_days=window)
    return {
        "window_days": window,
        "generated_at": datetime.now(timezone.utc),
        **snapshot,
    }
