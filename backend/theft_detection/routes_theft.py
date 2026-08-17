"""
Energy Theft Detection API routes.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session

from backend.utils.db import get_db
from backend.api.repository import GridRepository
from backend.theft_detection.theft_service import theft_service
from backend.utils.logger import logger

router = APIRouter(prefix="/api/theft", tags=["Theft Detection"])


@router.post("/train")
def train_theft_model(
    contamination: float = Query(0.05, ge=0.001, le=0.5,
                                  description="Expected fraction of anomalous meters"),
    db: Session = Depends(get_db),
):
    """Trains an unsupervised Isolation Forest on meter consumption behavior."""
    repo = GridRepository(db)
    meters = repo.get_meter_consumption_profile(limit=5000)
    if not meters:
        raise HTTPException(status_code=400, detail="No meter consumption data found. Run load_to_sqlite.py first.")
    try:
        metrics = theft_service.train(meters, contamination=contamination)
        return {"status": "trained", "metrics": metrics}
    except Exception as e:
        logger.exception("Theft model training failed")
        raise HTTPException(status_code=500, detail=f"Training failed: {type(e).__name__}: {e}")


@router.get("/anomalies")
def list_anomalies(
    limit: int = Query(20, ge=1, le=500, description="How many top-anomalous meters to return"),
    db: Session = Depends(get_db),
):
    """Returns the most anomalous meters, ranked most-suspicious first."""
    repo = GridRepository(db)
    meters = repo.get_meter_consumption_profile(limit=5000)
    try:
        scored = theft_service.score_meters(meters)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"count": limit, "meters": scored[:limit]}


@router.get("/predict/{meter_id}")
def predict_meter(meter_id: str, db: Session = Depends(get_db)):
    """Scores a single meter for anomalous (potential theft) behavior."""
    repo = GridRepository(db)
    meter = repo.get_meter_profile_by_id(meter_id)
    if not meter:
        raise HTTPException(status_code=404, detail=f"Meter '{meter_id}' not found in consumption profile")
    try:
        result = theft_service.score_meters([meter])
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result[0]
