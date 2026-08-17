"""
Predictive Maintenance API routes.

Same DI pattern as forecasting: route -> service -> (repository for data).
The route layer stays thin; all ML logic lives in maintenance_service.py.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session

from backend.utils.db import get_db
from backend.api.repository import GridRepository
from backend.maintenance.maintenance_service import maintenance_service
from backend.utils.logger import logger

router = APIRouter(prefix="/api/maintenance", tags=["Predictive Maintenance"])


@router.post("/train")
def train_maintenance_model(db: Session = Depends(get_db)):
    """Trains an XGBoost classifier on all transformers to predict failure probability."""
    repo = GridRepository(db)
    transformers = repo.get_transformers(limit=5000)
    if not transformers:
        raise HTTPException(status_code=400, detail="No transformer data found. Run load_to_sqlite.py first.")
    try:
        metrics = maintenance_service.train(transformers)
        return {"status": "trained", "metrics": metrics}
    except Exception as e:
        logger.exception("Maintenance training failed")
        raise HTTPException(status_code=500, detail=f"Training failed: {type(e).__name__}: {e}")


@router.get("/predict/{transformer_id}")
def predict_transformer_failure(transformer_id: str, db: Session = Depends(get_db)):
    """Predicts failure probability for a single transformer."""
    repo = GridRepository(db)
    transformer = repo.get_transformer_by_id(transformer_id)
    if not transformer:
        raise HTTPException(status_code=404, detail=f"Transformer '{transformer_id}' not found")
    try:
        return maintenance_service.predict_one(transformer)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/at-risk")
def at_risk_transformers(
    threshold: float = Query(0.80, ge=0, le=1, description="Failure probability threshold for alerting"),
    limit: int = Query(500, le=5000),
    db: Session = Depends(get_db),
):
    """Ranks all transformers by predicted failure probability; used to feed the Alert Center."""
    repo = GridRepository(db)
    transformers = repo.get_transformers(limit=limit)
    try:
        predictions = maintenance_service.predict_many(transformers)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    flagged = [p for p in predictions if p["failure_probability"] >= threshold]
    return {"threshold": threshold, "flagged_count": len(flagged), "transformers": flagged}
