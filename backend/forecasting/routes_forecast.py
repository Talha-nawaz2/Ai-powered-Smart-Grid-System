"""
Demand Forecasting API routes.

Concept: notice the pattern repeats from Session 2 — route depends on a
service via Depends(). The only difference is this service wraps an ML
model instead of a database repository. Same architecture, different
engine underneath. That's the point of layering: the SHAPE of the code
stays consistent as the platform grows.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session

from backend.utils.db import get_db
from backend.api.repository import GridRepository
from backend.forecasting.forecast_service import forecast_service
from backend.utils.logger import logger

router = APIRouter(prefix="/api/forecast", tags=["Demand Forecasting"])


@router.post("/train")
def train_forecast_model(db: Session = Depends(get_db)):
    """Trains (or retrains) the Prophet model on the full grid_demand_hourly history."""
    repo = GridRepository(db)
    history = repo.get_demand_history(limit=5000)
    if not history:
        raise HTTPException(status_code=400, detail="No demand history found. Run load_to_sqlite.py first.")
    try:
        result = forecast_service.train(history)
        return {"status": "trained", **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # Actionable setup issue (e.g. CmdStan not installed) - 400, not 500,
        # since it's not a server bug, it's an environment fix the user can make.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Surface the real error instead of a bare 500 - critical for debugging
        # environment-specific issues (e.g. Prophet/cmdstan install problems)
        # that only show up on the student's machine, not in development.
        logger.exception("Forecast training failed")
        raise HTTPException(status_code=500, detail=f"Training failed: {type(e).__name__}: {e}")


@router.get("/hourly")
def forecast_hourly(hours: int = Query(24, ge=1, le=168)):
    """Forecast the next N hours (default 24, max 1 week)."""
    try:
        return {"horizon": f"{hours}h", "mode": forecast_service.mode,
                "forecast": forecast_service.forecast(periods=hours, freq="h")}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/daily")
def forecast_daily(days: int = Query(7, ge=1, le=30)):
    """Forecast the next N days."""
    try:
        return {"horizon": f"{days}d", "forecast": forecast_service.forecast(periods=days, freq="D")}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/weekly")
def forecast_weekly(weeks: int = Query(4, ge=1, le=12)):
    """Forecast the next N weeks."""
    try:
        return {"horizon": f"{weeks}w", "forecast": forecast_service.forecast(periods=weeks, freq="W")}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
