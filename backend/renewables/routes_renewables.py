"""
Renewable Intelligence API routes.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session

from backend.utils.db import get_db
from backend.api.repository import GridRepository
from backend.renewables.renewable_service import renewable_service

router = APIRouter(prefix="/api/renewables", tags=["Renewable Intelligence"])


@router.post("/train")
def train_renewable_models(db: Session = Depends(get_db)):
    """Trains both the solar and wind generation regressors."""
    repo = GridRepository(db)
    solar_data = repo.get_solar_training_data()
    wind_data = repo.get_wind_training_data()
    if not solar_data or not wind_data:
        raise HTTPException(status_code=400, detail="Missing generation logs. Run load_to_sqlite.py first.")

    solar_metrics = renewable_service.train_solar(solar_data)
    wind_metrics = renewable_service.train_wind(wind_data)
    return {"status": "trained", "solar": solar_metrics, "wind": wind_metrics}


@router.get("/predict/solar")
def predict_solar(
    solar_irradiance_wm2: float = Query(..., ge=0, le=1200),
    temperature_c: float = Query(..., ge=-10, le=55),
    capacity_mw: float = Query(..., gt=0),
):
    try:
        prediction = renewable_service.predict_solar(solar_irradiance_wm2, temperature_c, capacity_mw)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"predicted_generation_mw": prediction}


@router.get("/predict/wind")
def predict_wind(
    wind_speed_kmh: float = Query(..., ge=0, le=150),
    capacity_mw: float = Query(..., gt=0),
):
    try:
        prediction = renewable_service.predict_wind(wind_speed_kmh, capacity_mw)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"predicted_generation_mw": prediction}


@router.get("/battery/recommendation/{battery_id}")
def battery_recommendation(
    battery_id: str,
    current_generation_mw: float = Query(..., description="Current combined solar+wind output (MW)"),
    current_demand_mw: float = Query(..., description="Current grid demand (MW)"),
    db: Session = Depends(get_db),
):
    """Rule-based charge/discharge/hold recommendation for a specific battery asset."""
    repo = GridRepository(db)
    batteries = repo.get_battery_storage()
    battery = next((b for b in batteries if b["battery_id"] == battery_id), None)
    if not battery:
        raise HTTPException(status_code=404, detail=f"Battery '{battery_id}' not found")

    return renewable_service.recommend_battery_action(current_generation_mw, current_demand_mw, battery)
