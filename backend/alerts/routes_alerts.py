"""
Alert Center + Simulator API routes.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.utils.db import get_db
from backend.utils.logger import logger
from backend.alerts.alert_service import generate_alerts
from backend.alerts.simulator_service import simulate_tick

router = APIRouter(prefix="/api/alerts", tags=["Alerts & Simulator"])


@router.get("")
def list_alerts(db: Session = Depends(get_db)):
    """Generates the current alert list by applying threshold rules to existing model outputs."""
    alerts = generate_alerts(db)
    counts = {}
    for a in alerts:
        counts[a["severity"]] = counts.get(a["severity"], 0) + 1
    return {"total": len(alerts), "counts_by_severity": counts, "alerts": alerts}


@router.post("/simulate-tick")
def run_simulator_tick(db: Session = Depends(get_db)):
    """Applies one round of random sensor drift to transformers, demand, and batteries."""
    try:
        changes = simulate_tick(db)
        return {"status": "simulated", **changes}
    except Exception as e:
        logger.exception("Simulator tick failed")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {type(e).__name__}: {e}")
