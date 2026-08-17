"""
Alert Center Service.

Concept: this file deliberately contains ZERO machine learning. It only
reads outputs already produced by services from Sessions 3-6 (maintenance
predictions, demand history, theft scores, battery state) and applies
simple, explainable threshold rules. This is intentional: alert logic in
real operations tooling needs to be instantly understandable and tunable
by a human on call at 2am, not another model to debug under pressure.
"""

from sqlalchemy.orm import Session

from backend.api.repository import GridRepository
from backend.maintenance.maintenance_service import maintenance_service
from backend.theft_detection.theft_service import theft_service
from backend.utils.config import settings
from backend.utils.logger import logger

SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3}


def generate_alerts(db: Session) -> list[dict]:
    repo = GridRepository(db)
    alerts: list[dict] = []

    # ---------- Maintenance alerts ----------
    try:
        preds = maintenance_service.predict_many(repo.get_transformers(limit=5000))
        for p in preds:
            if p["failure_probability"] >= settings.TRANSFORMER_FAILURE_ALERT_THRESHOLD:
                alerts.append({
                    "type": "MAINTENANCE",
                    "severity": "CRITICAL" if p["failure_probability"] >= 0.9 else "HIGH",
                    "message": f"Transformer {p['transformer_id']} at "
                               f"{p['failure_probability']:.0%} failure risk",
                    "entity_id": p["transformer_id"],
                })
    except RuntimeError:
        pass  # model not trained yet - silently skip this category, not an error itself

    # ---------- Demand alerts ----------
    latest_demand = repo.get_latest_demand()
    if latest_demand and latest_demand >= settings.DEMAND_SPIKE_THRESHOLD_MW:
        alerts.append({
            "type": "DEMAND",
            "severity": "HIGH",
            "message": f"Grid demand at {latest_demand} MW exceeds threshold "
                       f"{settings.DEMAND_SPIKE_THRESHOLD_MW} MW",
            "entity_id": "GRID",
        })

    # ---------- Theft alerts ----------
    try:
        scored = theft_service.score_meters(repo.get_meter_consumption_profile(limit=5000))
        for m in [m for m in scored if m["is_anomalous"]][:10]:
            alerts.append({
                "type": "THEFT",
                "severity": "HIGH",
                "message": f"Meter {m['meter_id']} flagged anomalous "
                           f"(score {m['anomaly_score']})",
                "entity_id": m["meter_id"],
            })
    except RuntimeError:
        pass

    # ---------- Renewable / battery alerts ----------
    for b in repo.get_battery_storage():
        if b["current_charge_pct"] < 15:
            alerts.append({
                "type": "RENEWABLE",
                "severity": "MODERATE",
                "message": f"Battery {b['battery_id']} charge critically low "
                           f"({b['current_charge_pct']}%)",
                "entity_id": b["battery_id"],
            })

    alerts.sort(key=lambda a: SEVERITY_ORDER.get(a["severity"], 4))
    logger.info(f"Generated {len(alerts)} alerts "
                 f"({sum(1 for a in alerts if a['severity']=='CRITICAL')} critical)")
    return alerts
