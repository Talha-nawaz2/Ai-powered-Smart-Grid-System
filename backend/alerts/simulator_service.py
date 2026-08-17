"""
Grid Simulator.

Concept: since there's no real SCADA feed, we simulate one. Each "tick"
writes small, physically-plausible perturbations directly into SQLite:
transformer sensor drift (recomputed with the SAME health_score formula
used in Session 1's generator, so the data stays internally consistent),
a new live demand reading appended to the time series, and battery charge
drift. This is what makes the dashboard feel alive on repeated clicks
instead of showing one static snapshot forever.
"""

import random
import datetime
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.utils.logger import logger


def simulate_tick(db: Session) -> dict:
    changes = {"transformers_updated": 0, "demand_points_added": 0, "batteries_updated": 0}

    # ---------- 1. Jitter a random sample of transformers ----------
    rows = db.execute(text("""
        SELECT transformer_id, age_years, temperature_c, vibration_mm_s,
               oil_quality_index, past_failures_5y
        FROM transformers ORDER BY RANDOM() LIMIT 20
    """)).mappings().all()

    for r in rows:
        new_temp = max(25, min(115, r["temperature_c"] + random.uniform(-3, 4)))
        new_vib = max(0.2, r["vibration_mm_s"] + random.uniform(-0.3, 0.4))
        new_oil = max(5, min(100, r["oil_quality_index"] + random.uniform(-2, 1)))
        # Same composite formula as Session 1's gen_transformers() - keeps
        # health_score internally consistent with what drives it.
        new_health = max(0, min(100,
            100 - (r["age_years"] * 0.9) - (new_temp - 55) * 0.5
            - (new_vib * 4) - (r["past_failures_5y"] * 6) + (new_oil * 0.2)
        ))
        new_failed = 1 if new_health < 40 else 0

        db.execute(text("""
            UPDATE transformers
            SET temperature_c=:t, vibration_mm_s=:v, oil_quality_index=:o,
                health_score=:h, failed_last_year=:f
            WHERE transformer_id=:tid
        """), {"t": round(new_temp, 1), "v": round(new_vib, 2), "o": round(new_oil, 1),
               "h": round(new_health, 1), "f": new_failed, "tid": r["transformer_id"]})
        changes["transformers_updated"] += 1

    # ---------- 2. Append a new live demand reading ----------
    last = db.execute(text(
        "SELECT demand_mw FROM grid_demand_hourly ORDER BY timestamp DESC LIMIT 1"
    )).mappings().first()
    last_val = last["demand_mw"] if last else 700.0
    new_val = max(300, last_val + random.uniform(-25, 25))
    now = datetime.datetime.utcnow()
    db.execute(text("INSERT INTO grid_demand_hourly (timestamp, demand_mw) VALUES (:ts, :v)"),
               {"ts": str(now), "v": round(new_val, 1)})
    changes["demand_points_added"] = 1

    # ---------- 3. Jitter battery charge on a few batteries ----------
    batteries = db.execute(text(
        "SELECT battery_id, current_charge_pct FROM battery_storage ORDER BY RANDOM() LIMIT 5"
    )).mappings().all()
    for b in batteries:
        new_charge = max(0, min(100, b["current_charge_pct"] + random.uniform(-8, 8)))
        db.execute(text("UPDATE battery_storage SET current_charge_pct=:c WHERE battery_id=:bid"),
                   {"c": round(new_charge, 1), "bid": b["battery_id"]})
        changes["batteries_updated"] += 1

    db.commit()
    logger.info(f"Simulator tick applied: {changes}")
    return changes
