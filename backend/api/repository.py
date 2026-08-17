"""
Grid Repository — Repository Pattern implementation.

Concept: Every query touching the database lives HERE and nowhere else.
The Service layer never writes SQL; it just calls repo methods and gets
back plain Python dicts/lists. This is what lets you unit-test the
Service layer with a FAKE repository (no real DB needed) later on.
"""

from typing import Optional
from sqlalchemy import text
from sqlalchemy.orm import Session


class GridRepository:
    def __init__(self, db: Session):
        self.db = db

    def _rows(self, query: str, params: Optional[dict] = None) -> list[dict]:
        result = self.db.execute(text(query), params or {})
        return [dict(row) for row in result.mappings().all()]

    # ---------- Substations ----------
    def get_substations(self, limit: int = 100) -> list[dict]:
        return self._rows("SELECT * FROM substations LIMIT :limit", {"limit": limit})

    def count_substations(self) -> int:
        return self._rows("SELECT COUNT(*) as c FROM substations")[0]["c"]

    # ---------- Transformers ----------
    def get_transformers(self, limit: int = 100, region: Optional[str] = None) -> list[dict]:
        if region:
            q = """SELECT t.* FROM transformers t
                    JOIN substations s ON t.substation_id = s.substation_id
                    WHERE s.region = :region LIMIT :limit"""
            return self._rows(q, {"region": region, "limit": limit})
        return self._rows("SELECT * FROM transformers LIMIT :limit", {"limit": limit})

    def get_transformer_by_id(self, transformer_id: str) -> Optional[dict]:
        rows = self._rows(
            "SELECT * FROM transformers WHERE transformer_id = :tid", {"tid": transformer_id}
        )
        return rows[0] if rows else None

    def count_transformers(self) -> int:
        return self._rows("SELECT COUNT(*) as c FROM transformers")[0]["c"]

    def avg_transformer_health(self) -> float:
        return round(self._rows("SELECT AVG(health_score) as a FROM transformers")[0]["a"], 2)

    def get_risky_transformers(self, health_threshold: float = 40.0) -> list[dict]:
        return self._rows(
            "SELECT * FROM transformers WHERE health_score < :t ORDER BY health_score ASC",
            {"t": health_threshold},
        )

    # ---------- Consumers ----------
    def get_consumers(self, limit: int = 100, region: Optional[str] = None) -> list[dict]:
        if region:
            return self._rows(
                "SELECT * FROM consumers WHERE region = :region LIMIT :limit",
                {"region": region, "limit": limit},
            )
        return self._rows("SELECT * FROM consumers LIMIT :limit", {"limit": limit})

    def count_consumers(self) -> int:
        return self._rows("SELECT COUNT(*) as c FROM consumers")[0]["c"]

    # ---------- Demand ----------
    def get_demand_history(self, limit: int = 200) -> list[dict]:
        return self._rows(
            "SELECT timestamp, demand_mw FROM grid_demand_hourly ORDER BY timestamp DESC LIMIT :limit",
            {"limit": limit},
        )

    def get_latest_demand(self) -> Optional[float]:
        rows = self._rows("SELECT demand_mw FROM grid_demand_hourly ORDER BY timestamp DESC LIMIT 1")
        return rows[0]["demand_mw"] if rows else None

    # ---------- Weather ----------
    def get_weather(self, region: Optional[str] = None, limit: int = 100) -> list[dict]:
        if region:
            return self._rows(
                "SELECT * FROM weather WHERE region = :region ORDER BY timestamp DESC LIMIT :limit",
                {"region": region, "limit": limit},
            )
        return self._rows("SELECT * FROM weather ORDER BY timestamp DESC LIMIT :limit", {"limit": limit})

    # ---------- Renewables ----------
    def get_solar_farms(self) -> list[dict]:
        return self._rows("SELECT * FROM solar_farms")

    def get_wind_farms(self) -> list[dict]:
        return self._rows("SELECT * FROM wind_farms")

    def total_solar_capacity(self) -> float:
        return self._rows("SELECT SUM(capacity_mw) as s FROM solar_farms")[0]["s"] or 0.0

    def total_wind_capacity(self) -> float:
        return self._rows("SELECT SUM(capacity_mw) as s FROM wind_farms")[0]["s"] or 0.0

    # ---------- Meter Consumption (theft detection) ----------
    def get_meter_consumption_profile(self, limit: int = 5000) -> list[dict]:
        return self._rows("SELECT * FROM meter_consumption_profile LIMIT :limit", {"limit": limit})

    def get_meter_profile_by_id(self, meter_id: str) -> Optional[dict]:
        rows = self._rows(
            "SELECT * FROM meter_consumption_profile WHERE meter_id = :mid", {"mid": meter_id}
        )
        return rows[0] if rows else None

    # ---------- Renewables (generation + battery) ----------
    def get_solar_training_data(self) -> list[dict]:
        q = """
        SELECT s.generation_mw, w.solar_irradiance_wm2, w.temperature_c, f.capacity_mw
        FROM solar_generation_log s
        JOIN weather w ON s.timestamp = w.timestamp AND s.region = w.region
        JOIN solar_farms f ON s.farm_id = f.farm_id
        """
        return self._rows(q)

    def get_wind_training_data(self) -> list[dict]:
        q = """
        SELECT s.generation_mw, w.wind_speed_kmh, f.capacity_mw
        FROM wind_generation_log s
        JOIN weather w ON s.timestamp = w.timestamp AND s.region = w.region
        JOIN wind_farms f ON s.farm_id = f.farm_id
        """
        return self._rows(q)

    def get_battery_storage(self) -> list[dict]:
        return self._rows("SELECT * FROM battery_storage")

    def get_latest_weather_by_region(self, region: str) -> Optional[dict]:
        rows = self._rows(
            "SELECT * FROM weather WHERE region = :region ORDER BY timestamp DESC LIMIT 1",
            {"region": region},
        )
        return rows[0] if rows else None

    # ---------- Knowledge Graph ----------
    def get_power_plants(self) -> list[dict]:
        return self._rows("SELECT * FROM power_plants")

    def get_transmission_lines(self) -> list[dict]:
        return self._rows("SELECT * FROM transmission_lines")

    def get_meters(self, limit: int = 5000) -> list[dict]:
        return self._rows("SELECT * FROM smart_meters LIMIT :limit", {"limit": limit})
