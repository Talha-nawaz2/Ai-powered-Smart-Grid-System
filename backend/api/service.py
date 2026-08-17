"""
Grid Service — business logic layer.

Concept: Notice this file has ZERO SQL in it. It only calls repo methods
and combines/interprets results. If tomorrow the "risky transformer"
threshold changes from a business meeting decision, you edit ONE line
here — the API route and the database queries never need to change.
This separation is called Separation of Concerns.
"""

from backend.api.repository import GridRepository
from backend.utils.config import settings
from backend.utils.logger import logger


class GridService:
    def __init__(self, repo: GridRepository):
        self.repo = repo

    def get_overview(self) -> dict:
        """Aggregates data across multiple entities into one dashboard-ready summary."""
        overview = {
            "total_substations": self.repo.count_substations(),
            "total_transformers": self.repo.count_transformers(),
            "total_consumers": self.repo.count_consumers(),
            "avg_transformer_health": self.repo.avg_transformer_health(),
            "risky_transformer_count": len(self.repo.get_risky_transformers()),
            "latest_demand_mw": self.repo.get_latest_demand(),
            "total_solar_capacity_mw": round(self.repo.total_solar_capacity(), 1),
            "total_wind_capacity_mw": round(self.repo.total_wind_capacity(), 1),
        }
        logger.info(f"Grid overview generated: {overview['total_transformers']} transformers, "
                     f"{overview['risky_transformer_count']} risky")
        return overview

    def list_transformers(self, limit: int = 100, region: str | None = None) -> list[dict]:
        return self.repo.get_transformers(limit=limit, region=region)

    def get_transformer(self, transformer_id: str) -> dict | None:
        return self.repo.get_transformer_by_id(transformer_id)

    def list_risky_transformers(self, threshold: float | None = None) -> list[dict]:
        """Business rule: a transformer is 'risky' below a configurable health threshold."""
        threshold = threshold if threshold is not None else 40.0
        return self.repo.get_risky_transformers(health_threshold=threshold)

    def list_consumers(self, limit: int = 100, region: str | None = None) -> list[dict]:
        return self.repo.get_consumers(limit=limit, region=region)

    def demand_history(self, limit: int = 200) -> list[dict]:
        return self.repo.get_demand_history(limit=limit)

    def weather(self, region: str | None = None, limit: int = 100) -> list[dict]:
        return self.repo.get_weather(region=region, limit=limit)

    def renewables_summary(self) -> dict:
        return {
            "solar_farms": self.repo.get_solar_farms(),
            "wind_farms": self.repo.get_wind_farms(),
            "total_solar_capacity_mw": round(self.repo.total_solar_capacity(), 1),
            "total_wind_capacity_mw": round(self.repo.total_wind_capacity(), 1),
        }
