"""
API response schemas (Pydantic models).

Concept: Without these, FastAPI would return raw dicts and anyone calling
your API has to guess the shape. With Pydantic schemas, FastAPI auto-
generates interactive docs (Swagger UI) AND validates every response,
so a bug that returns the wrong field type fails loudly during dev,
not silently in production.
"""

from datetime import date
from typing import Optional
from pydantic import BaseModel


class SubstationOut(BaseModel):
    substation_id: str
    region: str
    capacity_mva: float
    num_transformers: int
    commissioned_year: int


class TransformerOut(BaseModel):
    transformer_id: str
    substation_id: str
    capacity_kva: int
    age_years: int
    load_factor: float
    temperature_c: float
    vibration_mm_s: float
    oil_quality_index: float
    past_failures_5y: int
    health_score: float
    failed_last_year: int


class ConsumerOut(BaseModel):
    consumer_id: str
    name: str
    region: str
    consumer_type: str
    avg_monthly_kwh: float


class DemandPointOut(BaseModel):
    timestamp: str
    demand_mw: float


class GridOverviewOut(BaseModel):
    total_substations: int
    total_transformers: int
    total_consumers: int
    avg_transformer_health: float
    risky_transformer_count: int
    latest_demand_mw: Optional[float]
    total_solar_capacity_mw: float
    total_wind_capacity_mw: float
