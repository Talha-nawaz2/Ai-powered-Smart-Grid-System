"""
Grid Data Integration API routes.

Concept: Dependency Injection (DI) — instead of each route creating its
own DB connection and repository, FastAPI's `Depends()` builds the chain
for us: get_db -> GridRepository -> GridService -> route function.
This makes routes trivially testable: swap `get_grid_service` with a fake
in tests and no route code changes.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.utils.db import get_db
from backend.api.repository import GridRepository
from backend.api.service import GridService

router = APIRouter(prefix="/api/grid", tags=["Grid Data"])


def get_grid_service(db: Session = Depends(get_db)) -> GridService:
    repo = GridRepository(db)
    return GridService(repo)


@router.get("/overview")
def grid_overview(service: GridService = Depends(get_grid_service)):
    """High-level KPIs for the executive dashboard."""
    return service.get_overview()


@router.get("/substations")
def list_substations(
    limit: int = Query(100, le=500),
    service: GridService = Depends(get_grid_service),
):
    return service.repo.get_substations(limit=limit)


@router.get("/transformers")
def list_transformers(
    limit: int = Query(100, le=500),
    region: Optional[str] = None,
    service: GridService = Depends(get_grid_service),
):
    return service.list_transformers(limit=limit, region=region)


@router.get("/transformers/risky")
def risky_transformers(
    threshold: float = Query(40.0, ge=0, le=100, description="Health score below which a transformer is flagged"),
    service: GridService = Depends(get_grid_service),
):
    results = service.list_risky_transformers(threshold=threshold)
    return {"threshold": threshold, "count": len(results), "transformers": results}


@router.get("/transformers/{transformer_id}")
def get_transformer(transformer_id: str, service: GridService = Depends(get_grid_service)):
    result = service.get_transformer(transformer_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Transformer '{transformer_id}' not found")
    return result


@router.get("/consumers")
def list_consumers(
    limit: int = Query(100, le=500),
    region: Optional[str] = None,
    service: GridService = Depends(get_grid_service),
):
    return service.list_consumers(limit=limit, region=region)


@router.get("/demand")
def demand_history(
    limit: int = Query(200, le=2000),
    service: GridService = Depends(get_grid_service),
):
    return service.demand_history(limit=limit)


@router.get("/weather")
def weather(
    region: Optional[str] = None,
    limit: int = Query(100, le=1000),
    service: GridService = Depends(get_grid_service),
):
    return service.weather(region=region, limit=limit)


@router.get("/renewables")
def renewables_summary(service: GridService = Depends(get_grid_service)):
    return service.renewables_summary()
