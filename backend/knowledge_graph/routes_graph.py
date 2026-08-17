"""
Knowledge Graph API routes.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session

from backend.utils.db import get_db
from backend.api.repository import GridRepository
from backend.knowledge_graph.graph_service import graph_service
from backend.utils.logger import logger

router = APIRouter(prefix="/api/graph", tags=["Knowledge Graph"])


@router.post("/build")
def build_graph(db: Session = Depends(get_db)):
    """Builds (or rebuilds) the full grid knowledge graph from the database."""
    repo = GridRepository(db)
    try:
        stats = graph_service.build(repo)
        return {"status": "built", **stats}
    except Exception as e:
        logger.exception("Graph build failed")
        raise HTTPException(status_code=500, detail=f"Graph build failed: {type(e).__name__}: {e}")


@router.get("/stats")
def graph_stats():
    try:
        return graph_service.get_stats()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/trace/{node_id}")
def trace_upstream(node_id: str):
    """Traces a node (consumer/meter/transformer) all the way up to its power plant(s)."""
    try:
        return {"node_id": node_id, "upstream_path": graph_service.trace_upstream(node_id)}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/path")
def shortest_path(source: str, target: str):
    """Finds the shortest electrical path between any two nodes in the grid."""
    try:
        return graph_service.shortest_path(source, target)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/neighbors/{node_id}")
def neighbors(node_id: str):
    try:
        return graph_service.neighbors(node_id)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/critical-substations")
def critical_substations(top_n: int = Query(10, ge=1, le=50)):
    """Ranks substations by how many transformers depend on them (cascade-failure risk)."""
    try:
        return {"critical_substations": graph_service.critical_substations(top_n)}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
