"""
FastAPI application entrypoint.

Run:  uvicorn backend.api.main:app --reload
Docs: http://127.0.0.1:8000/docs   (auto-generated Swagger UI)
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.utils.config import settings
from backend.utils.logger import logger
from backend.api.routes_grid import router as grid_router
from backend.forecasting.routes_forecast import router as forecast_router
from backend.maintenance.routes_maintenance import router as maintenance_router
from backend.theft_detection.routes_theft import router as theft_router
from backend.renewables.routes_renewables import router as renewables_router
from backend.knowledge_graph.routes_graph import router as graph_router
from backend.copilot.routes_copilot import router as copilot_router
from backend.dashboard.routes_dashboard import router as dashboard_router
from backend.alerts.routes_alerts import router as alerts_router

app = FastAPI(
    title=settings.APP_NAME,
    description="Enterprise Smart Grid Intelligence & Energy Optimization Platform",
    version="0.2.0",
)

# CORS: allows the future dashboard frontend (Session 9) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Self-hosted static assets (Chart.js, Plotly). These are served from OUR own
# server instead of an external CDN (cdnjs) - this is a deliberate reliability
# fix: some networks/ISPs block or throttle third-party CDN domains, which
# silently breaks any dashboard that depends on them. Self-hosting removes
# that entire class of failure and also means the dashboard works even with
# no internet access beyond reaching this server itself.
STATIC_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(grid_router)
app.include_router(forecast_router)
app.include_router(maintenance_router)
app.include_router(theft_router)
app.include_router(renewables_router)
app.include_router(graph_router)
app.include_router(copilot_router)
app.include_router(dashboard_router)
app.include_router(alerts_router)


@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "app": settings.APP_NAME, "env": settings.ENV}


@app.on_event("startup")
def on_startup():
    logger.info(f"{settings.APP_NAME} starting up in '{settings.ENV}' mode")
