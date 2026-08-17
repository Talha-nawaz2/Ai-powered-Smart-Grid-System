"""
Dashboard route.

Concept: this route just renders the HTML shell. ALL the actual data comes
from the same REST APIs built in Sessions 2-8, fetched client-side via
JavaScript. This is a clean split: FastAPI serves the page once, then the
browser talks directly to your existing API — no server-side templating of
data needed, and the dashboard automatically reflects Sessions 1-8's real
Grid Data, Forecasting, Maintenance, Theft, Renewables, and Graph modules.
"""

from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter(tags=["Dashboard"])
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@router.get("/dashboard")
def dashboard(request: Request):
    # Note: newer Starlette requires `request` as the first positional arg
    # to TemplateResponse, not inside the context dict (that was the old API).
    return templates.TemplateResponse(request, "dashboard.html", {})
