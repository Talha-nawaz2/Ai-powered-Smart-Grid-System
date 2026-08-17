"""
Copilot Tools — the functions an LLM (or the rule-based fallback) can call.

Concept: this file is the "toolbox." Each function wraps an existing
service (forecast_service, maintenance_service, etc.) we already built and
tested in earlier sessions — the Copilot doesn't reimplement any logic, it
just gives an LLM a structured way to CALL what already exists. This is
the entire idea behind "agentic" AI systems: an LLM deciding WHICH
function to call and with WHAT arguments, based on natural language.
"""

from typing import Callable
from sqlalchemy.orm import Session

from backend.api.repository import GridRepository
from backend.api.service import GridService
from backend.forecasting.forecast_service import forecast_service
from backend.maintenance.maintenance_service import maintenance_service
from backend.theft_detection.theft_service import theft_service
from backend.knowledge_graph.graph_service import graph_service


def tool_grid_overview(db: Session, **kwargs) -> dict:
    return GridService(GridRepository(db)).get_overview()


def tool_risky_transformers(db: Session, threshold: float = 0.8, **kwargs) -> dict:
    repo = GridRepository(db)
    try:
        preds = maintenance_service.predict_many(repo.get_transformers(limit=5000))
    except RuntimeError as e:
        return {"error": str(e)}
    flagged = [p for p in preds if p["failure_probability"] >= threshold]
    return {"threshold": threshold, "count": len(flagged), "top_5": flagged[:5]}


def tool_forecast_demand(db: Session, hours: int = 24, **kwargs) -> dict:
    try:
        forecast = forecast_service.forecast(periods=hours, freq="h")
    except RuntimeError as e:
        return {"error": str(e)}
    peak = max(forecast, key=lambda f: f["predicted_demand_mw"])
    return {"horizon_hours": hours, "peak": peak, "forecast_sample": forecast[:5]}


def tool_theft_anomalies(db: Session, limit: int = 5, **kwargs) -> dict:
    repo = GridRepository(db)
    try:
        scored = theft_service.score_meters(repo.get_meter_consumption_profile(limit=5000))
    except RuntimeError as e:
        return {"error": str(e)}
    return {"count": limit, "top_anomalies": scored[:limit]}


def tool_trace_consumer_supply(db: Session, node_id: str, **kwargs) -> dict:
    try:
        path = graph_service.trace_upstream(node_id)
    except (RuntimeError, ValueError) as e:
        return {"error": str(e)}
    return {"node_id": node_id, "upstream_path": path}


def tool_critical_substations(db: Session, top_n: int = 5, **kwargs) -> dict:
    try:
        result = graph_service.critical_substations(top_n)
    except RuntimeError as e:
        return {"error": str(e)}
    return {"critical_substations": result}


TOOL_REGISTRY: dict[str, Callable] = {
    "get_grid_overview": tool_grid_overview,
    "get_risky_transformers": tool_risky_transformers,
    "forecast_demand": tool_forecast_demand,
    "get_theft_anomalies": tool_theft_anomalies,
    "trace_consumer_supply": tool_trace_consumer_supply,
    "get_critical_substations": tool_critical_substations,
}

# OpenAI Chat Completions tool-calling schema format
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_grid_overview",
            "description": "Get high-level grid KPIs: substation/transformer/consumer counts, "
                            "average transformer health, risky transformer count, latest demand, "
                            "renewable capacity.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_risky_transformers",
            "description": "Get transformers the ML model predicts are at high risk of failure, "
                            "ranked by probability.",
            "parameters": {
                "type": "object",
                "properties": {"threshold": {"type": "number", "description": "0-1, default 0.8"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forecast_demand",
            "description": "Forecast grid electricity demand for the next N hours (Prophet model).",
            "parameters": {
                "type": "object",
                "properties": {"hours": {"type": "integer", "description": "default 24"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_theft_anomalies",
            "description": "Get the meters most likely committing energy theft, based on "
                            "anomalous consumption behavior (unsupervised detection).",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "description": "default 5"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trace_consumer_supply",
            "description": "Trace the electrical supply chain for a consumer/meter/transformer "
                            "ID up to its power plant. Use for outage explanations.",
            "parameters": {
                "type": "object",
                "properties": {"node_id": {"type": "string", "description": "e.g. CUS-000000"}},
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_critical_substations",
            "description": "Get substations ranked by how many transformers depend on them "
                            "(cascade-failure risk).",
            "parameters": {
                "type": "object",
                "properties": {"top_n": {"type": "integer", "description": "default 5"}},
                "required": [],
            },
        },
    },
]
