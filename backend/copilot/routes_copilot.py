"""
AI Grid Copilot API route.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.utils.db import get_db
from backend.copilot.copilot_service import copilot_service

router = APIRouter(prefix="/api/copilot", tags=["AI Grid Copilot"])


class CopilotQuery(BaseModel):
    query: str


@router.post("/ask")
def ask_copilot(payload: CopilotQuery, db: Session = Depends(get_db)):
    """
    Ask the Grid Copilot a natural-language question. Uses OpenAI tool-calling
    if OPENAI_API_KEY is configured, otherwise falls back to rule-based
    intent routing over the same underlying tools.
    """
    return copilot_service.answer(payload.query, db)
