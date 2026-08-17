"""
Central configuration for the Smart Grid Intelligence Platform.

Concept for you (Talha): This is the "single source of truth" pattern.
Instead of hardcoding paths/keys all over the codebase, every module imports
`settings` from here. Enterprise systems ALWAYS centralize config so you can
change environments (dev/staging/prod) by only editing .env, not code.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent.parent  # smart-grid/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")

    # App
    APP_NAME: str = "Smart Grid Intelligence Platform"
    ENV: str = "development"
    DEBUG: bool = True

    # Paths
    DATA_DIR: Path = BASE_DIR / "backend" / "data"
    SAMPLE_DATA_DIR: Path = BASE_DIR / "backend" / "data" / "sample_data"
    DB_PATH: Path = BASE_DIR / "backend" / "data" / "smart_grid.db"

    # Database
    DATABASE_URL: str = f"sqlite:///{DB_PATH}"

    # ML thresholds (enterprise systems externalize thresholds, not hardcode them)
    TRANSFORMER_FAILURE_ALERT_THRESHOLD: float = 0.80
    DEMAND_SPIKE_THRESHOLD_MW: float = 950.0
    THEFT_ANOMALY_SCORE_THRESHOLD: float = -0.15

    # LLM / Copilot
    # Gemini is the recommended default: Google's free tier requires no credit
    # card and includes function/tool-calling (unlike OpenAI, which requires
    # paid credits from the start). OpenAI is kept as an optional alternative -
    # if both keys are set, Gemini is tried first.
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    OPENAI_API_KEY: str = ""
    COPILOT_MODEL: str = "gpt-4o-mini"

    # Data generation
    RANDOM_SEED: int = 42
    MAX_ROWS_PER_DATASET: int = 5000


settings = Settings()
