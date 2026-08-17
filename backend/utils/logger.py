"""
Centralized logging.

Concept: Enterprise apps never use print(). They use structured loggers so
logs can be filtered by level (INFO/WARNING/ERROR), written to files, and
later shipped to monitoring tools (Datadog, ELK, etc.). We simulate that
here with loguru, writing to console + a rotating log file.
"""

import sys
from pathlib import Path
from loguru import logger

from backend.utils.config import settings

LOG_DIR = settings.DATA_DIR.parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()  # remove default handler
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{module}</cyan> - {message}")
logger.add(LOG_DIR / "smart_grid.log", level="DEBUG", rotation="5 MB", retention=3)

__all__ = ["logger"]
