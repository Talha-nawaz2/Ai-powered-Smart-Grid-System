"""
CSV -> SQLite Loader

Concept: This is the "Extract-Load" half of a mini ETL pipeline. We keep it
separate from generate_data.py (generation) and from the future repository
layer (querying) — each file has ONE job. That separation is what "layered
architecture" means in practice, not just a buzzword in the README.

Run: python -m backend.data.load_to_sqlite
"""

import pandas as pd
from sqlalchemy import inspect

from backend.utils.config import settings
from backend.utils.db import engine
from backend.utils.logger import logger

TABLES = {
    "substations": "substations.csv",
    "power_plants": "power_plants.csv",
    "solar_farms": "solar_farms.csv",
    "wind_farms": "wind_farms.csv",
    "battery_storage": "battery_storage.csv",
    "transformers": "transformers.csv",
    "consumers": "consumers.csv",
    "smart_meters": "smart_meters.csv",
    "meter_consumption_profile": "meter_consumption_profile.csv",
    "weather": "weather.csv",
    "grid_demand_hourly": "grid_demand_hourly.csv",
    "solar_generation_log": "solar_generation_log.csv",
    "wind_generation_log": "wind_generation_log.csv",
    "transmission_lines": "transmission_lines.csv",
}


def load_all():
    logger.info(f"Loading CSVs into SQLite at {settings.DB_PATH}")
    for table_name, filename in TABLES.items():
        path = settings.SAMPLE_DATA_DIR / filename
        if not path.exists():
            logger.warning(f"Skipping {table_name}: {filename} not found. Run generate_data.py first.")
            continue
        df = pd.read_csv(path)
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        logger.info(f"  loaded '{table_name}' ({len(df)} rows)")

    inspector = inspect(engine)
    logger.info(f"SQLite now has {len(inspector.get_table_names())} tables.")


if __name__ == "__main__":
    load_all()
