"""
Synthetic Smart Grid Data Generator
====================================

Concept for you (Talha): Real utility companies never hand out SCADA data
for student projects, so we SIMULATE a realistic grid using statistical
distributions (normal, uniform) + light domain logic (e.g. solar output
depends on hour-of-day + weather, not pure randomness). This is how most
"AI for X" portfolio projects are built when no real dataset exists.

Run:  python -m backend.data.generate_data
Output: CSVs in backend/data/sample_data/
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

from backend.utils.config import settings
from backend.utils.logger import logger

np.random.seed(settings.RANDOM_SEED)
fake = Faker()
Faker.seed(settings.RANDOM_SEED)

OUT = settings.SAMPLE_DATA_DIR
OUT.mkdir(parents=True, exist_ok=True)

REGIONS = ["Islamabad", "Rawalpindi", "Lahore", "Karachi", "Peshawar", "Multan", "Faisalabad"]


def gen_substations(n=50):
    df = pd.DataFrame({
        "substation_id": [f"SUB-{i:04d}" for i in range(n)],
        "region": np.random.choice(REGIONS, n),
        "capacity_mva": np.round(np.random.uniform(20, 200, n), 1),
        "num_transformers": np.random.randint(3, 15, n),
        "lat": np.round(np.random.uniform(24.0, 34.0, n), 4),
        "lon": np.round(np.random.uniform(66.0, 74.5, n), 4),
        "commissioned_year": np.random.randint(1985, 2023, n),
    })
    df.to_csv(OUT / "substations.csv", index=False)
    logger.info(f"substations.csv -> {len(df)} rows")
    return df


def gen_power_plants(n=30):
    types = ["Thermal", "Hydro", "Nuclear", "Gas", "Coal"]
    df = pd.DataFrame({
        "plant_id": [f"PLT-{i:04d}" for i in range(n)],
        "type": np.random.choice(types, n, p=[0.35, 0.25, 0.1, 0.2, 0.1]),
        "capacity_mw": np.round(np.random.uniform(100, 1500, n), 1),
        "region": np.random.choice(REGIONS, n),
        "commissioned_year": np.random.randint(1970, 2022, n),
        "efficiency_pct": np.round(np.random.uniform(35, 55, n), 1),
    })
    df.to_csv(OUT / "power_plants.csv", index=False)
    logger.info(f"power_plants.csv -> {len(df)} rows")
    return df


def gen_solar_farms(n=25):
    df = pd.DataFrame({
        "farm_id": [f"SOL-{i:04d}" for i in range(n)],
        "capacity_mw": np.round(np.random.uniform(5, 100, n), 1),
        "region": np.random.choice(REGIONS, n),
        "panel_type": np.random.choice(["Monocrystalline", "Polycrystalline", "Thin-Film"], n),
        "lat": np.round(np.random.uniform(24.0, 34.0, n), 4),
        "lon": np.round(np.random.uniform(66.0, 74.5, n), 4),
    })
    df.to_csv(OUT / "solar_farms.csv", index=False)
    logger.info(f"solar_farms.csv -> {len(df)} rows")
    return df


def gen_wind_farms(n=15):
    df = pd.DataFrame({
        "farm_id": [f"WND-{i:04d}" for i in range(n)],
        "capacity_mw": np.round(np.random.uniform(10, 150, n), 1),
        "region": np.random.choice(REGIONS, n),
        "turbine_count": np.random.randint(5, 60, n),
        "lat": np.round(np.random.uniform(24.0, 34.0, n), 4),
        "lon": np.round(np.random.uniform(66.0, 74.5, n), 4),
    })
    df.to_csv(OUT / "wind_farms.csv", index=False)
    logger.info(f"wind_farms.csv -> {len(df)} rows")
    return df


def gen_battery_storage(substation_ids, n=40):
    df = pd.DataFrame({
        "battery_id": [f"BAT-{i:04d}" for i in range(n)],
        "substation_id": np.random.choice(substation_ids, n),
        "capacity_mwh": np.round(np.random.uniform(1, 50, n), 1),
        "current_charge_pct": np.round(np.random.uniform(10, 100, n), 1),
        "cycle_count": np.random.randint(50, 3000, n),
        "health_pct": np.round(np.random.uniform(70, 100, n), 1),
    })
    df.to_csv(OUT / "battery_storage.csv", index=False)
    logger.info(f"battery_storage.csv -> {len(df)} rows")
    return df


def gen_transformers(substation_ids, n=500):
    age = np.random.randint(0, 40, n)
    load_factor = np.clip(np.random.normal(0.65, 0.15, n), 0.1, 1.2)
    temperature = np.clip(np.random.normal(55, 12, n) + age * 0.3, 25, 110)
    vibration = np.clip(np.random.normal(2.0, 0.8, n) + age * 0.02, 0.2, 8)
    oil_quality = np.clip(np.random.normal(80, 15, n) - age * 0.6, 5, 100)
    past_failures = np.random.poisson(age / 15 + 0.2, n)

    # Health score: enterprise-style composite index (domain logic, not random label)
    health_score = np.clip(
        100 - (age * 0.9) - (temperature - 55) * 0.5 - (vibration * 4) - (past_failures * 6) + (oil_quality * 0.2),
        0, 100
    )
    # Failure probability rises as health drops (used later as ML training signal source)
    failure_risk_label = (health_score < 40).astype(int)

    df = pd.DataFrame({
        "transformer_id": [f"TRF-{i:05d}" for i in range(n)],
        "substation_id": np.random.choice(substation_ids, n),
        "capacity_kva": np.random.choice([100, 250, 500, 1000, 1500, 2000], n),
        "age_years": age,
        "load_factor": np.round(load_factor, 3),
        "temperature_c": np.round(temperature, 1),
        "vibration_mm_s": np.round(vibration, 2),
        "oil_quality_index": np.round(oil_quality, 1),
        "past_failures_5y": past_failures,
        "health_score": np.round(health_score, 1),
        "failed_last_year": failure_risk_label,
    })
    df.to_csv(OUT / "transformers.csv", index=False)
    logger.info(f"transformers.csv -> {len(df)} rows")
    return df


def gen_consumers_and_meters(transformer_ids, n=3000):
    types = np.random.choice(["Residential", "Commercial", "Industrial"], n, p=[0.75, 0.18, 0.07])
    base_load = np.select(
        [types == "Residential", types == "Commercial", types == "Industrial"],
        [np.random.normal(6, 2, n), np.random.normal(30, 10, n), np.random.normal(150, 60, n)]
    )
    base_load = np.clip(base_load, 0.5, None)

    consumers = pd.DataFrame({
        "consumer_id": [f"CUS-{i:06d}" for i in range(n)],
        "name": [fake.name() for _ in range(n)],
        "region": np.random.choice(REGIONS, n),
        "consumer_type": types,
        "connection_date": [fake.date_between(start_date="-15y", end_date="-30d") for _ in range(n)],
        "avg_monthly_kwh": np.round(base_load * 30, 1),
    })
    consumers.to_csv(OUT / "consumers.csv", index=False)
    logger.info(f"consumers.csv -> {len(consumers)} rows")

    meters = pd.DataFrame({
        "meter_id": [f"MTR-{i:06d}" for i in range(n)],
        "consumer_id": consumers["consumer_id"],
        "transformer_id": np.random.choice(transformer_ids, n),
        "meter_type": np.random.choice(["Smart", "Smart-Prepaid"], n, p=[0.7, 0.3]),
        "install_date": [fake.date_between(start_date="-8y", end_date="-30d") for _ in range(n)],
    })
    meters.to_csv(OUT / "smart_meters.csv", index=False)
    logger.info(f"smart_meters.csv -> {len(meters)} rows")
    return consumers, meters


def gen_meter_consumption_profile(meters_df, consumers_df, n_sample=None):
    """
    Cross-sectional per-meter feature table used for THEFT DETECTION.
    Real theft detection uses aggregated behavioral features per meter
    (mean, std, night-usage ratio) rather than raw time series - this
    keeps the dataset small (<=5000 rows) while remaining realistic.
    """
    merged = meters_df.merge(consumers_df, on="consumer_id")
    n = min(len(merged), settings.MAX_ROWS_PER_DATASET) if n_sample is None else n_sample
    sample = merged.sample(n=n, random_state=settings.RANDOM_SEED).reset_index(drop=True)

    expected_kwh = sample["avg_monthly_kwh"].values
    actual_kwh = np.clip(expected_kwh * np.random.normal(1.0, 0.12, n), 0.5, None)

    # Inject ~4% theft cases: actual usage much lower than expected (classic theft signature)
    theft_mask = np.random.rand(n) < 0.04
    actual_kwh[theft_mask] = expected_kwh[theft_mask] * np.random.uniform(0.15, 0.45, theft_mask.sum())

    night_usage_ratio = np.clip(np.random.normal(0.3, 0.08, n), 0.05, 0.7)
    night_usage_ratio[theft_mask] = np.random.uniform(0.55, 0.9, theft_mask.sum())  # theft often spikes at night (bypass)

    voltage_variance = np.clip(np.random.normal(2.0, 1.0, n), 0.1, None)
    voltage_variance[theft_mask] += np.random.uniform(2, 6, theft_mask.sum())  # tampering causes voltage irregularity

    df = pd.DataFrame({
        "meter_id": sample["meter_id"],
        "consumer_id": sample["consumer_id"],
        "region": sample["region"],
        "consumer_type": sample["consumer_type"],
        "expected_monthly_kwh": np.round(expected_kwh, 1),
        "actual_monthly_kwh": np.round(actual_kwh, 1),
        "usage_ratio": np.round(actual_kwh / expected_kwh, 3),
        "night_usage_ratio": np.round(night_usage_ratio, 3),
        "voltage_variance": np.round(voltage_variance, 2),
        "billing_disputes_1y": np.random.poisson(0.3, n) + theft_mask.astype(int) * np.random.randint(1, 3, n),
        "is_theft_flag_ground_truth": theft_mask.astype(int),  # kept for evaluation only, NOT used as a model feature
    })
    df.to_csv(OUT / "meter_consumption_profile.csv", index=False)
    logger.info(f"meter_consumption_profile.csv -> {len(df)} rows ({theft_mask.sum()} synthetic theft cases)")
    return df


def gen_weather(days=90, regions=None):
    regions = regions or REGIONS
    rows = []
    start = datetime.now() - timedelta(days=days)
    timestamps = [start + timedelta(hours=h) for h in range(days * 24)]
    # cap total rows at MAX_ROWS_PER_DATASET by limiting regions*hours
    max_hours = settings.MAX_ROWS_PER_DATASET // len(regions)
    timestamps = timestamps[-max_hours:]

    for region in regions:
        for ts in timestamps:
            hour = ts.hour
            day_of_year = ts.timetuple().tm_yday
            seasonal = 10 * np.sin((day_of_year / 365) * 2 * np.pi)
            diurnal = 6 * np.sin((hour - 6) / 24 * 2 * np.pi)
            temp = 25 + seasonal + diurnal + np.random.normal(0, 2)
            solar_irr = max(0, np.sin((hour - 6) / 12 * np.pi)) * np.random.uniform(700, 1000) if 6 <= hour <= 18 else 0
            rows.append({
                "timestamp": ts,
                "region": region,
                "temperature_c": round(temp, 1),
                "humidity_pct": round(np.clip(np.random.normal(55, 15), 10, 100), 1),
                "wind_speed_kmh": round(np.clip(np.random.weibull(2) * 15, 0, 80), 1),
                "solar_irradiance_wm2": round(solar_irr, 1),
                "precipitation_mm": round(max(0, np.random.exponential(0.5) - 0.3), 2),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "weather.csv", index=False)
    logger.info(f"weather.csv -> {len(df)} rows")
    return df


def gen_grid_demand_hourly(days=90):
    """
    Aggregate grid-wide demand time series -> used by Prophet for forecasting.
    Domain logic: daily seasonality (evening peak) + weekly seasonality (lower on
    weekends) + slow upward trend (grid growth) + noise.
    """
    start = datetime.now() - timedelta(days=days)
    n_hours = min(days * 24, settings.MAX_ROWS_PER_DATASET)
    timestamps = [start + timedelta(hours=h) for h in range(n_hours)]

    demand = []
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        weekday = ts.weekday()
        trend = 600 + i * 0.01  # slow growth
        daily_pattern = 150 * np.sin((hour - 15) / 24 * 2 * np.pi) + 100 if 7 <= hour <= 22 else -50
        weekend_dip = -40 if weekday >= 5 else 0
        noise = np.random.normal(0, 15)
        value = max(300, trend + daily_pattern + weekend_dip + noise)
        demand.append(round(value, 1))

    df = pd.DataFrame({"timestamp": timestamps, "demand_mw": demand})
    df.to_csv(OUT / "grid_demand_hourly.csv", index=False)
    logger.info(f"grid_demand_hourly.csv -> {len(df)} rows")
    return df


def gen_solar_wind_generation(solar_df, wind_df, weather_df):
    """Generation logs tied to weather (solar_irradiance / wind_speed) -> realistic correlation for ML."""
    w = weather_df.groupby("region").tail(200).reset_index(drop=True)  # cap size

    solar_rows = []
    for _, farm in solar_df.iterrows():
        sample_w = w[w["region"] == farm["region"]]
        if sample_w.empty:
            continue
        sample_w = sample_w.sample(min(20, len(sample_w)), random_state=1)
        for _, wr in sample_w.iterrows():
            efficiency = np.random.uniform(0.16, 0.22)
            gen_mw = round(farm["capacity_mw"] * (wr["solar_irradiance_wm2"] / 1000) * efficiency, 2)
            solar_rows.append({"farm_id": farm["farm_id"], "timestamp": wr["timestamp"],
                                "region": farm["region"], "generation_mw": max(0, gen_mw)})

    wind_rows = []
    for _, farm in wind_df.iterrows():
        sample_w = w[w["region"] == farm["region"]]
        if sample_w.empty:
            continue
        sample_w = sample_w.sample(min(20, len(sample_w)), random_state=2)
        for _, wr in sample_w.iterrows():
            speed = wr["wind_speed_kmh"]
            # simplified turbine power curve
            factor = 0 if speed < 10 else min(1.0, (speed - 10) / 30)
            gen_mw = round(farm["capacity_mw"] * factor * np.random.uniform(0.85, 1.0), 2)
            wind_rows.append({"farm_id": farm["farm_id"], "timestamp": wr["timestamp"],
                               "region": farm["region"], "generation_mw": max(0, gen_mw)})

    pd.DataFrame(solar_rows).to_csv(OUT / "solar_generation_log.csv", index=False)
    pd.DataFrame(wind_rows).to_csv(OUT / "wind_generation_log.csv", index=False)
    logger.info(f"solar_generation_log.csv -> {len(solar_rows)} rows")
    logger.info(f"wind_generation_log.csv -> {len(wind_rows)} rows")


def gen_transmission_lines(substation_ids, n=200):
    df = pd.DataFrame({
        "line_id": [f"LN-{i:04d}" for i in range(n)],
        "from_substation": np.random.choice(substation_ids, n),
        "to_substation": np.random.choice(substation_ids, n),
        "voltage_kv": np.random.choice([132, 220, 500], n),
        "length_km": np.round(np.random.uniform(2, 300, n), 1),
        "capacity_mw": np.round(np.random.uniform(50, 800, n), 1),
        "current_load_pct": np.round(np.random.uniform(20, 98, n), 1),
    })
    df = df[df["from_substation"] != df["to_substation"]]
    df.to_csv(OUT / "transmission_lines.csv", index=False)
    logger.info(f"transmission_lines.csv -> {len(df)} rows")
    return df


def run_all():
    logger.info("=== Generating Smart Grid synthetic datasets ===")
    substations = gen_substations()
    gen_power_plants()
    solar_farms = gen_solar_farms()
    wind_farms = gen_wind_farms()
    gen_battery_storage(substations["substation_id"].tolist())
    transformers = gen_transformers(substations["substation_id"].tolist())
    consumers, meters = gen_consumers_and_meters(transformers["transformer_id"].tolist())
    gen_meter_consumption_profile(meters, consumers)
    weather = gen_weather()
    gen_grid_demand_hourly()
    gen_solar_wind_generation(solar_farms, wind_farms, weather)
    gen_transmission_lines(substations["substation_id"].tolist())
    logger.info("=== All datasets generated successfully in backend/data/sample_data/ ===")


if __name__ == "__main__":
    run_all()
