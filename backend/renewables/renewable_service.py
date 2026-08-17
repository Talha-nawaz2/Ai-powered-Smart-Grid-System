"""
Renewable Intelligence Service.

Two ML regressors (solar, wind generation) + one rule-based battery
dispatch engine.

Concept: not every problem needs ML. Solar/wind generation depends on
weather in a noisy, nonlinear way that's genuinely worth learning from
data (RandomForestRegressor). Battery dispatch, on the other hand, follows
a known physical/economic rule: charge on surplus, discharge on deficit,
respect capacity limits. Encoding that as an ML model would just make it
harder to explain, slower to run, and no more accurate than the direct
rule. Recognizing this distinction is a real engineering judgment call,
not a shortcut.
"""

import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from backend.utils.config import settings
from backend.utils.logger import logger

MODEL_DIR = settings.DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SOLAR_MODEL_PATH = MODEL_DIR / "solar_model.pkl"
WIND_MODEL_PATH = MODEL_DIR / "wind_model.pkl"

SOLAR_FEATURES = ["solar_irradiance_wm2", "temperature_c", "capacity_mw"]
WIND_FEATURES = ["wind_speed_kmh", "capacity_mw"]
TARGET = "generation_mw"


class RenewableService:
    def __init__(self):
        self.solar_model: RandomForestRegressor | None = None
        self.wind_model: RandomForestRegressor | None = None

    # ---------- Training ----------
    def train_solar(self, data: list[dict]) -> dict:
        df = pd.DataFrame(data)
        X, y = df[SOLAR_FEATURES], df[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=settings.RANDOM_SEED
        )
        model = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=settings.RANDOM_SEED)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = {
            "trained_on_rows": len(df),
            "mae_mw": round(mean_absolute_error(y_test, preds), 3),
            "r2_score": round(r2_score(y_test, preds), 3),
            "feature_importance": dict(zip(SOLAR_FEATURES, model.feature_importances_.round(3).tolist())),
        }
        self.solar_model = model
        with open(SOLAR_MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Solar model trained: {metrics}")
        return metrics

    def train_wind(self, data: list[dict]) -> dict:
        df = pd.DataFrame(data)
        X, y = df[WIND_FEATURES], df[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=settings.RANDOM_SEED
        )
        model = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=settings.RANDOM_SEED)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = {
            "trained_on_rows": len(df),
            "mae_mw": round(mean_absolute_error(y_test, preds), 3),
            "r2_score": round(r2_score(y_test, preds), 3),
            "feature_importance": dict(zip(WIND_FEATURES, model.feature_importances_.round(3).tolist())),
        }
        self.wind_model = model
        with open(WIND_MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Wind model trained: {metrics}")
        return metrics

    # ---------- Loading ----------
    def _load_solar(self):
        if self.solar_model is None:
            if not SOLAR_MODEL_PATH.exists():
                raise RuntimeError("Solar model not trained yet. Call /api/renewables/train first.")
            with open(SOLAR_MODEL_PATH, "rb") as f:
                self.solar_model = pickle.load(f)

    def _load_wind(self):
        if self.wind_model is None:
            if not WIND_MODEL_PATH.exists():
                raise RuntimeError("Wind model not trained yet. Call /api/renewables/train first.")
            with open(WIND_MODEL_PATH, "rb") as f:
                self.wind_model = pickle.load(f)

    # ---------- Prediction ----------
    def predict_solar(self, solar_irradiance_wm2: float, temperature_c: float, capacity_mw: float) -> float:
        self._load_solar()
        row = pd.DataFrame([{
            "solar_irradiance_wm2": solar_irradiance_wm2,
            "temperature_c": temperature_c,
            "capacity_mw": capacity_mw,
        }])
        return round(float(self.solar_model.predict(row)[0]), 2)

    def predict_wind(self, wind_speed_kmh: float, capacity_mw: float) -> float:
        self._load_wind()
        row = pd.DataFrame([{"wind_speed_kmh": wind_speed_kmh, "capacity_mw": capacity_mw}])
        return round(float(self.wind_model.predict(row)[0]), 2)

    # ---------- Battery Recommendation (rule-based) ----------
    @staticmethod
    def recommend_battery_action(
        current_generation_mw: float,
        current_demand_mw: float,
        battery: dict,
    ) -> dict:
        """
        Rule-based dispatch logic:
          surplus = generation - demand
          surplus > 0  -> charge battery (if not full)
          surplus < 0  -> discharge battery to cover deficit (if not empty)
          surplus ~= 0 -> hold
        """
        surplus_mw = round(current_generation_mw - current_demand_mw, 2)
        capacity_mwh = battery["capacity_mwh"]
        charge_pct = battery["current_charge_pct"]
        charge_mwh = capacity_mwh * (charge_pct / 100)
        headroom_mwh = capacity_mwh - charge_mwh

        if surplus_mw > 1:
            chargeable_mw = min(surplus_mw, headroom_mwh)  # can't charge faster than remaining headroom (1hr step)
            action = "CHARGE" if headroom_mwh > 0.1 else "HOLD (battery full)"
            amount = round(chargeable_mw, 2) if headroom_mwh > 0.1 else 0.0
        elif surplus_mw < -1:
            dischargeable_mw = min(abs(surplus_mw), charge_mwh)
            action = "DISCHARGE" if charge_mwh > 0.1 else "HOLD (battery empty)"
            amount = round(dischargeable_mw, 2) if charge_mwh > 0.1 else 0.0
        else:
            action = "HOLD"
            amount = 0.0

        return {
            "battery_id": battery["battery_id"],
            "surplus_mw": surplus_mw,
            "current_charge_pct": charge_pct,
            "recommended_action": action,
            "recommended_amount_mw": amount,
        }


renewable_service = RenewableService()
