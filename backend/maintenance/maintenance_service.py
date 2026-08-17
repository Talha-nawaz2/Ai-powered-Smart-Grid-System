"""
Predictive Maintenance Service (XGBoost).

Concept: Classification models output a PROBABILITY (0-1), not just a
yes/no. We keep the raw probability all the way through the API so the
Service/Route layers can apply business thresholds (e.g. "alert if >80%")
without retraining anything — same separation-of-concerns idea as Session 2.

IMPORTANT: 'health_score' is deliberately EXCLUDED from features because
it was used to derive the training label itself (see Session 1's
gen_transformers()). Including it would be data leakage — the model
would just learn to echo the label instead of learning real patterns
from age/temperature/vibration/oil_quality. This mirrors a very common
real-world mistake: accidentally training on a feature that encodes
the answer.
"""

import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from backend.utils.config import settings
from backend.utils.logger import logger

MODEL_DIR = settings.DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "maintenance_model.pkl"

# NOTE: health_score intentionally excluded (data leakage — see module docstring)
FEATURES = [
    "capacity_kva",
    "age_years",
    "load_factor",
    "temperature_c",
    "vibration_mm_s",
    "oil_quality_index",
    "past_failures_5y",
]
TARGET = "failed_last_year"


class MaintenanceService:
    def __init__(self):
        self.model: XGBClassifier | None = None
        self.metrics: dict = {}

    def train(self, transformers: list[dict]) -> dict:
        df = pd.DataFrame(transformers)
        X = df[FEATURES]
        y = df[TARGET]

        # stratify=y keeps the same failure ratio in train and test splits —
        # important here since failures are a minority class (~imbalanced)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=settings.RANDOM_SEED, stratify=y
        )

        logger.info(f"Training XGBoost on {len(X_train)} transformers "
                     f"({y_train.sum()} failures in train set)")

        model = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            eval_metric="logloss",
            random_state=settings.RANDOM_SEED,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        self.metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 3),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 3),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 3),
            "test_set_size": len(y_test),
            "test_set_failures": int(y_test.sum()),
        }

        importances = dict(zip(FEATURES, model.feature_importances_.round(3).tolist()))
        self.metrics["feature_importance"] = dict(
            sorted(importances.items(), key=lambda x: x[1], reverse=True)
        )

        self.model = model
        self.save_model()
        logger.info(f"Maintenance model trained. Metrics: {self.metrics}")
        return self.metrics

    def save_model(self):
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self) -> bool:
        if MODEL_PATH.exists():
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            return True
        return False

    def predict_one(self, transformer: dict) -> dict:
        if self.model is None and not self.load_model():
            raise RuntimeError("Maintenance model not trained yet. Call /api/maintenance/train first.")

        row = pd.DataFrame([{f: transformer[f] for f in FEATURES}])
        proba = float(self.model.predict_proba(row)[0, 1])
        return {
            "transformer_id": transformer.get("transformer_id"),
            "failure_probability": round(proba, 4),
            "risk_level": self._risk_level(proba),
        }

    def predict_many(self, transformers: list[dict]) -> list[dict]:
        if self.model is None and not self.load_model():
            raise RuntimeError("Maintenance model not trained yet. Call /api/maintenance/train first.")

        df = pd.DataFrame(transformers)
        probs = self.model.predict_proba(df[FEATURES])[:, 1]
        results = []
        for t, p in zip(transformers, probs):
            results.append({
                "transformer_id": t["transformer_id"],
                "failure_probability": round(float(p), 4),
                "risk_level": self._risk_level(float(p)),
            })
        return sorted(results, key=lambda r: r["failure_probability"], reverse=True)

    @staticmethod
    def _risk_level(proba: float) -> str:
        if proba >= 0.80:
            return "CRITICAL"
        if proba >= 0.5:
            return "HIGH"
        if proba >= 0.25:
            return "MODERATE"
        return "LOW"


maintenance_service = MaintenanceService()
