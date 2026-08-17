"""
Energy Theft Detection Service (Isolation Forest — unsupervised).

Concept: This model is trained WITHOUT theft labels. It only learns what
"normal" consumption behavior looks like across thousands of meters, then
flags whichever meters deviate most from that norm. The ground-truth
column (is_theft_flag_ground_truth) exists ONLY so we can score how well
the unsupervised model happens to agree with known theft cases — it is
never fed into .fit(). This is a realistic simulation of anomaly detection
in production: you rarely have labels for the thing you're hunting.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from backend.utils.config import settings
from backend.utils.logger import logger

MODEL_DIR = settings.DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "theft_model.pkl"
SCALER_PATH = MODEL_DIR / "theft_scaler.pkl"

FEATURES = [
    "usage_ratio",
    "night_usage_ratio",
    "voltage_variance",
    "billing_disputes_1y",
]
# NOTE: expected/actual_monthly_kwh excluded on purpose — usage_ratio already
# captures their relationship in a scale-free way (a 10kWh house and a
# 1000kWh factory can both have usage_ratio=0.3). Feeding raw kWh values
# would bias the model toward flagging big consumers regardless of behavior.


class TheftDetectionService:
    def __init__(self):
        self.model: IsolationForest | None = None
        self.scaler: StandardScaler | None = None

    def train(self, meters: list[dict], contamination: float = 0.05) -> dict:
        df = pd.DataFrame(meters)
        X = df[FEATURES]

        # Isolation Forest is distance/split sensitive to feature scale,
        # so we standardize (mean=0, std=1) before fitting.
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        logger.info(f"Training Isolation Forest on {len(df)} meters "
                     f"(contamination={contamination})")

        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,  # our prior belief: ~5% of meters are anomalous
            random_state=settings.RANDOM_SEED,
        )
        model.fit(X_scaled)

        # decision_function: higher = more normal, lower/negative = more anomalous
        raw_scores = model.decision_function(X_scaled)
        predictions = model.predict(X_scaled)  # 1 = normal, -1 = anomaly
        predicted_anomaly = (predictions == -1).astype(int)

        self.model = model
        self.scaler = scaler
        self.save_model()

        metrics = {
            "trained_on_rows": len(df),
            "contamination": contamination,
            "flagged_anomalies": int(predicted_anomaly.sum()),
        }

        # Optional evaluation against secret ground truth, if present
        if "is_theft_flag_ground_truth" in df.columns:
            y_true = df["is_theft_flag_ground_truth"].values
            metrics["evaluation_against_ground_truth"] = {
                "note": "Ground truth used ONLY for scoring, never for training",
                "actual_theft_cases": int(y_true.sum()),
                "precision": round(precision_score(y_true, predicted_anomaly, zero_division=0), 3),
                "recall": round(recall_score(y_true, predicted_anomaly, zero_division=0), 3),
                "f1_score": round(f1_score(y_true, predicted_anomaly, zero_division=0), 3),
            }

        logger.info(f"Theft model trained. Metrics: {metrics}")
        return metrics

    def save_model(self):
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)

    def load_model(self) -> bool:
        if MODEL_PATH.exists() and SCALER_PATH.exists():
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            with open(SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)
            return True
        return False

    def score_meters(self, meters: list[dict]) -> list[dict]:
        if self.model is None and not self.load_model():
            raise RuntimeError("Theft model not trained yet. Call /api/theft/train first.")

        df = pd.DataFrame(meters)
        X_scaled = self.scaler.transform(df[FEATURES])
        raw_scores = self.model.decision_function(X_scaled)
        predictions = self.model.predict(X_scaled)

        results = []
        for i, row in df.iterrows():
            results.append({
                "meter_id": row["meter_id"],
                "consumer_id": row.get("consumer_id"),
                "region": row.get("region"),
                "anomaly_score": round(float(raw_scores[i]), 4),  # lower = more anomalous
                "is_anomalous": bool(predictions[i] == -1),
                "usage_ratio": row["usage_ratio"],
                "night_usage_ratio": row["night_usage_ratio"],
            })
        return sorted(results, key=lambda r: r["anomaly_score"])  # most anomalous first


theft_service = TheftDetectionService()
