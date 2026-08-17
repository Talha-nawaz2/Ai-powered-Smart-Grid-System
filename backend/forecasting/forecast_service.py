"""
Demand Forecasting Service.

Primary engine: Prophet (time-series decomposition with daily/weekly seasonality).
Fallback engine: a hand-built trend + seasonal decomposition model, used
automatically if Prophet's compiled CmdStan backend isn't available on this
machine (a common, environment-specific installation issue).

Concept: this is a real resilience pattern used in production systems -
if a heavyweight dependency with an external compiled component fails to
initialize, the system degrades to a simpler but still statistically
sound method rather than taking the whole feature down. The fallback uses
only pandas/numpy (already required elsewhere), so it can never itself
fail to install.

Fallback method, explained: additive decomposition.
  1. Fit a straight line (trend) through demand over time.
  2. Subtract that trend from every observation to get "residuals" -
     what's left after removing the trend is the repeating daily/weekly
     pattern.
  3. Average the residuals by (weekday, hour) to build a seasonal lookup
     table - "how far above/below trend is a typical Tuesday 6pm?"
  4. To forecast a future timestamp: trend value at that point in time,
     plus the seasonal offset for its (weekday, hour).
This is the same additive idea Prophet uses internally (trend + seasonality),
just without Prophet's more sophisticated changepoint detection.
"""

import pickle
import numpy as np
import pandas as pd
from prophet import Prophet

from backend.utils.logger import logger
from backend.utils.config import settings

MODEL_DIR = settings.DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "demand_forecast_model.pkl"
FALLBACK_PATH = MODEL_DIR / "demand_forecast_fallback.pkl"
MODE_PATH = MODEL_DIR / "demand_forecast_mode.txt"


class DemandForecastService:
    def __init__(self):
        self.model: Prophet | None = None       # used when mode == "prophet"
        self.fallback: dict | None = None         # used when mode == "seasonal_fallback"
        self.mode: str | None = None

    def _prepare_training_frame(self, history: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(history)
        df = df.rename(columns={"timestamp": "ds", "demand_mw": "y"})
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds").reset_index(drop=True)
        return df[["ds", "y"]]

    def train(self, history: list[dict]) -> dict:
        df = self._prepare_training_frame(history)
        if len(df) < 48:
            raise ValueError("Need at least 48 hours of history to train a meaningful model")

        try:
            self._train_prophet(df)
            self.mode = "prophet"
        except Exception as e:
            logger.warning(f"Prophet training failed ({type(e).__name__}: {e}); "
                             f"falling back to seasonal decomposition forecasting.")
            self._train_seasonal_fallback(df)
            self.mode = "seasonal_fallback"

        MODE_PATH.write_text(self.mode)
        logger.info(f"Forecast training complete using mode='{self.mode}'.")
        return {"mode": self.mode, "trained_on_rows": len(df),
                "start": str(df["ds"].min()), "end": str(df["ds"].max())}

    # ---------- Primary: Prophet ----------
    def _train_prophet(self, df: pd.DataFrame):
        logger.info(f"Training Prophet on {len(df)} hourly demand points "
                     f"({df['ds'].min()} -> {df['ds'].max()})")
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
        )
        model.fit(df)  # this is where the CmdStan AttributeError surfaces, if it's going to
        self.model = model
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

    # ---------- Fallback: trend + seasonal decomposition ----------
    def _train_seasonal_fallback(self, df: pd.DataFrame):
        ds_min = df["ds"].min()
        day_idx = (df["ds"] - ds_min).dt.total_seconds() / 86400.0

        slope, intercept = np.polyfit(day_idx, df["y"], 1)
        trend_pred = intercept + slope * day_idx
        residual = df["y"] - trend_pred

        seasonal_df = pd.DataFrame({
            "weekday": df["ds"].dt.weekday, "hour": df["ds"].dt.hour, "residual": residual,
        })
        seasonal_table = seasonal_df.groupby(["weekday", "hour"])["residual"].mean().to_dict()

        self.fallback = {
            "slope": float(slope), "intercept": float(intercept), "ds_min": ds_min,
            "ds_max": df["ds"].max(),
            "seasonal_table": seasonal_table, "residual_std": float(residual.std()),
        }
        with open(FALLBACK_PATH, "wb") as f:
            pickle.dump(self.fallback, f)
        logger.info(f"Seasonal fallback trained: slope={slope:.4f} MW/day, "
                     f"residual_std={residual.std():.1f} MW")

    def load_model(self) -> bool:
        if MODE_PATH.exists():
            self.mode = MODE_PATH.read_text().strip()
            if self.mode == "prophet" and MODEL_PATH.exists():
                with open(MODEL_PATH, "rb") as f:
                    self.model = pickle.load(f)
                return True
            if self.mode == "seasonal_fallback" and FALLBACK_PATH.exists():
                with open(FALLBACK_PATH, "rb") as f:
                    self.fallback = pickle.load(f)
                return True
        return False

    def forecast(self, periods: int = 24, freq: str = "h") -> list[dict]:
        if self.mode is None and not self.load_model():
            raise RuntimeError("Model not trained yet. Call /api/forecast/train first.")

        if self.mode == "prophet":
            return self._forecast_prophet(periods, freq)
        return self._forecast_seasonal_fallback_predict(periods, freq)

    def _forecast_prophet(self, periods: int, freq: str) -> list[dict]:
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast_df = self.model.predict(future)
        future_only = forecast_df.tail(periods)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        return [
            {"timestamp": str(row["ds"]), "predicted_demand_mw": round(row["yhat"], 1),
             "lower_bound_mw": round(row["yhat_lower"], 1), "upper_bound_mw": round(row["yhat_upper"], 1)}
            for _, row in future_only.iterrows()
        ]

    def _forecast_seasonal_fallback_predict(self, periods: int, freq: str) -> list[dict]:
        fb = self.fallback
        offset = pd.tseries.frequencies.to_offset(freq)
        overall_mean_residual = float(np.mean(list(fb["seasonal_table"].values())))
        ds_min = pd.Timestamp(fb["ds_min"])
        last_actual = pd.Timestamp(fb["ds_max"])  # forecast continues from here, NOT from ds_min

        results = []
        for i in range(1, periods + 1):
            ts = last_actual + i * offset
            day_idx = (ts - ds_min).total_seconds() / 86400.0  # trend was fit relative to ds_min
            trend_val = fb["intercept"] + fb["slope"] * day_idx
            seasonal_val = fb["seasonal_table"].get((ts.weekday(), ts.hour), overall_mean_residual)
            pred = trend_val + seasonal_val
            band = 1.5 * fb["residual_std"]
            results.append({
                "timestamp": str(ts),
                "predicted_demand_mw": round(pred, 1),
                "lower_bound_mw": round(pred - band, 1),
                "upper_bound_mw": round(pred + band, 1),
            })
        return results


# Module-level singleton so the trained model persists across requests within one server run
forecast_service = DemandForecastService()
