from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.trading_logic import load_saved_artifacts, generate_actions

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]


def _model_output_dir() -> str:
    with open("src/config.toml", "rb") as f:
        config = tomllib.load(f)
    return f"{config['GOLD_DIR']}/trained_models"


class ModelUnavailableError(RuntimeError):
    pass


class MockPredictor:
    """Simple rule-based predictor for development and testing when no model is trained.

    Uses Log Return (the first feature computed by the Part 1 ETL pipeline):
    - Log Return >  +neutral_band  ->  BUY  (UP)
    - Log Return <  -neutral_band  ->  SELL (DOWN)
    - otherwise                    ->  HOLD (STABLE)
    """

    def __init__(self, neutral_band: float = 0.0) -> None:
        self.neutral_band = neutral_band

    def predict(self, features_df: pd.DataFrame) -> dict[str, Any]:
        if features_df.empty or "Log Return" not in features_df.columns:
            return {
                "movement": "STABLE",
                "action": "HOLD",
                "confidence": 0.5,
                "reason": "No usable Log Return data found.",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        latest_return = features_df["Log Return"].dropna()
        if latest_return.empty:
            return {
                "movement": "STABLE",
                "action": "HOLD",
                "confidence": 0.5,
                "reason": "Latest log return is missing.",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        log_return = float(latest_return.iloc[-1])
        confidence = float(np.clip(abs(log_return) * 20, 0.5, 0.95))

        if log_return > self.neutral_band:
            movement, action = "UP", "BUY"
        elif log_return < -self.neutral_band:
            movement, action = "DOWN", "SELL"
        else:
            movement, action = "STABLE", "HOLD"

        return {
            "movement": movement,
            "action": action,
            "confidence": round(confidence, 2),
            "reason": f"Signal from latest Log Return = {log_return:.4f}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


class ModelPredictor:
    """Loads trained model artifacts for a company and generates a live trading signal.

    Model artifacts are produced by running:
        python manage.py --stage model_training
    """

    def __init__(self, company: str, model_output_dir: str | Path | None = None) -> None:
        self.company = company
        self.model_output_dir = Path(model_output_dir or _model_output_dir())

    def predict(self, features_df: pd.DataFrame) -> dict[str, Any]:
        if features_df.empty:
            raise ModelUnavailableError("No transformed features available for model inference.")

        try:
            classification_model, regression_model, metadata = load_saved_artifacts(
                self.model_output_dir, self.company
            )
        except FileNotFoundError as exc:
            raise ModelUnavailableError(
                f"No trained model found for '{self.company}'. "
                "Run `python manage.py --stage model_training` first."
            ) from exc

        feature_columns = metadata.get("feature_columns", [])
        missing = [col for col in feature_columns if col not in features_df.columns]
        if missing:
            raise ModelUnavailableError(
                f"Missing feature columns required by the trained model: {', '.join(missing)}"
            )

        X = features_df[feature_columns].dropna().tail(1)
        if X.empty:
            raise ModelUnavailableError("No complete feature row available for inference.")

        predicted_class = int(classification_model.predict(X)[0])
        predicted_probability = float(classification_model.predict_proba(X)[0][1])
        predicted_return = float(regression_model.predict(X)[0])

        # Use the same action logic as src/trading_logic.py
        single_row = pd.DataFrame({
            "predicted_class": [predicted_class],
            "predicted_probability": [predicted_probability],
            "predicted_return": [predicted_return],
        })
        action_row = generate_actions(single_row).iloc[0]
        action = action_row["action"]
        trade_size = int(action_row["trade_size"])

        confidence = predicted_probability if predicted_class == 1 else 1 - predicted_probability
        movement = {"BUY": "UP", "SELL": "DOWN", "HOLD": "STABLE"}[action]

        return {
            "movement": movement,
            "action": action,
            "trade_size": trade_size,
            "confidence": round(float(confidence), 2),
            "predicted_return": round(float(predicted_return), 6),
            "reason": (
                f"class={predicted_class}, confidence={confidence:.2f}, "
                f"predicted_return={predicted_return:.4f}"
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
