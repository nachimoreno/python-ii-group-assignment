from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd


class Predictor(Protocol):
    def predict(self, features_df: pd.DataFrame) -> dict[str, Any]:
        pass


class ModelUnavailableError(RuntimeError):
    pass


class MockPredictor:
    """
    Very simple signal:
    - Return_1d > +threshold -> BUY
    - Return_1d < -threshold -> SELL
    - otherwise -> HOLD
    """

    def __init__(self, neutral_band: float = 0.0) -> None:
        self.neutral_band = neutral_band

    def predict(self, features_df: pd.DataFrame) -> dict[str, Any]:
        print("[Predictor] Generating mock signal...")
        if features_df.empty or "Return_1d" not in features_df.columns:
            return {
                "movement": "STABLE",
                "action": "HOLD",
                "confidence": 0.5,
                "reason": "No usable Return_1d data found.",
                "generated_at": datetime.now(UTC).isoformat(),
            }

        latest_return = features_df["Return_1d"].dropna()
        if latest_return.empty:
            return {
                "movement": "STABLE",
                "action": "HOLD",
                "confidence": 0.5,
                "reason": "Latest return is missing.",
                "generated_at": datetime.now(UTC).isoformat(),
            }

        return_1d = float(latest_return.iloc[-1])
        confidence = float(np.clip(abs(return_1d) * 20, 0.5, 0.95))
        if return_1d > self.neutral_band:
            movement = "UP"
            action = "BUY"
        elif return_1d < -self.neutral_band:
            movement = "DOWN"
            action = "SELL"
        else:
            movement = "STABLE"
            action = "HOLD"

        return {
            "movement": movement,
            "action": action,
            "confidence": round(confidence, 2),
            "reason": f"Signal from latest Return_1d = {return_1d:.4f}",
            "generated_at": datetime.now(UTC).isoformat(),
        }


class ModelPredictor:
    def __init__(self, model_path: str = "models/model.joblib") -> None:
        self.model_path = Path(model_path)

    def predict(self, features_df: pd.DataFrame) -> dict[str, Any]:
        raise ModelUnavailableError(
            "ModelPredictor is a stub. Add a trained model artifact and implement inference logic."
        )
