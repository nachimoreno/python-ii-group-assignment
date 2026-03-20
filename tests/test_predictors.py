import json
import pickle

import numpy as np
import pandas as pd
import pytest

from predictors import MockPredictor, ModelPredictor, ModelUnavailableError


def make_features(log_return: float, n: int = 2) -> pd.DataFrame:
    return pd.DataFrame({"Log Return": [log_return] * n})


# ── MockPredictor ─────────────────────────────────────────────────────────────

def test_mock_predictor_buy_on_positive_log_return():
    signal = MockPredictor(neutral_band=0.001).predict(make_features(0.02))
    assert signal["action"] == "BUY"
    assert signal["movement"] == "UP"


def test_mock_predictor_sell_on_negative_log_return():
    signal = MockPredictor(neutral_band=0.001).predict(make_features(-0.015))
    assert signal["action"] == "SELL"
    assert signal["movement"] == "DOWN"


def test_mock_predictor_hold_within_neutral_band():
    signal = MockPredictor(neutral_band=0.001).predict(make_features(0.0))
    assert signal["action"] == "HOLD"
    assert signal["movement"] == "STABLE"


def test_mock_predictor_hold_on_empty_dataframe():
    signal = MockPredictor().predict(pd.DataFrame())
    assert signal["action"] == "HOLD"
    assert signal["confidence"] == 0.5


def test_mock_predictor_hold_when_log_return_column_missing():
    signal = MockPredictor().predict(pd.DataFrame({"Close": [100.0]}))
    assert signal["action"] == "HOLD"


# ── ModelPredictor ────────────────────────────────────────────────────────────

class DummyClassifier:
    def predict(self, X):
        return [1] * len(X)

    def predict_proba(self, X):
        return np.array([[0.1, 0.9]] * len(X))


class DummyRegressor:
    def predict(self, X):
        return np.array([0.025] * len(X))


def write_artifacts(path, feature_columns):
    path.mkdir(parents=True, exist_ok=True)
    (path / "classification_model.pkl").write_bytes(pickle.dumps(DummyClassifier()))
    (path / "regression_model.pkl").write_bytes(pickle.dumps(DummyRegressor()))
    (path / "metadata.json").write_text(
        json.dumps({"company": "APPLE INC", "feature_columns": feature_columns})
    )


def test_model_predictor_loads_and_returns_signal(tmp_path):
    write_artifacts(tmp_path / "Apple", ["Log Return", "Volatility 5d"])
    predictor = ModelPredictor(company="APPLE INC", model_output_dir=tmp_path)
    features = pd.DataFrame({"Log Return": [0.02], "Volatility 5d": [0.01]})
    signal = predictor.predict(features)
    assert signal["action"] in {"BUY", "SELL", "HOLD"}
    assert signal["movement"] in {"UP", "DOWN", "STABLE"}
    assert 0.0 <= signal["confidence"] <= 1.0


def test_model_predictor_raises_when_artifacts_missing(tmp_path):
    predictor = ModelPredictor(company="APPLE INC", model_output_dir=tmp_path)
    with pytest.raises(ModelUnavailableError, match="No trained model"):
        predictor.predict(pd.DataFrame({"Log Return": [0.01]}))


def test_model_predictor_raises_when_features_missing(tmp_path):
    write_artifacts(tmp_path / "Apple", ["Log Return", "Volatility 5d"])
    predictor = ModelPredictor(company="APPLE INC", model_output_dir=tmp_path)
    with pytest.raises(ModelUnavailableError, match="Missing"):
        predictor.predict(pd.DataFrame({"Log Return": [0.01]}))  # Volatility 5d absent


def test_model_predictor_raises_on_empty_dataframe(tmp_path):
    write_artifacts(tmp_path / "Apple", ["Log Return"])
    predictor = ModelPredictor(company="APPLE INC", model_output_dir=tmp_path)
    with pytest.raises(ModelUnavailableError):
        predictor.predict(pd.DataFrame())
