import pandas as pd
import pytest

from go_live_service import build_go_live_result, engineer_live_features, GoLiveResult
from predictors import MockPredictor


def make_raw_prices(n: int = 40) -> pd.DataFrame:
    """Minimal raw share-price DataFrame matching PySimFin output."""
    import numpy as np
    dates = pd.date_range("2024-01-01", periods=n, freq="B").astype(str)
    close = 100.0 + np.cumsum(np.random.default_rng(42).normal(0, 1, n))
    return pd.DataFrame({
        "Date": dates,
        "Ticker": ["AAPL"] * n,
        "Open":      close - 0.5,
        "High":      close + 1.0,
        "Low":       close - 1.0,
        "Close":     close,
        "Adj. Close": close,
        "Volume":    [1_000_000] * n,
    })


# ── engineer_live_features ────────────────────────────────────────────────────

def test_engineer_live_features_adds_log_return(monkeypatch):
    df = make_raw_prices(40)
    features = engineer_live_features(df)
    assert "Log Return" in features.columns


def test_engineer_live_features_adds_volatility_columns(monkeypatch):
    df = make_raw_prices(40)
    features = engineer_live_features(df)
    for window in (5, 10, 20):
        assert f"Volatility {window}d" in features.columns


def test_engineer_live_features_drops_nulls(monkeypatch):
    df = make_raw_prices(40)
    features = engineer_live_features(df)
    assert not features.isnull().any().any()


def test_engineer_live_features_skips_market_cap_when_no_shares_outstanding():
    df = make_raw_prices(40)
    features = engineer_live_features(df)
    assert "Log Market Cap" not in features.columns


def test_engineer_live_features_adds_market_cap_when_shares_outstanding_present():
    df = make_raw_prices(40)
    df["Shares Outstanding"] = 15_000_000_000
    features = engineer_live_features(df)
    assert "Log Market Cap" in features.columns


# ── build_go_live_result ──────────────────────────────────────────────────────

def test_build_go_live_result_returns_correct_shape():
    raw = make_raw_prices(40)
    result = build_go_live_result(raw_data=raw, predictor=MockPredictor())
    assert isinstance(result, GoLiveResult)
    assert len(result.raw_data) == 40
    assert "Log Return" in result.features.columns
    assert result.signal["action"] in {"BUY", "SELL", "HOLD"}
    assert result.signal["movement"] in {"UP", "DOWN", "STABLE"}


def test_build_go_live_result_raises_on_empty_input():
    with pytest.raises(ValueError, match="No share-price data"):
        build_go_live_result(raw_data=pd.DataFrame(), predictor=MockPredictor())


def test_build_go_live_result_raises_when_date_range_too_short():
    # Only 5 rows — not enough to compute 20-day windows, so drop_nulls empties the frame
    with pytest.raises(ValueError, match="no usable rows|Feature engineering"):
        build_go_live_result(raw_data=make_raw_prices(5), predictor=MockPredictor())
