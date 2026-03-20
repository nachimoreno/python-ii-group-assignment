import os
from datetime import date, timedelta

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

import streamlit as st
from dotenv import load_dotenv

from go_live_service import build_go_live_result
from predictors import MockPredictor, ModelPredictor, ModelUnavailableError
from simfin_wrapper import PySimFin, PySimFinAuthError, PySimFinRateLimitError, PySimFinRequestError

load_dotenv()

st.set_page_config(page_title="Go Live", layout="wide")
st.title("Go Live")
st.caption("Fetch live data from SimFin, apply ETL transformations, and view model signals.")


def resolve_api_key() -> str | None:
    try:
        if st.secrets.get("SIMFIN_API_KEY"):
            return str(st.secrets["SIMFIN_API_KEY"]).strip()
    except Exception:
        pass
    key = os.environ.get("SIMFIN_API_KEY") or os.environ.get("API_KEY")
    return key.strip() if key else None


def format_error(exc: Exception) -> str:
    if isinstance(exc, PySimFinAuthError):
        return "Authentication failed. Check your SIMFIN_API_KEY."
    if isinstance(exc, PySimFinRateLimitError):
        return "Rate limit reached. Wait a moment and try again."
    if isinstance(exc, PySimFinRequestError):
        return f"SimFin request failed: {exc}"
    if isinstance(exc, ModelUnavailableError):
        return str(exc)
    return f"Unexpected error: {exc}"


# ── Load tickers ──────────────────────────────────────────────────────────────
try:
    with open("src/config.toml", "rb") as f:
        universe = tomllib.load(f).get("tickers", [])
except Exception as exc:
    st.error(f"Could not load tickers from src/config.toml: {exc}")
    st.stop()

# ── API key ───────────────────────────────────────────────────────────────────
api_key = resolve_api_key()
if not api_key:
    st.error("Missing API key. Add `SIMFIN_API_KEY` to your `.env` file or Streamlit Cloud secrets.")
    st.stop()

# ── Controls ──────────────────────────────────────────────────────────────────
default_end = date.today()
default_start = default_end - timedelta(days=120)

selected_ticker = st.selectbox("Ticker", options=universe)
start_date = st.date_input("Start date", value=default_start, max_value=default_end)
end_date = st.date_input("End date", value=default_end, min_value=start_date, max_value=default_end)
use_mock = st.checkbox("Use MockPredictor (no trained model needed)", value=False)

# ── Run ───────────────────────────────────────────────────────────────────────
if st.button("Run Live Analysis", type="primary"):
    predictor = MockPredictor() if use_mock else ModelPredictor(company=selected_ticker)

    try:
        with st.spinner("Fetching market data from SimFin..."):
            client = PySimFin(api_key=api_key)
            raw_data = client.get_share_prices(
                ticker=selected_ticker,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
            )
        result = build_go_live_result(raw_data=raw_data, predictor=predictor)
    except Exception as exc:
        st.error(format_error(exc))
        st.stop()

    # ── Signal ────────────────────────────────────────────────────────────────
    st.success("Analysis complete.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Movement", result.signal.get("movement", "—"))
    col2.metric("Action", result.signal.get("action", "—"))
    col3.metric("Confidence", result.signal.get("confidence", "—"))
    st.caption(str(result.signal.get("reason", "")))

    # ── Data tables ───────────────────────────────────────────────────────────
    st.subheader("Raw Share Prices")
    st.dataframe(result.raw_data.tail(20), use_container_width=True)

    st.subheader("Engineered Features")
    feature_cols = [c for c in result.features.columns if c not in ("Date", "Ticker")]
    st.dataframe(result.features[["Date"] + feature_cols].tail(20), use_container_width=True)
