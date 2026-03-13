import os
from datetime import date, timedelta

import streamlit as st
from dotenv import load_dotenv

from app_helpers import format_client_error, resolve_simfin_api_key
from go_live_service import build_go_live_result
from predictors import MockPredictor
from project_config import load_company_universe
from simfin_wrapper import PySimFin

load_dotenv()

st.set_page_config(page_title="Go Live", layout="wide")
st.title("Go Live")
st.caption("Fetch data from SimFin, apply ETL transformations, and view model signals.")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_share_prices_cached(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: str,
    base_url: str,
):
    client = PySimFin(api_key=api_key, base_url=base_url)
    return client.get_share_prices(ticker=ticker, start=start_date, end=end_date)


try:
    universe = load_company_universe()
except Exception as exc:
    st.error(f"Could not load company universe from config.toml: {exc}")
    st.stop()

api_key = resolve_simfin_api_key(st.secrets, os.environ)
if not api_key:
    st.error(
        "Missing API key. Add `SIMFIN_API_KEY` to your local `.env` file or Streamlit Cloud secrets."
    )
    st.stop()

default_end = date.today()
default_start = default_end - timedelta(days=120)

selected_ticker = st.selectbox("Ticker", options=universe)
start_date = st.date_input("Start date", value=default_start, max_value=default_end)
end_date = st.date_input("End date", value=default_end, min_value=start_date, max_value=default_end)
predictor = MockPredictor()

if st.button("Run Live Analysis", type="primary"):
    base_url = os.getenv("SIMFIN_BASE_URL", "https://backend.simfin.com/api/v3")
    try:
        with st.spinner("Fetching market data from SimFin..."):
            raw_data = fetch_share_prices_cached(
                ticker=selected_ticker,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                api_key=api_key,
                base_url=base_url,
            )

        result = build_go_live_result(raw_data=raw_data, predictor=predictor)
    except Exception as exc:
        st.error(format_client_error(exc))
        st.stop()

    st.success("Analysis finished.")
    st.write(f"Ticker: `{selected_ticker}` | Start: `{start_date}` | End: `{end_date}`")
    st.write(f"Predicted movement: **{result.signal.get('movement', 'UNKNOWN')}**")
    st.write(f"Suggested action: **{result.signal.get('action', 'HOLD')}**")
    st.write(f"Confidence: **{result.signal.get('confidence', 'N/A')}**")
    st.caption(str(result.signal.get("reason", "")))

    st.subheader("Raw Share Prices")
    st.dataframe(result.raw_data.tail(20), use_container_width=True)

    st.subheader("Transformed Features")
    preview_columns = [col for col in ["Date", "Ticker", "Close", "Adj. Close", "Return_1d"] if col in result.features.columns]
    st.dataframe(result.features[preview_columns].tail(20), use_container_width=True)
