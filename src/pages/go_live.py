import os
from datetime import date, timedelta
import tomllib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PySimFin import PySimFin
from PySimFin import transform_share_prices, predict_company_classification, run_strategy_backtest

companies_dict = {
    "AMZN": "AMAZON COM INC",
    "AAPL": "APPLE INC",
    "MSFT": "MICROSOFT CORP",
    "AAON": "AAON INC",
    "ALL": "ALLSTATE CORP",
}

st.set_page_config(page_title="Live Analysis", layout="wide")
st.title("Live Analysis")
st.caption("Fetch live data from SimFin, apply ETL transformations, and view model signals.")

default_end = date.today()
default_start = default_end - timedelta(days=120)

try:
    with open("src/config.toml", "rb") as f:
        tickers = tomllib.load(f).get("tickers", [])
except Exception as exc:
    st.error(f"go_live.py: Error loading tickers from src/config.toml: {exc}")
    tickers = []

col1, col2, col3, col4 = st.columns(4, vertical_alignment="bottom")
with col1:
    selected_ticker = st.selectbox("Ticker", options=tickers)
    company = companies_dict[selected_ticker]
with col2:
    start_date = st.date_input("Start date", value=default_start, max_value=default_end)
with col3:
    end_date = st.date_input("End date", value=default_end, min_value=start_date, max_value=default_end)

business_days_in_range = len(pd.bdate_range(start=start_date, end=end_date))
date_range_is_valid = business_days_in_range >= 30

with col4:
    analysis_run = False
    run_analysis = st.button(
        "Run Live Analysis",
        type="primary",
        disabled=not date_range_is_valid,
    )

if not date_range_is_valid:
    st.warning(
        f"Please select at least 30 business days. Current selection has {business_days_in_range} business days."
    )
    
if run_analysis:
    analysis_run = True

try:
    with st.spinner(f"Fetching market data for {selected_ticker}..."):
        client = PySimFin()
        raw_data = client.get_share_prices(
                ticker=selected_ticker,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
            )

        transformed_data = transform_share_prices(raw_data)

        if not analysis_run:
            st.header("Latest Data")
            st.dataframe(raw_data)
        else:
            st.header("Analysis Results")
            predictions_df = predict_company_classification(
                input_dataframe=transformed_data,
                company=company,
                model_output_dir=os.path.join('ETL', 'data', 'gold', 'trained_models'),
            )
            raw_data_with_predictions = raw_data.iloc[20:-1].copy().reset_index(drop=True)

            raw_data_with_predictions["Prediction Confidence"] = predictions_df["Prediction Confidence"].to_numpy()
            raw_data_with_predictions["Prediction"] = predictions_df["Prediction"].to_numpy()
            raw_data_with_predictions["Prediction Class"] = (
                predictions_df["Prediction"]
                .map({1: "Bullish", 0: "Bearish"})
                .to_numpy()
            )

            market_sentiment = raw_data_with_predictions['Prediction'].mean()

            st.dataframe(raw_data_with_predictions.drop(columns=['Prediction']))

            st.header("Market Sentiment")
            col1, col2 = st.columns(2)

            is_bullish = market_sentiment > 0.5
            background_color = "#d4edda" if is_bullish else "#f8d7da"
            text_color = "#155724" if is_bullish else "#721c24"

            sentiment_label = "Bullish" if is_bullish else "Bearish"
            recommendation_label = "BUY" if is_bullish else "SELL"
            bias_text = (
                f"({(market_sentiment - 0.5)*2:.2%} bias towards bullish)"
                if is_bullish
                else f"({(0.5 - market_sentiment)*2:.2%} bias towards bearish)"
            )

            with col1:
                st.markdown(
                    f"""
                    <div style="
                        background-color: {background_color};
                        color: {text_color};
                        padding: 24px;
                        border-radius: 12px;
                        text-align: center;
                        min-height: 180px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
                    ">
                        <div style="font-size: 16px; font-weight: 600; margin-bottom: 12px;">
                            Market sentiment
                        </div>
                        <div style="font-size: 32px; font-weight: 700; margin-bottom: 10px;">
                            {sentiment_label}
                        </div>
                        <div style="font-size: 16px;">
                            {bias_text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                    <div style="
                        background-color: {background_color};
                        color: {text_color};
                        padding: 24px;
                        border-radius: 12px;
                        text-align: center;
                        min-height: 180px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
                    ">
                        <div style="font-size: 16px; font-weight: 600; margin-bottom: 12px;">
                            Recommendation
                        </div>
                        <div style="font-size: 32px; font-weight: 700;">
                            {recommendation_label}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            backtest_dataframe, backtest_summary = run_strategy_backtest(
                prediction_dataframe=raw_data_with_predictions,
                price_column="Close",
                initial_cash=10000.0,
            )

            st.subheader("")
            st.header("Backtest Summary")

            summary_col1, summary_col2, summary_col3 = st.columns(3)

            with summary_col1:
                st.metric("Initial Cash", f"${backtest_summary['initial_cash']:,.2f}")
                st.metric("Final Portfolio Value", f"${backtest_summary['final_portfolio_value']:,.2f}")

            with summary_col2:
                st.metric("Total Return", f"{backtest_summary['total_return']:.2%}")
                st.metric("Buy Trades", backtest_summary["buy_trades"])

            with summary_col3:
                st.metric("Sell Trades", backtest_summary["sell_trades"])
                st.metric("Hold Days", backtest_summary["hold_days"])

            st.dataframe(backtest_dataframe)

except Exception as exc:
    st.error(f"go_live.py: Error fetching market data: {exc}")
    raise exc
        
