from dotenv import load_dotenv
import pandas as pd
import requests
import simfin as sf
import os
import polars as pl
import json
import pickle


class PySimFinError(Exception):
    pass


class PySimFinAuthError(PySimFinError):
    pass


class PySimFinRateLimitError(PySimFinError):
    pass


class PySimFinRequestError(PySimFinError):
    pass


_PRICE_COLUMNS = {
    "Opening Price": "Open",
    "Highest Price": "High",
    "Lowest Price": "Low",
    "Last Closing Price": "Close",
    "Adjusted Closing Price": "Adj. Close",
    "Trading Volume": "Volume",
    "Dividend Paid": "Dividend",
    "Common Shares Outstanding": "Shares Outstanding",
}

def generate_trading_actions(
    prediction_dataframe: pd.DataFrame,
    buy_confidence_threshold: float = 0.55,
    sell_confidence_threshold: float = 0.55,
    base_trade_size: int = 1,
    high_confidence_trade_size: int = 5,
    high_confidence_threshold: float = 0.70,
) -> pd.DataFrame:
    action_dataframe = prediction_dataframe.copy()

    action_list = []
    trade_size_list = []

    for _, row in action_dataframe.iterrows():
        prediction = row["Prediction"]
        prediction_confidence = float(row["Prediction Confidence"])

        action = "HOLD"
        trade_size = 0

        if prediction in [1, "Bullish"]:
            if prediction_confidence >= high_confidence_threshold:
                action = "BUY"
                trade_size = high_confidence_trade_size
            elif prediction_confidence >= buy_confidence_threshold:
                action = "BUY"
                trade_size = base_trade_size

        elif prediction in [0, "Bearish"]:
            if prediction_confidence >= high_confidence_threshold:
                action = "SELL"
                trade_size = high_confidence_trade_size
            elif prediction_confidence >= sell_confidence_threshold:
                action = "SELL"
                trade_size = base_trade_size

        action_list.append(action)
        trade_size_list.append(trade_size)

    action_dataframe["Action"] = action_list
    action_dataframe["Trade Size"] = trade_size_list

    return action_dataframe


def backtest_trading_strategy(
    action_dataframe: pd.DataFrame,
    price_column: str = "Close",
    initial_cash: float = 10000.0,
) -> pd.DataFrame:
    backtest_dataframe = action_dataframe.copy()

    cash = float(initial_cash)
    shares_held = 0

    executed_action_list = []
    executed_trade_size_list = []
    trade_value_list = []
    cash_history = []
    shares_held_history = []
    portfolio_value_history = []

    for _, row in backtest_dataframe.iterrows():
        current_price = float(row[price_column])
        requested_action = row["Action"]
        requested_trade_size = int(row["Trade Size"])

        executed_action = "HOLD"
        executed_trade_size = 0
        trade_value = 0.0

        if requested_action == "BUY" and requested_trade_size > 0:
            affordable_trade_size = int(cash // current_price)
            shares_to_buy = min(requested_trade_size, affordable_trade_size)

            if shares_to_buy > 0:
                trade_value = shares_to_buy * current_price
                cash -= trade_value
                shares_held += shares_to_buy
                executed_action = "BUY"
                executed_trade_size = shares_to_buy

        elif requested_action == "SELL" and requested_trade_size > 0:
            shares_to_sell = min(requested_trade_size, shares_held)

            if shares_to_sell > 0:
                trade_value = shares_to_sell * current_price
                cash += trade_value
                shares_held -= shares_to_sell
                executed_action = "SELL"
                executed_trade_size = shares_to_sell

        portfolio_value = cash + shares_held * current_price

        executed_action_list.append(executed_action)
        executed_trade_size_list.append(executed_trade_size)
        trade_value_list.append(trade_value)
        cash_history.append(cash)
        shares_held_history.append(shares_held)
        portfolio_value_history.append(portfolio_value)

    backtest_dataframe["Executed Action"] = executed_action_list
    backtest_dataframe["Executed Trade Size"] = executed_trade_size_list
    backtest_dataframe["Trade Value"] = trade_value_list
    backtest_dataframe["Cash"] = cash_history
    backtest_dataframe["Shares Held"] = shares_held_history
    backtest_dataframe["Portfolio Value"] = portfolio_value_history

    return backtest_dataframe


def summarize_backtest_results(
    backtest_dataframe: pd.DataFrame,
    initial_cash: float = 10000.0,
) -> dict:
    if backtest_dataframe.empty:
        return {
            "initial_cash": initial_cash,
            "final_portfolio_value": initial_cash,
            "total_return": 0.0,
            "buy_trades": 0,
            "sell_trades": 0,
            "hold_days": 0,
            "total_shares_bought": 0,
            "total_shares_sold": 0,
        }

    final_portfolio_value = float(backtest_dataframe["Portfolio Value"].iloc[-1])
    total_return = (final_portfolio_value - initial_cash) / initial_cash

    buy_trades = int((backtest_dataframe["Executed Action"] == "BUY").sum())
    sell_trades = int((backtest_dataframe["Executed Action"] == "SELL").sum())
    hold_days = int((backtest_dataframe["Executed Action"] == "HOLD").sum())

    total_shares_bought = int(
        backtest_dataframe.loc[
            backtest_dataframe["Executed Action"] == "BUY",
            "Executed Trade Size",
        ].sum()
    )

    total_shares_sold = int(
        backtest_dataframe.loc[
            backtest_dataframe["Executed Action"] == "SELL",
            "Executed Trade Size",
        ].sum()
    )

    return {
        "initial_cash": float(initial_cash),
        "final_portfolio_value": final_portfolio_value,
        "total_return": float(total_return),
        "buy_trades": buy_trades,
        "sell_trades": sell_trades,
        "hold_days": hold_days,
        "total_shares_bought": total_shares_bought,
        "total_shares_sold": total_shares_sold,
    }


def run_strategy_backtest(
    prediction_dataframe: pd.DataFrame,
    price_column: str = "Close",
    initial_cash: float = 10000.0,
    buy_confidence_threshold: float = 0.55,
    sell_confidence_threshold: float = 0.55,
    base_trade_size: int = 1,
    high_confidence_trade_size: int = 5,
    high_confidence_threshold: float = 0.70,
) -> tuple[pd.DataFrame, dict]:
    action_dataframe = generate_trading_actions(
        prediction_dataframe=prediction_dataframe,
        buy_confidence_threshold=buy_confidence_threshold,
        sell_confidence_threshold=sell_confidence_threshold,
        base_trade_size=base_trade_size,
        high_confidence_trade_size=high_confidence_trade_size,
        high_confidence_threshold=high_confidence_threshold,
    )

    backtest_dataframe = backtest_trading_strategy(
        action_dataframe=action_dataframe,
        price_column=price_column,
        initial_cash=initial_cash,
    )

    backtest_summary = summarize_backtest_results(
        backtest_dataframe=backtest_dataframe,
        initial_cash=initial_cash,
    )

    return backtest_dataframe, backtest_summary
    

def transform_share_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['Dividend'])

    lf = pl.from_pandas(df)

    lf = lf.with_columns(
        # Log Return
        (pl.col('Adj. Close') / pl.col('Adj. Close').shift(1)).log().alias(f'Log Return')
    )

    for window in [5, 10, 20]:
        lf = lf.with_columns(
            # Rolling Log Returns
            (pl.col('Adj. Close') / pl.col('Adj. Close').shift(window)).log().alias(f'Log Return {window}d'),

            # Volatility
            pl.col('Log Return').rolling_std(window).log1p().alias(f'Volatility {window}d'),

            # Moving Averages
            pl.col("Adj. Close").rolling_mean(5).alias(f"Moving Average {window}d"),

            # Momentum Pct. Change
            ((pl.col("Adj. Close") / pl.col("Adj. Close").shift(window)) - 1).alias(f"Momentum Pct. {window}d"),

            # Log Volume Ratio
            ((pl.col("Volume") / pl.col("Volume").rolling_mean(window))).log().alias(f"Log Volume Ratio {window}d"),
        )

        lf = lf.with_columns(
            # Log MA Ratio
            (pl.col("Adj. Close") / pl.col(f"Moving Average {window}d")).log().alias(f"Log MA Ratio {window}d")
        )

    lf = lf.with_columns(
        # Intraday Returns
        ((pl.col('Close') / pl.col('Open')) - 1).alias('Intraday Pct. Return'),

        # Ranges
        (pl.col('High') - pl.col('Low')).alias('Range'),
        ((pl.col('High') - pl.col('Low')) / pl.col('Close')).alias('Range Pct.'),

        # Close Position
        (((pl.col('Close') - pl.col('Low')) / (pl.col('High') - pl.col('Low'))) - 0.5).alias('Close Position'),

        # Log Volume Change
        (pl.col("Volume") / pl.col("Volume").shift(1)).log().tanh().alias("Log Volume Change"),

        # Log Market Cap
        (pl.col("Adj. Close") * pl.col("Shares Outstanding")).log().alias("Log Market Cap"),

        # Dilution / Issuance
        (pl.col("Shares Outstanding") / pl.col("Shares Outstanding").shift(1) - 1).alias('Delta Pct. Dilution / Issuance'),

        # Volume Return Interaction
        (pl.col("Log Return") * pl.col("Log Volume Ratio 5d")).tanh().alias("Interaction Return Volume 5d"),
        (pl.col("Log Return") * pl.col("Log Volume Ratio 10d")).tanh().alias("Interaction Return Volume 10d"),
        (pl.col("Log Return") * pl.col("Log Volume Ratio 20d")).tanh().alias("Interaction Return Volume 20d"),

        # Volume Volatility Interaction
        (pl.col("Volatility 5d") * pl.col("Log Volume Ratio 5d")).tanh().alias("Interaction Volatility Volume 5d"),
        (pl.col("Volatility 10d") * pl.col("Log Volume Ratio 10d")).tanh().alias("Interaction Volatility Volume 10d"),
        (pl.col("Volatility 20d") * pl.col("Log Volume Ratio 20d")).tanh().alias("Interaction Volatility Volume 20d"),

        # Momentum Volatility Interaction
        (pl.col("Momentum Pct. 5d") * pl.col("Volatility 5d")).tanh().alias("Interaction Momentum Volatility 5d"),
        (pl.col("Momentum Pct. 10d") * pl.col("Volatility 10d")).tanh().alias("Interaction Momentum Volume 10d"),
        (pl.col("Momentum Pct. 20d") * pl.col("Volatility 20d")).tanh().alias("Interaction Momentum Volume 20d"),

        # Target Engineering
        (pl.col("Adj. Close").shift(-1) / pl.col("Adj. Close") - 1).alias("Target Return Metric"),
        ((pl.col("Adj. Close").shift(-1) / pl.col("Adj. Close") - 1) > 0).alias("Target Return Class")
    )

    lf = lf.drop([
        "Open", 
        "High",
        "Low",
        "Close",
        "Adj. Close",
        "Moving Average 5d",
        "Moving Average 10d",
        "Moving Average 20d",
        "Volume",
        "Range",
        "Shares Outstanding"
    ])

    lf = lf.drop_nulls()

    df = lf.to_pandas()

    return df


def predict_company_classification(
    input_dataframe: pd.DataFrame,
    company: str,
    model_output_dir: str,
) -> pd.DataFrame:
    company_simple_name = company.split()[0].title()
    company_model_dir = os.path.join(model_output_dir, company_simple_name)

    classification_model_path = os.path.join(company_model_dir, "classification_model.pkl")
    metadata_path = os.path.join(company_model_dir, "metadata.json")

    if not os.path.exists(classification_model_path):
        raise FileNotFoundError(
            f"Classification model not found for {company} at {classification_model_path}"
        )

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata not found for {company} at {metadata_path}"
        )

    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    with open(classification_model_path, "rb") as classification_model_file:
        classification_model = pickle.load(classification_model_file)

    feature_columns = metadata["feature_columns"]

    missing_feature_columns = [
        feature_column
        for feature_column in feature_columns
        if feature_column not in input_dataframe.columns
    ]
    if missing_feature_columns:
        raise ValueError(
            f"Input dataframe is missing required feature columns: {missing_feature_columns}"
        )

    model_input_dataframe = input_dataframe[feature_columns].copy()

    prediction_probabilities = classification_model.predict_proba(model_input_dataframe)
    predicted_classes = classification_model.predict(model_input_dataframe)

    output_dataframe = input_dataframe.copy()
    output_dataframe["Prediction"] = predicted_classes
    output_dataframe["Probability Class 0"] = prediction_probabilities[:, 0]
    output_dataframe["Probability Class 1"] = prediction_probabilities[:, 1]
    output_dataframe["Prediction Confidence"] = prediction_probabilities.max(axis=1)

    return output_dataframe


class PySimFin:
    """Python wrapper for the SimFin REST API v3."""

    BASE_URL = "https://backend.simfin.com/api/v3"

    def __init__(self) -> None:
        try:
            load_dotenv()
            self.api_key = os.getenv("API_KEY")
        except Exception as e:
            raise PySimFinAuthError("PySimFin.py: Failed to load API key from environment variables.")

        self.headers = {
            "accept": "application/json",
            "Authorization": f"api-key {self.api_key}"
        }

    def get_share_prices(self, ticker: str, start: str = '2026-02-01', end: str = '2026-03-22') -> pd.DataFrame:
        try:
            url = f"{self.BASE_URL}/companies/prices/compact"
            params = {
                "ticker": ticker.upper(),
                "start": start,
                "end": end
            }
            response = requests.get(url, headers=self.headers, params=params)
        except Exception as e:
            print('PySimFin.py: Failed to load share prices. Raising exception.')
            raise e

        if response.status_code in (401, 403):
            print(f"Response: {response.text}")
            raise PySimFinAuthError("PySimFin.py: Authentication failed. Check your SimFin API key.")
        if response.status_code == 429:
            print(f"Response: {response.text}")
            raise PySimFinRateLimitError("PySimFin.py: Rate limit reached. Wait and try again.")
        if response.status_code >= 400:
            print(f"Response: {response.text}")
            raise PySimFinRequestError(f"PySimFin.py: Request failed ({response.status_code}): {response.text}")

        payload = response.json()
        item = payload[0]
        df = pd.DataFrame(item["data"], columns=item["columns"])
        df = df.rename(columns=_PRICE_COLUMNS)
        df.set_index('Date', inplace=True)
        return df

    def get_financial_statement(self, ticker: str, start: str = '2025-01-01', end: str = '2026-03-22', statement: str = 'pl,bs,cf,derived') -> dict[str, pd.DataFrame]:
        try:
            response = requests.get(
                f"{self.BASE_URL}/companies/statements/compact",
                params={
                    "ticker": ticker.upper(),
                    "start": start,
                    "end": end,
                    "statements": statement
                },
                headers=self.headers
            )
        except Exception as e:
            print('PySimFin.py: Failed to load financial statement. Raising exception.')
            raise e

        if response.status_code in (401, 403):
            raise PySimFinAuthError("PySimFin.py: Authentication failed. Check your SimFin API key.")
        if response.status_code == 429:
            raise PySimFinRateLimitError("PySimFin.py: Rate limit reached. Wait and try again.")
        if response.status_code >= 400:
            raise PySimFinRequestError(f"PySimFin.py: Request failed ({response.status_code}): {response.text}")

        payload = response.json()
        dict_of_dfs = {}
        for item in payload[0]['statements']:
            df = pd.DataFrame(item["data"], columns=item["columns"])
            dict_of_dfs[item['statement']] = df
        return dict_of_dfs