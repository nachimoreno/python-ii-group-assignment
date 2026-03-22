# PySimFin.py

PySimFin.py is a small utility module for working with SimFin market data, building prediction features, running a classification model, and backtesting simple trading decisions.

## What is in the module

The module has four main parts:

- custom exceptions for auth, rate-limit, and request failures
- helper functions for trading actions and backtesting
- a feature-engineering function for share-price data
- a PySimFin class for calling the SimFin API

## Main pieces

### Custom exceptions

- PySimFinError: base module error
- PySimFinAuthError: auth or API key problems
- PySimFinRateLimitError: SimFin rate limit reached
- PySimFinRequestError: other API request failures

### Trading helpers

- generate_trading_actions:
  takes model predictions and confidence scores, then labels each row as BUY, SELL, or HOLD, with a trade size based on confidence thresholds

- backtest_trading_strategy:
  simulates a simple long-only strategy using cash, share holdings, and a chosen price column

- summarize_backtest_results:
  returns a small summary dictionary with final portfolio value, return, trade counts, and total shares bought and sold

- run_strategy_backtest:
  convenience wrapper that runs all three steps above in sequence

### Feature engineering

- transform_share_prices:
  takes raw price data and creates technical features like returns, volatility, moving averages, momentum, volume ratios, and target columns for modeling

  it uses polars internally, then converts back to pandas

### Prediction helper

- predict_company_classification:
  loads a saved classification model and metadata from disk, checks the required feature columns, then adds prediction outputs and probabilities to the dataframe

### SimFin API wrapper

The PySimFin class wraps two SimFin REST API endpoints:

- get_share_prices:
  downloads compact share-price history for a ticker and returns a pandas dataframe

- get_financial_statement:
  downloads compact financial statements and returns a dictionary of pandas dataframes

## How it works overall

The intended workflow is:

1. create a PySimFin object
2. fetch share-price data
3. transform the data into engineered features
4. load a saved classifier and generate predictions
5. convert predictions into trading actions
6. backtest those actions
7. summarize the results

## Notes

- the module expects an API_KEY in environment variables
- it is built mainly around pandas dataframes
- the backtest is very simple and does not include transaction costs or slippage
- there may be a small bug in the moving-average calculation because the rolling window looks fixed at 5 in that section
