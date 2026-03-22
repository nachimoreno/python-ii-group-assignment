# Live Analysis

Live Analysis is a Streamlit page for running live stock analysis using SimFin market data, the PySimFin pipeline, and a saved classification model.

## What the page does

This page lets a user:

- choose a stock ticker
- choose a date range
- fetch recent share-price data from SimFin
- transform that data into model features
- run a saved classifier on the transformed data
- view bullish or bearish predictions
- see an overall market sentiment and simple recommendation
- run a basic backtest on the predictions

## Main flow

1. The page loads available tickers from src/config.toml
2. The user selects:
   - a ticker
   - a start date
   - an end date
3. The page checks that the selected range contains at least 30 business days
4. It fetches raw share-price data with PySimFin
5. It applies transform_share_prices to engineer model features
6. If the user clicks Run Live Analysis:
   - it runs predict_company_classification
   - attaches predictions and confidence scores to the price data
   - calculates average prediction direction as market sentiment
   - shows a bullish or bearish recommendation
   - runs run_strategy_backtest
   - displays summary metrics and a trading log

## What appears on the page

- a ticker selector
- start and end date inputs
- a Run Live Analysis button
- a raw market-data table
- a predictions table
- a market sentiment card
- a recommendation card
- backtest summary metrics
- a backtest actions log

## Important details

- the page always fetches and shows the latest raw data for the selected ticker
- the actual model analysis only runs after the button is pressed
- sentiment is based on the average of binary predictions:
  - above 0.5 = Bullish
  - otherwise = Bearish
- the recommendation is simply BUY for bullish sentiment or SELL for bearish sentiment
- the backtest starts with 10000 dollars and uses Close as the trade price

## In short

go_live.py is an interactive dashboard page that connects live SimFin price data to the PySimFin feature-engineering, prediction, sentiment, and backtesting pipeline so a user can inspect recent signals and simulated trading results for a selected stock.
