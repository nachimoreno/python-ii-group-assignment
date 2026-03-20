from dataclasses import dataclass
from typing import Any

import pandas as pd
import polars as pl


@dataclass
class GoLiveResult:
    raw_data: pd.DataFrame
    features: pd.DataFrame
    signal: dict[str, Any]


def engineer_live_features(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering as src/feature_engineering.py on live API data.

    Input is the raw DataFrame from PySimFin.get_share_prices().
    Returns the enriched DataFrame (price columns kept for display).
    """
    lf = pl.from_pandas(raw_data).lazy().sort("Date")

    lf = lf.with_columns(
        (pl.col("Adj. Close") / pl.col("Adj. Close").shift(1)).log().alias("Log Return")
    )

    for window in [5, 10, 20]:
        lf = lf.with_columns(
            (pl.col("Adj. Close") / pl.col("Adj. Close").shift(window)).log().alias(f"Log Return {window}d"),
            pl.col("Log Return").rolling_std(window).log1p().alias(f"Volatility {window}d"),
            pl.col("Adj. Close").rolling_mean(window).alias(f"Moving Average {window}d"),
            ((pl.col("Adj. Close") / pl.col("Adj. Close").shift(window)) - 1).alias(f"Momentum Pct. {window}d"),
            (pl.col("Volume") / pl.col("Volume").rolling_mean(window)).log().alias(f"Log Volume Ratio {window}d"),
        )
        lf = lf.with_columns(
            (pl.col("Adj. Close") / pl.col(f"Moving Average {window}d")).log().alias(f"Log MA Ratio {window}d"),
        )

    lf = lf.with_columns(
        ((pl.col("Close") / pl.col("Open")) - 1).alias("Intraday Pct. Return"),
        (pl.col("High") - pl.col("Low")).alias("Range"),
        ((pl.col("High") - pl.col("Low")) / pl.col("Close")).alias("Range Pct."),
        (((pl.col("Close") - pl.col("Low")) / (pl.col("High") - pl.col("Low"))) - 0.5).alias("Close Position"),
        (pl.col("Volume") / pl.col("Volume").shift(1)).log().tanh().alias("Log Volume Change"),
        (pl.col("Log Return") * pl.col("Log Volume Ratio 5d")).tanh().alias("Interaction Return Volume 5d"),
        (pl.col("Log Return") * pl.col("Log Volume Ratio 10d")).tanh().alias("Interaction Return Volume 10d"),
        (pl.col("Log Return") * pl.col("Log Volume Ratio 20d")).tanh().alias("Interaction Return Volume 20d"),
        (pl.col("Volatility 5d") * pl.col("Log Volume Ratio 5d")).tanh().alias("Interaction Volatility Volume 5d"),
        (pl.col("Volatility 10d") * pl.col("Log Volume Ratio 10d")).tanh().alias("Interaction Volatility Volume 10d"),
        (pl.col("Volatility 20d") * pl.col("Log Volume Ratio 20d")).tanh().alias("Interaction Volatility Volume 20d"),
        (pl.col("Momentum Pct. 5d") * pl.col("Volatility 5d")).tanh().alias("Interaction Momentum Volatility 5d"),
        (pl.col("Momentum Pct. 10d") * pl.col("Volatility 10d")).tanh().alias("Interaction Momentum Volume 10d"),
        (pl.col("Momentum Pct. 20d") * pl.col("Volatility 20d")).tanh().alias("Interaction Momentum Volume 20d"),
    )

    # Optional features that require Shares Outstanding
    if "Shares Outstanding" in raw_data.columns:
        lf = lf.with_columns(
            (pl.col("Adj. Close") * pl.col("Shares Outstanding")).log().alias("Log Market Cap"),
            (pl.col("Shares Outstanding") / pl.col("Shares Outstanding").shift(1) - 1).alias("Delta Pct. Dilution / Issuance"),
        )

    return lf.drop_nulls().collect().to_pandas()


def build_go_live_result(raw_data: pd.DataFrame, predictor: Any) -> GoLiveResult:
    if raw_data.empty:
        raise ValueError("No share-price data returned for the selected ticker and period.")

    features = engineer_live_features(raw_data)
    if features.empty:
        raise ValueError(
            "Feature engineering produced no usable rows. Try a longer date range (at least 30 days)."
        )

    signal = predictor.predict(features)
    return GoLiveResult(raw_data=raw_data, features=features, signal=signal)
