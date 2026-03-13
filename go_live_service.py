from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from data_cleaning import clean_share_prices_frame
from predictors import Predictor


@dataclass
class GoLiveResult:
    raw_data: pd.DataFrame
    features: pd.DataFrame
    signal: dict[str, Any]


def build_go_live_result(raw_data: pd.DataFrame, predictor: Predictor) -> GoLiveResult:
    print("[GoLive] Building go-live result...")
    if raw_data.empty:
        raise ValueError("No share-price data returned for the selected ticker and period.")

    # Reuse the same ETL transformation used in batch mode.
    transformed = clean_share_prices_frame(raw_data)
    if transformed.empty:
        raise ValueError("No transformed features available after ETL processing.")

    signal = predictor.predict(transformed)
    print("[GoLive] Signal generated successfully.")
    return GoLiveResult(raw_data=raw_data, features=transformed, signal=signal)


def run_go_live(
    ticker: str,
    start: date,
    end: date,
    client: Any,
    predictor: Predictor,
) -> GoLiveResult:
    print(f"[GoLive] Fetching data for {ticker} from {start} to {end}...")
    raw_data = client.get_share_prices(ticker=ticker, start=start.isoformat(), end=end.isoformat())
    return build_go_live_result(raw_data=raw_data, predictor=predictor)
