import pandas as pd
import requests


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


class PySimFin:
    """Python wrapper for the SimFin REST API v3."""

    BASE_URL = "https://backend.simfin.com/api/v3"

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("api_key is required.")
        self.headers = {"Authorization": f"api-key {api_key}"}

    def get_share_prices(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        try:
            response = requests.get(
                f"{self.BASE_URL}/companies/prices/compact",
                headers=self.headers,
                params={"ticker": ticker.upper(), "start": start, "end": end},
            )
        except requests.RequestException as exc:
            raise PySimFinRequestError(f"Network error: {exc}") from exc

        if response.status_code in (401, 403):
            raise PySimFinAuthError("Authentication failed. Check your SimFin API key.")
        if response.status_code == 429:
            raise PySimFinRateLimitError("Rate limit reached. Wait and try again.")
        if response.status_code >= 400:
            raise PySimFinRequestError(f"Request failed ({response.status_code}): {response.text[:200]}")

        payload = response.json()
        item = payload[0]
        df = pd.DataFrame(item["data"], columns=item["columns"])
        return df.rename(columns=_PRICE_COLUMNS)

    def get_financial_statement(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        try:
            response = requests.get(
                f"{self.BASE_URL}/companies/statements/compact",
                headers=self.headers,
                params={"ticker": ticker.upper(), "start": start, "end": end, "statements": "pl"},
            )
        except requests.RequestException as exc:
            raise PySimFinRequestError(f"Network error: {exc}") from exc

        if response.status_code in (401, 403):
            raise PySimFinAuthError("Authentication failed. Check your SimFin API key.")
        if response.status_code == 429:
            raise PySimFinRateLimitError("Rate limit reached. Wait and try again.")
        if response.status_code >= 400:
            raise PySimFinRequestError(f"Request failed ({response.status_code}): {response.text[:200]}")

        payload = response.json()
        item = payload[0]
        return pd.DataFrame(item["data"], columns=item["columns"])
