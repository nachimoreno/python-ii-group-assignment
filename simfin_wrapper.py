import time

import numpy as np
import pandas as pd
import requests


class PySimFinError(Exception):
    """Base error for SimFin wrapper."""


class PySimFinAuthError(PySimFinError):
    """Raised when API key is invalid or unauthorized."""


class PySimFinRateLimitError(PySimFinError):
    """Raised when SimFin rate limit is exceeded."""


class PySimFinRequestError(PySimFinError):
    """Raised for generic request/network/API response failures."""


class PySimFin:
    """
    Simple object-oriented wrapper around SimFin API.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://backend.simfin.com/api/v3",
        min_interval: float = 0.5,
        timeout: int = 20,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required.")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.min_interval = min_interval
        self.timeout = timeout
        self.session = session or requests.Session()
        self._last_request_at: float | None = None

    def get_share_prices(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        print(f"[PySimFin] Fetching share prices: ticker={ticker}, start={start}, end={end}")
        params = {"ticker": ticker.upper(), "start": start, "end": end}
        response = self._request("/companies/prices", params)
        records = self._parse_payload(response)
        frame = self._to_dataframe(records, self._share_price_schema())
        print(f"[PySimFin] Share prices rows fetched: {len(frame)}")
        return frame

    def get_financial_statement(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        print(
            f"[PySimFin] Fetching financial statements: ticker={ticker}, start={start}, end={end}"
        )
        params = {"ticker": ticker.upper(), "start": start, "end": end}
        response = self._request("/companies/statements", params)
        records = self._parse_payload(response)
        frame = self._to_dataframe(records, self._financial_statement_schema())
        print(f"[PySimFin] Financial statement rows fetched: {len(frame)}")
        return frame

    def _request(self, path: str, params: dict[str, str]) -> requests.Response:
        self._throttle()

        url = f"{self.base_url}{path}"
        headers = {"Authorization": f"api-key {self.api_key}"}

        try:
            response = self.session.get(url, headers=headers, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise PySimFinRequestError(f"Network error while calling SimFin: {exc}") from exc

        if response.status_code in (401, 403):
            raise PySimFinAuthError("Authentication failed. Check your SimFin API key.")
        if response.status_code == 429:
            raise PySimFinRateLimitError("Rate limit reached (free tier: max 2 requests/second).")
        if response.status_code >= 400:
            raise PySimFinRequestError(
                f"Request failed with status {response.status_code}: {response.text[:200]}"
            )

        return response

    def _throttle(self) -> None:
        """
        Keep a minimum interval between requests (SimFin free tier limit protection).
        """
        now = time.monotonic()
        if self._last_request_at is not None:
            elapsed = now - self._last_request_at
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
                now = time.monotonic()
        self._last_request_at = now

    def _parse_payload(self, response: requests.Response) -> list[dict]:
        """
        Keep parsing intentionally simple:
        - list[dict] payload
        - {"data": list[dict]} payload
        Anything else returns empty records.
        """
        try:
            payload = response.json()
        except ValueError as exc:
            raise PySimFinRequestError("Invalid JSON returned by SimFin API.") from exc

        if isinstance(payload, list):
            if payload and isinstance(payload[0], dict):
                return payload
            return []

        if isinstance(payload, dict):
            data = payload.get("data", [])
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return data
            if isinstance(data, list) and not data:
                return []

        return []

    @staticmethod
    def _to_dataframe(records: list[dict], schema: dict[str, str]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame({column: pd.Series(dtype=dtype) for column, dtype in schema.items()})

        frame = pd.DataFrame(records)

        # Ensure all expected columns exist before casting.
        for column in schema:
            if column not in frame.columns:
                frame[column] = np.nan

        for column, dtype in schema.items():
            if dtype == "string":
                frame[column] = frame[column].astype("string")
            elif dtype == "float64":
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            elif dtype == "Int64":
                frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
            elif dtype.startswith("datetime"):
                frame[column] = pd.to_datetime(frame[column], errors="coerce")

        ordered_columns = list(schema.keys()) + [
            column for column in frame.columns if column not in schema
        ]
        return frame[ordered_columns]

    @staticmethod
    def _share_price_schema() -> dict[str, str]:
        return {
            "Date": "string",
            "Ticker": "string",
            "Open": "float64",
            "High": "float64",
            "Low": "float64",
            "Close": "float64",
            "Adj. Close": "float64",
            "Volume": "float64",
            "Dividend": "float64",
        }

    @staticmethod
    def _financial_statement_schema() -> dict[str, str]:
        return {
            "Ticker": "string",
            "Report Date": "string",
            "Fiscal Year": "Int64",
            "Fiscal Period": "string",
            "Revenue": "float64",
            "Net Income": "float64",
        }
