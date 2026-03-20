import pytest
import requests

from simfin_wrapper import PySimFin, PySimFinAuthError, PySimFinRateLimitError, PySimFinRequestError


class FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


COMPACT_PAYLOAD = [{
    "ticker": "AAPL",
    "columns": ["Date", "Open", "High", "Low", "Close", "Adj. Close", "Volume"],
    "data": [["2024-01-02", 184.0, 186.0, 183.0, 185.0, 185.0, 50_000_000]],
}]


def test_request_sends_auth_header_and_params(monkeypatch):
    captured = {}

    def fake_get(url, **kw):
        captured.update(url=url, headers=kw.get("headers"), params=kw.get("params"))
        return FakeResponse(200, COMPACT_PAYLOAD)

    monkeypatch.setattr(requests, "get", fake_get)
    client = PySimFin(api_key="test-key")
    client.get_share_prices(ticker="AAPL", start="2024-01-01", end="2024-01-31")

    assert "companies/prices/compact" in captured["url"]
    assert captured["headers"]["Authorization"] == "api-key test-key"
    assert captured["params"] == {"ticker": "AAPL", "start": "2024-01-01", "end": "2024-01-31"}


@pytest.mark.parametrize("status,exc_cls", [
    (401, PySimFinAuthError),
    (403, PySimFinAuthError),
    (429, PySimFinRateLimitError),
    (500, PySimFinRequestError),
])
def test_http_errors_raise_correct_exception(monkeypatch, status, exc_cls):
    monkeypatch.setattr(requests, "get", lambda *a, **kw: FakeResponse(status, {}))
    client = PySimFin(api_key="key")
    with pytest.raises(exc_cls):
        client.get_share_prices("AAPL", "2024-01-01", "2024-01-02")


def test_network_error_raises_request_error(monkeypatch):
    def boom(*a, **kw):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "get", boom)
    client = PySimFin(api_key="key")
    with pytest.raises(PySimFinRequestError):
        client.get_share_prices("AAPL", "2024-01-01", "2024-01-02")


def test_compact_format_parsed_correctly(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **kw: FakeResponse(200, COMPACT_PAYLOAD))
    client = PySimFin(api_key="key")
    df = client.get_share_prices("AAPL", "2024-01-01", "2024-01-03")

    assert len(df) == 1
    assert df.iloc[0]["Close"] == 185.0
    assert list(df.columns) == ["Date", "Open", "High", "Low", "Close", "Adj. Close", "Volume"]
