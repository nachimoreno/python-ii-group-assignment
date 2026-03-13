from pathlib import Path
from typing import Any
import tomllib


def load_settings(config_path: str | Path = "config.toml") -> dict[str, Any]:
    path = Path(config_path)
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def load_company_universe(config_path: str | Path = "config.toml") -> list[str]:
    config = load_settings(config_path)

    tickers = config.get("tickers", [])
    if isinstance(tickers, list) and tickers:
        return [str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()]

    companies = config.get("companies", [])
    if isinstance(companies, list) and companies:
        return [str(company).strip() for company in companies if str(company).strip()]

    raise ValueError("No company universe configured. Add `tickers` or `companies` in config.toml.")
