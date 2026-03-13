import os

from simfin_wrapper import PySimFinAuthError, PySimFinRateLimitError, PySimFinRequestError


def resolve_simfin_api_key(
    secrets=None,
    environ=None,
) -> str | None:
    """
    Resolve API key from Streamlit secrets first, then environment variables.
    """
    if secrets and "SIMFIN_API_KEY" in secrets and secrets["SIMFIN_API_KEY"]:
        return str(secrets["SIMFIN_API_KEY"]).strip()

    env = environ or os.environ
    api_key = env.get("SIMFIN_API_KEY") or env.get("API_KEY")
    return api_key.strip() if api_key else None


def format_client_error(exc: Exception) -> str:
    """
    Convert technical exceptions into clear user-friendly text.
    """
    if isinstance(exc, PySimFinAuthError):
        return "Authentication failed. Please verify your SIMFIN_API_KEY."
    if isinstance(exc, PySimFinRateLimitError):
        return "Rate limit reached. Please wait a moment and try again."
    if isinstance(exc, PySimFinRequestError):
        return f"SimFin request failed: {exc}"
    return f"Unexpected error: {exc}"
