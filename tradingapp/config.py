import os
from pathlib import Path

KEY_NAMES = {
    "FINNHUB_API_KEY": "Finnhub",
    "ALPACA_API_KEY": "Alpaca Key ID",
    "ALPACA_SECRET_KEY": "Alpaca Secret",
    "POLYGON_API_KEY": "Polygon.io",
    "NEWSAPI_KEY": "NewsAPI",
}


def load_env(path: str | Path = ".env") -> None:
    """Liest KEY=VALUE-Zeilen aus .env, ohne bereits gesetzte Variablen zu überschreiben."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_streamlit_secrets() -> None:
    """Übernimmt Keys aus Streamlit-Secrets (Streamlit Cloud), ohne gesetzte Variablen zu überschreiben."""
    try:
        import streamlit as st
        for k in KEY_NAMES:
            if k in st.secrets and st.secrets[k]:
                os.environ.setdefault(k, str(st.secrets[k]))
    except Exception:
        pass


def get_keys(overrides: dict | None = None) -> dict:
    keys = {k: os.environ.get(k, "") for k in KEY_NAMES}
    for k, v in (overrides or {}).items():
        if v:
            keys[k] = v
    return keys
