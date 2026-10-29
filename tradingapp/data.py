"""Kursdaten aus Alpaca, Polygon oder Yahoo Finance (Fallback ohne Key)."""
import datetime as dt

import pandas as pd
import requests

# Einstellungen pro Kerzen-Intervall
INTERVALS = {
    "15m": dict(yf="15m", yf_period="60d", alpaca="15Min", poly=(15, "minute"), days=59, bars_per_day=26),
    "1h": dict(yf="1h", yf_period="730d", alpaca="1Hour", poly=(1, "hour"), days=720, bars_per_day=7),
    "1d": dict(yf="1d", yf_period="5y", alpaca="1Day", poly=(1, "day"), days=365 * 5, bars_per_day=1),
}

COLS = ["Open", "High", "Low", "Close", "Volume"]


def is_crypto(symbol: str) -> bool:
    return symbol.upper().endswith(("-USD", "-EUR", "-USDT"))


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df[COLS].astype(float).dropna(subset=["Close"])
    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return df[~df.index.duplicated()].sort_index()


def from_yahoo(symbol: str, interval: str) -> pd.DataFrame:
    import yfinance as yf

    cfg = INTERVALS[interval]
    df = yf.Ticker(symbol).history(period=cfg["yf_period"], interval=cfg["yf"], auto_adjust=True)
    if df.empty:
        raise ValueError(f"Keine Yahoo-Daten für {symbol}")
    return _norm(df)


def from_alpaca(symbol: str, interval: str, key: str, secret: str) -> pd.DataFrame:
    cfg = INTERVALS[interval]
    start = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=cfg["days"])).strftime("%Y-%m-%dT%H:%M:%SZ")
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    params = {"timeframe": cfg["alpaca"], "start": start, "limit": 10000, "adjustment": "all", "feed": "iex"}
    rows, token = [], None
    while True:
        if token:
            params["page_token"] = token
        r = requests.get(f"https://data.alpaca.markets/v2/stocks/{symbol}/bars",
                         headers=headers, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        rows += js.get("bars") or []
        token = js.get("next_page_token")
        if not token:
            break
    if not rows:
        raise ValueError(f"Keine Alpaca-Daten für {symbol}")
    df = pd.DataFrame(rows).rename(columns={"t": "time", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    return _norm(df.set_index(pd.to_datetime(df["time"], utc=True)))


def from_polygon(symbol: str, interval: str, key: str) -> pd.DataFrame:
    cfg = INTERVALS[interval]
    mult, span = cfg["poly"]
    end = dt.date.today()
    start = end - dt.timedelta(days=cfg["days"])
    ticker = f"X:{symbol.replace('-', '')}" if is_crypto(symbol) else symbol
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{mult}/{span}/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": key}
    rows = []
    while url:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        rows += js.get("results") or []
        url = js.get("next_url")
        params = {"apiKey": key}
    if not rows:
        raise ValueError(f"Keine Polygon-Daten für {symbol}")
    df = pd.DataFrame(rows).rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    return _norm(df.set_index(pd.to_datetime(df["t"], unit="ms", utc=True)))


def get_prices(symbol: str, interval: str = "1h", keys: dict | None = None) -> tuple[pd.DataFrame, str]:
    """Holt Kerzen in der Reihenfolge Polygon, Alpaca, Yahoo. Gibt (DataFrame, Quelle) zurück."""
    keys = keys or {}
    errors = []
    if keys.get("POLYGON_API_KEY"):
        try:
            return from_polygon(symbol, interval, keys["POLYGON_API_KEY"]), "Polygon"
        except Exception as e:  # nächste Quelle versuchen
            errors.append(f"Polygon: {e}")
    if keys.get("ALPACA_API_KEY") and keys.get("ALPACA_SECRET_KEY") and not is_crypto(symbol):
        try:
            return from_alpaca(symbol, interval, keys["ALPACA_API_KEY"], keys["ALPACA_SECRET_KEY"]), "Alpaca"
        except Exception as e:
            errors.append(f"Alpaca: {e}")
    try:
        return from_yahoo(symbol, interval), "Yahoo"
    except Exception as e:
        errors.append(f"Yahoo: {e}")
    raise RuntimeError(f"Keine Kursdaten für {symbol}: " + " | ".join(errors))
