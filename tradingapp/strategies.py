"""Indikatoren und Signale nach bekannten Trader-Strategien, mit Backtest pro Aktie."""
import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    for n in (20, 50, 150, 200):
        df[f"SMA{n}"] = c.rolling(n).mean()
    df["EMA50"] = c.ewm(span=50, adjust=False).mean()
    df["RSI2"] = rsi(c, 2)
    df["RSI14"] = rsi(c, 14)
    sd = c.rolling(20).std()
    df["BB_up"], df["BB_lo"] = df["SMA20"] + 2 * sd, df["SMA20"] - 2 * sd
    df["BB_width"] = (df["BB_up"] - df["BB_lo"]) / df["SMA20"]
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_sig"] = df["MACD"].ewm(span=9, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["HH20"], df["LL20"] = h.rolling(20).max().shift(), l.rolling(20).min().shift()
    df["VolAvg"] = v.rolling(20).mean()
    # VWAP je Handelstag (bei Tageskerzen: rollierender 20er-VWAP)
    tp = (h + l + c) / 3
    day = df.index.normalize()
    if df.index.to_series().diff().median() < pd.Timedelta("1D"):
        pv = (tp * v).groupby(day).cumsum()
        vv = v.groupby(day).cumsum().replace(0, np.nan)
        df["VWAP"] = pv / vv
        df["DayFirst"] = ~day.duplicated()
    else:
        df["VWAP"] = (tp * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
        df["DayFirst"] = False
    return df


def _cross_up(a, b):
    return (a > b) & (a.shift() <= b.shift())


def _cross_dn(a, b):
    return (a < b) & (a.shift() >= b.shift())


def s_connors_rsi2(df):
    long_ = (df["RSI2"] < 10) & (df["Close"] > df["SMA200"])
    short = (df["RSI2"] > 90) & (df["Close"] < df["SMA200"])
    return long_.astype(int) - short.astype(int)


def s_bollinger_squeeze(df):
    tight = df["BB_width"].shift() < df["BB_width"].rolling(120).quantile(0.2).shift()
    long_ = tight & _cross_up(df["Close"], df["BB_up"])
    short = tight & _cross_dn(df["Close"], df["BB_lo"])
    return long_.astype(int) - short.astype(int)


def s_macd_momentum(df):
    long_ = _cross_up(df["MACD"], df["MACD_sig"]) & (df["Close"] > df["EMA50"])
    short = _cross_dn(df["MACD"], df["MACD_sig"]) & (df["Close"] < df["EMA50"])
    return long_.astype(int) - short.astype(int)


def s_vwap_pullback(df):
    up = df["Close"] > df["EMA50"]
    dn = df["Close"] < df["EMA50"]
    long_ = up & (df["Low"] <= df["VWAP"]) & (df["Close"] > df["VWAP"]) & (df["Close"].shift() > df["VWAP"].shift())
    short = dn & (df["High"] >= df["VWAP"]) & (df["Close"] < df["VWAP"]) & (df["Close"].shift() < df["VWAP"].shift())
    return long_.astype(int) - short.astype(int)


def s_opening_range_breakout(df):
    if not df["DayFirst"].any():
        return pd.Series(0, index=df.index)
    day = df.index.normalize()
    or_hi = df["High"].where(df["DayFirst"]).groupby(day).transform("first")
    or_lo = df["Low"].where(df["DayFirst"]).groupby(day).transform("first")
    long_ = _cross_up(df["Close"], or_hi) & ~df["DayFirst"]
    short = _cross_dn(df["Close"], or_lo) & ~df["DayFirst"]
    # nur der erste Ausbruch je Tag zählt
    long_ = long_ & (long_.astype(int).groupby(day).cumsum() == 1)
    short = short & (short.astype(int).groupby(day).cumsum() == 1)
    return long_.astype(int) - short.astype(int)


def s_minervini_breakout(df):
    bars_per_day = max(1, round(len(df) / max(1, df.index.normalize().nunique())))
    hi52 = df["High"].rolling(min(252 * bars_per_day, len(df)), min_periods=20).max()
    trend = (df["Close"] > df["SMA50"]) & (df["SMA50"] > df["SMA150"]) & (df["SMA150"] > df["SMA200"])
    near_high = df["Close"] >= 0.75 * hi52
    breakout = _cross_up(df["Close"], df["HH20"]) & (df["Volume"] > 1.5 * df["VolAvg"])
    return (trend & near_high & breakout).astype(int)


def s_turtle_donchian(df):
    return _cross_up(df["Close"], df["HH20"]).astype(int) - _cross_dn(df["Close"], df["LL20"]).astype(int)


STRATEGIES = {
    "Connors RSI(2) Reversion": (s_connors_rsi2, "Kauft starke Rücksetzer im Aufwärtstrend (Larry Connors)."),
    "Bollinger Squeeze": (s_bollinger_squeeze, "Ausbruch nach enger Bandbreite (John Bollinger / TTM Squeeze)."),
    "MACD Momentum": (s_macd_momentum, "MACD-Kreuzung in Trendrichtung (Gerald Appel)."),
    "VWAP Pullback": (s_vwap_pullback, "Rücklauf an den VWAP im Trend, Klassiker der Daytrader."),
    "Opening Range Breakout": (s_opening_range_breakout, "Ausbruch aus der ersten Kerze des Tages (Toby Crabel)."),
    "Minervini/O'Neil Breakout": (s_minervini_breakout, "Trend-Template + Ausbruch auf hohem Volumen (Mark Minervini, William O'Neil)."),
    "Turtle Donchian": (s_turtle_donchian, "20-Perioden-Hoch/Tief-Ausbruch (Richard Dennis, Turtle Traders)."),
}


def evaluate(df: pd.DataFrame, horizon: int, recent_bars: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Berechnet Signale, testet jede Strategie auf der Historie dieser Aktie und meldet aktive Signale.

    Rückgabe: (DataFrame mit Indikatoren + Signalspalten, Tabelle mit Backtest-Statistik je Strategie)
    """
    ind = add_indicators(df)
    fwd = ind["Close"].shift(-horizon) / ind["Close"] - 1
    uncond = fwd.mean()
    rows = []
    for name, (fn, desc) in STRATEGIES.items():
        sig = fn(ind).fillna(0).astype(int)
        ind[f"sig::{name}"] = sig
        trades = sig[(sig != 0) & fwd.notna()]
        pnl = trades * fwd[trades.index]  # Rendite in Signalrichtung
        n = len(pnl)
        mean = pnl.mean() if n else np.nan
        t = mean / (pnl.std(ddof=1) / np.sqrt(n)) if n > 2 and pnl.std() > 0 else np.nan
        last = sig.iloc[-recent_bars:]
        active = int(last[last != 0].iloc[-1]) if (last != 0).any() else 0
        rows.append(dict(Strategie=name, Beschreibung=desc, Aktiv=active, Trades=n,
                         Trefferquote=(pnl > 0).mean() if n else np.nan,
                         Ø_Rendite=mean, t_Wert=t))
    stats = pd.DataFrame(rows)
    stats.attrs["uncond_fwd"] = uncond
    return ind, stats


def signal_edge(stats: pd.DataFrame) -> float:
    """Erwartete Überrendite (über den Horizont, in Long-Richtung) aus den aktiven Signalen.

    Jede aktive Strategie trägt ihre historische Ø-Rendite bei, geschrumpft nach Anzahl Trades
    und nur, wenn sie auf dieser Aktie überhaupt einen positiven Vorteil hatte.
    """
    uncond = stats.attrs.get("uncond_fwd", 0.0) or 0.0
    act = stats[(stats["Aktiv"] != 0) & (stats["Trades"] >= 5)]
    edges, weights = [], []
    for _, r in act.iterrows():
        if not (r["Ø_Rendite"] > 0):
            continue
        shrink = r["Trades"] / (r["Trades"] + 20)
        excess = r["Ø_Rendite"] - r["Aktiv"] * uncond  # Überrendite ggü. einfachem Halten
        edges.append(r["Aktiv"] * excess * shrink)
        weights.append(max(r["t_Wert"], 0) + 0.5 if pd.notna(r["t_Wert"]) else 0.5)
    if not edges:
        return 0.0
    return float(np.average(edges, weights=weights))
