"""Führt alles zusammen: Daten, Strategien, News-Reaktionen, Monte Carlo, Empfehlung."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data, montecarlo, news as newsmod, strategies

DEFAULT_HORIZON = {"15m": 26, "1h": 21, "1d": 10}  # ≈ 1 Tag, 3 Tage, 2 Wochen


@dataclass
class Analysis:
    symbol: str
    interval: str
    horizon: int
    source: str
    price: float
    action: str
    trade: dict
    long: dict
    short: dict
    mc: dict
    signal_edge: float
    news_drift: float
    drift_per_bar: float
    sigma: float
    prices: pd.DataFrame
    strat_stats: pd.DataFrame
    news: pd.DataFrame
    news_stats: dict
    recent_news: pd.DataFrame
    paths_sample: np.ndarray
    log: list = field(default_factory=list)

    @property
    def entry(self):
        return self.price

    @property
    def target(self):
        return self.price * (1 + self.trade["direction"] * self.trade["tp"])

    @property
    def stop(self):
        return self.price * (1 - self.trade["direction"] * self.trade["sl"])


def analyze(symbol: str, interval: str = "1h", horizon: int | None = None, n_paths: int = 10000,
            keys: dict | None = None, news_days: int = 365, allow_short: bool = True,
            min_ev: float = 0.001, prices: pd.DataFrame | None = None, daily: pd.DataFrame | None = None,
            bench: pd.DataFrame | None = None, news_df: pd.DataFrame | None = None, seed: int | None = None) -> Analysis:
    keys = keys or {}
    log = []
    symbol = symbol.upper().strip()
    horizon = horizon or DEFAULT_HORIZON[interval]
    crypto = data.is_crypto(symbol)

    if prices is None:
        prices, src = data.get_prices(symbol, interval, keys)
    else:
        src = "übergeben"
    log.append(f"Kurse: {src}, {len(prices)} Kerzen ({interval})")
    if daily is None:
        daily = prices if interval == "1d" else data.get_prices(symbol, "1d", keys)[0]
    if bench is None:
        try:
            bench = data.get_prices("BTC-USD" if crypto else "SPY", "1d", keys)[0] if symbol not in ("SPY", "BTC-USD") else None
        except Exception as e:
            log.append(f"Benchmark nicht verfügbar: {e}")

    # 1) Strategien: Signale + wie gut sie bei genau diesem Wert funktioniert haben
    ind, sstats = strategies.evaluate(prices, horizon)
    edge = strategies.signal_edge(sstats)

    # 2) News: alle Nachrichten holen und Kursreaktion messen
    if news_df is None:
        news_df, nlog = newsmod.get_news(symbol, keys, news_days)
        log += nlog
    reacted = newsmod.event_study(news_df, daily, bench, prices if interval != "1d" else None, crypto)
    nstats = newsmod.reaction_stats(reacted)
    price = float(prices["Close"].iloc[-1])
    ndrift, recent = newsmod.expected_news_drift(reacted, nstats, price, prices)

    # 3) Drift pro Kerze: Grundtrend (halbiert) + Signalvorteil + erwartete News-Reaktion
    bpd = data.INTERVALS[interval]["bars_per_day"]
    news_bars = min(horizon, 3 * bpd)
    r = np.diff(np.log(prices["Close"].to_numpy()))
    base = 0.5 * float(np.nanmean(r[-min(len(r), 250 * bpd):]))
    drift = base + edge / horizon + ndrift * (news_bars / horizon) / horizon

    # 4) Monte Carlo
    paths, sigma = montecarlo.simulate(prices["Close"], horizon, n_paths, drift, seed=seed)
    long_ = montecarlo.optimize(paths, +1, sigma)
    short = montecarlo.optimize(paths, -1, sigma)
    mc = montecarlo.summary(paths)

    cands = [long_] + ([short] if allow_short else [])
    best = max(cands, key=lambda c: (c["ev"], c["p_win"]))
    if best["ev"] < min_ev:
        action = "WARTEN"
    else:
        action = "KAUFEN" if best["direction"] > 0 else "SHORT"

    return Analysis(symbol, interval, horizon, src, price, action, best, long_, short, mc, edge, ndrift, drift,
                    sigma, ind, sstats, reacted, nstats, recent, paths[:200], log)


def overview_row(a: Analysis) -> dict:
    active = a.strat_stats[a.strat_stats["Aktiv"] != 0]
    return {
        "Symbol": a.symbol,
        "Kurs": round(a.price, 4),
        "Empfehlung": a.action,
        "Einstieg": round(a.entry, 4),
        "Ziel": round(a.target, 4),
        "Stop": round(a.stop, 4),
        "Gewinnchance": a.trade["p_win"],
        "Ziel zuerst": a.trade["p_tp"],
        "Erwartung/Trade": a.trade["ev"],
        "Horizont (Kerzen)": a.horizon,
        "Aktive Signale": ", ".join(f"{n} ({'↑' if d > 0 else '↓'})" for n, d in zip(active["Strategie"], active["Aktiv"])) or "–",
        "News-Impuls": a.news_drift,
    }
