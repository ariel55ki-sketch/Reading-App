"""Offline-Test mit synthetischen Daten (keine API nötig): python -m pytest tests  oder  python tests/test_offline.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tradingapp import recommend  # noqa: E402


def fake_prices(freq, n, seed):
    rng = np.random.default_rng(seed)
    if freq == "1h":
        days = pd.bdate_range("2024-01-02", periods=n // 7 + 1, tz="UTC")
        idx = pd.DatetimeIndex([d + pd.Timedelta(hours=14 + h) for d in days for h in range(7)])[:n]
    else:
        idx = pd.bdate_range("2021-01-04", periods=n, tz="UTC")
    r = rng.standard_t(4, n) * 0.01 + 0.0002
    c = 100 * np.exp(np.cumsum(r))
    o = c * (1 + rng.normal(0, 0.002, n))
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.003, "Low": np.minimum(o, c) * 0.997,
                         "Close": c, "Volume": rng.integers(1e5, 1e6, n).astype(float)}, index=idx)


def fake_news(daily):
    rng = np.random.default_rng(3)
    heads = ["Company beats earnings estimates", "Analyst downgrades stock", "Firm unveils new product",
             "Regulator opens probe into company", "Company raises guidance", "CEO steps down"]
    t = daily.index[rng.integers(30, len(daily) - 1, 120)] + pd.Timedelta(hours=15)
    t = t.append(pd.DatetimeIndex([daily.index[-1] + pd.Timedelta(hours=15)]))
    df = pd.DataFrame({"time": t, "headline": rng.choice(heads, len(t)), "summary": "", "source": "test", "url": ""})
    from tradingapp.news import _category, _sentiment
    df["sentiment"] = df["headline"].map(_sentiment)
    df["category"] = df["headline"].map(_category)
    return df.sort_values("time").reset_index(drop=True)


def test_pipeline():
    hourly = fake_prices("1h", 3500, 1)
    daily = hourly["Close"].resample("1D").last().dropna().to_frame()
    daily = hourly.resample("1D").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    bench = fake_prices("1d", len(daily), 2).set_axis(daily.index)
    a = recommend.analyze("TEST", "1h", prices=hourly, daily=daily, bench=bench, news_df=fake_news(daily),
                          n_paths=5000, seed=0)
    row = recommend.overview_row(a)
    print(pd.Series(row).to_string())
    print(a.strat_stats[["Strategie", "Aktiv", "Trades", "Trefferquote", "Ø_Rendite"]].to_string())
    print(a.news_stats["by_category"].round(4).to_string())
    assert a.action in ("KAUFEN", "SHORT", "WARTEN")
    assert 0 <= a.trade["p_win"] <= 1
    assert a.news["abn_1d"].notna().sum() > 50


if __name__ == "__main__":
    test_pipeline()
    print("OK")
