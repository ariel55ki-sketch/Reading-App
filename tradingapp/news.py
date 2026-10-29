"""News holen, Stimmung bewerten und messen, wie der Kurs auf jede Nachricht reagiert hat."""
import datetime as dt
import time

import numpy as np
import pandas as pd
import requests

CATEGORIES = {
    "Quartalszahlen": ["earnings", "eps", "revenue", "quarter", "quarterly", "results", "beats", "misses", "profit", "sales"],
    "Prognose": ["guidance", "outlook", "forecast", "raises", "lowers", "expects"],
    "Analysten": ["upgrade", "downgrade", "price target", "analyst", "rating", "overweight", "underweight",
                  "outperform", "underperform", "buy rating", "sell rating"],
    "Übernahme/Deal": ["acquire", "acquisition", "merger", "buyout", "takeover", "stake", "partnership", "deal"],
    "Produkt": ["launch", "unveil", "product", "release", "approval", "fda", "contract", "order", "chip", "model"],
    "Recht/Regulierung": ["lawsuit", "sec ", "probe", "investigation", "antitrust", "fine", "regulator", "court", "ban"],
    "Management": ["ceo", "cfo", "resign", "appoint", "executive", "layoff", "layoffs", "steps down"],
    "Kapitalmaßnahme": ["buyback", "dividend", "offering", "split", "repurchase", "share sale"],
    "Makro/Markt": ["fed", "inflation", "interest rate", "tariff", "economy", "jobs report", "cpi", "recession"],
}

# Finanzspezifische Ergänzungen für VADER
FIN_LEXICON = {
    "beat": 1.8, "beats": 2.0, "miss": -1.8, "misses": -2.0, "upgrade": 2.0, "upgrades": 2.0, "upgraded": 2.0,
    "downgrade": -2.0, "downgrades": -2.0, "downgraded": -2.0, "raises": 1.5, "raised": 1.2, "cuts": -1.5,
    "lowers": -1.5, "surge": 2.0, "surges": 2.0, "soar": 2.2, "soars": 2.2, "jump": 1.5, "jumps": 1.5,
    "rally": 1.8, "plunge": -2.5, "plunges": -2.5, "tumble": -2.0, "tumbles": -2.0, "slump": -2.0,
    "sinks": -2.0, "record": 1.2, "lawsuit": -1.5, "probe": -1.5, "outperform": 1.5, "underperform": -1.5,
    "bullish": 1.8, "bearish": -1.8, "layoffs": -1.0, "buyback": 1.2, "recall": -1.5, "bankruptcy": -3.0,
    "default": -2.0, "guidance": 0.0, "overweight": 1.2, "underweight": -1.2, "weak": -1.5, "strong": 1.5,
}

_analyzer = None


def _sentiment(text: str) -> float:
    global _analyzer
    if _analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _analyzer = SentimentIntensityAnalyzer()
        _analyzer.lexicon.update(FIN_LEXICON)
    return _analyzer.polarity_scores(text)["compound"]


def _category(text: str) -> str:
    t = f" {text.lower()} "
    best, hits = "Sonstiges", 0
    for cat, words in CATEGORIES.items():
        n = sum(w in t for w in words)
        if n > hits:
            best, hits = cat, n
    return best


# ---------- Quellen ----------

def fetch_finnhub(symbol: str, key: str, days: int = 365) -> list[dict]:
    """Finnhub liefert pro Anfrage begrenzt viele Artikel, daher in Monatsfenstern abfragen."""
    out = []
    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    cur = start
    sym = symbol.replace("-USD", "")
    while cur <= end:
        nxt = min(cur + dt.timedelta(days=30), end)
        r = requests.get("https://finnhub.io/api/v1/company-news",
                         params={"symbol": sym, "from": cur.isoformat(), "to": nxt.isoformat(), "token": key},
                         timeout=30)
        if r.status_code == 429:
            time.sleep(2)
            continue
        r.raise_for_status()
        for a in r.json():
            out.append(dict(time=pd.to_datetime(a["datetime"], unit="s", utc=True), headline=a.get("headline", ""),
                            summary=a.get("summary", ""), source=a.get("source", "Finnhub"), url=a.get("url", "")))
        cur = nxt + dt.timedelta(days=1)
        time.sleep(0.25)
    return out


def fetch_newsapi(query: str, key: str) -> list[dict]:
    r = requests.get("https://newsapi.org/v2/everything",
                     params={"q": query, "language": "en", "sortBy": "publishedAt", "pageSize": 100, "apiKey": key},
                     timeout=30)
    r.raise_for_status()
    return [dict(time=pd.to_datetime(a["publishedAt"], utc=True), headline=a.get("title") or "",
                 summary=a.get("description") or "", source=(a.get("source") or {}).get("name", "NewsAPI"),
                 url=a.get("url", "")) for a in r.json().get("articles", [])]


def fetch_alpaca_news(symbol: str, key: str, secret: str, days: int = 365) -> list[dict]:
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    start = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {"symbols": symbol.replace("-", ""), "start": start, "limit": 50, "include_content": "false"}
    out, token = [], None
    for _ in range(200):
        if token:
            params["page_token"] = token
        r = requests.get("https://data.alpaca.markets/v1beta1/news", headers=headers, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        for a in js.get("news", []):
            out.append(dict(time=pd.to_datetime(a["created_at"], utc=True), headline=a.get("headline", ""),
                            summary=a.get("summary", ""), source=a.get("source", "Alpaca"), url=a.get("url", "")))
        token = js.get("next_page_token")
        if not token:
            break
    return out


def fetch_yahoo(symbol: str) -> list[dict]:
    import yfinance as yf

    out = []
    for a in yf.Ticker(symbol).news or []:
        c = a.get("content", a)
        t = c.get("pubDate") or c.get("providerPublishTime")
        if t is None:
            continue
        ts = pd.to_datetime(t, unit="s", utc=True) if isinstance(t, (int, float)) else pd.to_datetime(t, utc=True)
        url = (c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else c.get("link", "")
        out.append(dict(time=ts, headline=c.get("title", ""), summary=c.get("summary", ""),
                        source=(c.get("provider") or {}).get("displayName", "Yahoo") if isinstance(c.get("provider"), dict)
                        else c.get("publisher", "Yahoo"), url=url or ""))
    return out


def get_news(symbol: str, keys: dict, days: int = 365, company_name: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Sammelt News aus allen verbundenen Quellen. Rückgabe: (News-Tabelle, genutzte Quellen/Fehler)."""
    items, log = [], []
    sources = []
    if keys.get("FINNHUB_API_KEY"):
        sources.append(("Finnhub", lambda: fetch_finnhub(symbol, keys["FINNHUB_API_KEY"], days)))
    if keys.get("ALPACA_API_KEY") and keys.get("ALPACA_SECRET_KEY"):
        sources.append(("Alpaca", lambda: fetch_alpaca_news(symbol, keys["ALPACA_API_KEY"], keys["ALPACA_SECRET_KEY"], days)))
    if keys.get("NEWSAPI_KEY"):
        sources.append(("NewsAPI", lambda: fetch_newsapi(company_name or symbol, keys["NEWSAPI_KEY"])))
    sources.append(("Yahoo", lambda: fetch_yahoo(symbol)))
    for name, fn in sources:
        try:
            got = fn()
            items += got
            log.append(f"{name}: {len(got)} Artikel")
        except Exception as e:
            log.append(f"{name}: Fehler ({e})")
    df = pd.DataFrame(items, columns=["time", "headline", "summary", "source", "url"])
    if df.empty:
        return df.assign(sentiment=[], category=[]), log
    df["key"] = df["headline"].str.lower().str.replace(r"\W+", "", regex=True).str[:80]
    df = df.drop_duplicates("key").drop(columns="key").sort_values("time").reset_index(drop=True)
    text = df["headline"].fillna("") + ". " + df["summary"].fillna("")
    df["sentiment"] = df["headline"].fillna("").map(_sentiment) * 0.7 + text.map(_sentiment) * 0.3
    df["category"] = text.map(_category)
    return df, log


# ---------- Kursreaktion ----------

def event_study(news: pd.DataFrame, daily: pd.DataFrame, bench: pd.DataFrame | None = None,
                intraday: pd.DataFrame | None = None, crypto: bool = False) -> pd.DataFrame:
    """Misst für jede Nachricht die Kursreaktion nach 1 Std., 1 Tag und 3 Tagen (auch relativ zum Markt)."""
    if news.empty:
        return news.assign(r_1h=[], r_1d=[], r_3d=[], abn_1d=[], abn_3d=[])
    news = news.copy()
    dates = daily.index.normalize()
    closes = daily["Close"].to_numpy()
    b = None
    if bench is not None and not bench.empty:
        b = bench["Close"].reindex(daily.index).ffill().to_numpy()
    # Aktien: Nachrichten nach ~20 Uhr UTC (nach US-Börsenschluss) wirken am nächsten Handelstag
    shift = pd.Timedelta(0) if crypto else pd.Timedelta(hours=4)
    eff = (pd.DatetimeIndex(news["time"]) + shift).normalize()
    pos = dates.searchsorted(eff)
    r1, r3, a1, a3 = [], [], [], []
    for p in pos:
        if p < 1 or p >= len(closes):
            r1.append(np.nan); r3.append(np.nan); a1.append(np.nan); a3.append(np.nan)
            continue
        base = closes[p - 1]
        x1 = closes[p] / base - 1
        x3 = closes[min(p + 2, len(closes) - 1)] / base - 1 if p + 2 < len(closes) else np.nan
        r1.append(x1); r3.append(x3)
        if b is not None:
            a1.append(x1 - (b[p] / b[p - 1] - 1))
            a3.append(x3 - (b[p + 2] / b[p - 1] - 1) if p + 2 < len(b) else np.nan)
        else:
            a1.append(x1); a3.append(x3)
    news["r_1d"], news["r_3d"], news["abn_1d"], news["abn_3d"] = r1, r3, a1, a3

    news["r_1h"] = np.nan
    if intraday is not None and not intraday.empty:
        idx = intraday.index
        c = intraday["Close"].to_numpy()
        p = idx.searchsorted(pd.DatetimeIndex(news["time"]))
        vals = []
        for t, q in zip(news["time"], p):
            if q < 1 or q >= len(c) or idx[q] - t > pd.Timedelta(days=3):
                vals.append(np.nan)
            else:
                vals.append(c[q] / c[q - 1] - 1)
        news["r_1h"] = vals
    return news


def reaction_stats(news: pd.DataFrame) -> dict:
    """Wie reagiert diese Aktie typischerweise? Je Kategorie und je Stimmungsrichtung."""
    d = news.dropna(subset=["abn_1d"])
    if len(d) < 5:
        return dict(n=len(d), beta=0.0, corr=np.nan, by_category=pd.DataFrame(), by_sentiment=pd.DataFrame())
    s = d["sentiment"]
    # beta: wie stark (in %) die 3-Tage-Überrendite pro Stimmungspunkt mitläuft
    var = s.var()
    beta = float(np.cov(s, d["abn_3d"].fillna(d["abn_1d"]))[0, 1] / var) if var > 0 else 0.0
    d = d.assign(stimmung=pd.cut(s, [-1.01, -0.15, 0.15, 1.01], labels=["negativ", "neutral", "positiv"]),
                 richtig=np.sign(d["abn_1d"]) == np.sign(s))
    agg = dict(Anzahl=("abn_1d", "size"), Ø_1h=("r_1h", "mean"), Ø_1d=("abn_1d", "mean"),
               Ø_3d=("abn_3d", "mean"), Ø_Betrag_1d=("abn_1d", lambda x: x.abs().mean()),
               Ø_Stimmung=("sentiment", "mean"))
    by_cat = d.groupby("category").agg(**agg).sort_values("Anzahl", ascending=False)
    by_sent = d.groupby("stimmung", observed=True).agg(**agg)
    return dict(n=len(d), beta=beta, corr=float(s.corr(d["abn_1d"])), by_category=by_cat, by_sentiment=by_sent,
                hit_rate=float(d.loc[s.abs() > 0.15, "richtig"].mean()) if (s.abs() > 0.15).any() else np.nan)


def expected_news_drift(news: pd.DataFrame, stats: dict, last_price: float, prices: pd.DataFrame,
                        window_hours: float = 72) -> tuple[float, pd.DataFrame]:
    """Erwartete Restbewegung (über ~3 Tage) aus frischen Nachrichten, basierend auf den historischen Reaktionen.

    Pro frischer Nachricht: historische Ø-3-Tage-Reaktion der Kategorie/Stimmung (geschrumpft Richtung
    Stimmung * beta), abzüglich dessen, was seit Veröffentlichung schon passiert ist.
    """
    if news.empty or stats.get("n", 0) < 5:
        return 0.0, news.iloc[0:0]
    now = prices.index[-1]
    recent = news[news["time"] >= now - pd.Timedelta(hours=window_hours)].copy()
    if recent.empty:
        return 0.0, recent
    by_cat = stats["by_category"]
    preds = []
    for _, r in recent.iterrows():
        prior = stats["beta"] * r["sentiment"]
        if r["category"] in by_cat.index:
            g = by_cat.loc[r["category"]]
            n = g["Anzahl"]
            # Kategorie-Mittel in Richtung der Stimmung dieser Nachricht drehen
            cat_move = g["Ø_Betrag_1d"] * np.sign(r["sentiment"]) if abs(r["sentiment"]) > 0.15 else g["Ø_3d"]
            pred = (n * (0.5 * cat_move + 0.5 * prior) + 10 * prior) / (n + 10)
        else:
            pred = prior
        before = prices["Close"][prices.index < r["time"]]
        realized = last_price / before.iloc[-1] - 1 if len(before) else 0.0
        age_h = (now - r["time"]).total_seconds() / 3600
        preds.append((pred, realized, np.exp(-max(age_h, 0) / 48)))
    recent["erwartet_3d"] = [p for p, _, _ in preds]
    recent["bisher"] = [x for _, x, _ in preds]
    recent["gewicht"] = [w for _, _, w in preds]
    total_pred = np.average(recent["erwartet_3d"], weights=recent["gewicht"])
    realized = recent["bisher"].iloc[0]  # Bewegung seit der ältesten frischen Nachricht
    remaining = total_pred - realized if np.sign(total_pred) == np.sign(realized) else total_pred
    remaining = float(np.clip(remaining, -abs(total_pred), abs(total_pred)))
    return remaining, recent
