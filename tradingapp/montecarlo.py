"""Monte-Carlo-Simulation: tausende Kursverläufe und die beste Kombination aus Ziel und Stop."""
import numpy as np
import pandas as pd

LAMBDA = 0.94  # RiskMetrics-EWMA für die Volatilität


def ewma_sigma(r: np.ndarray, lam: float = LAMBDA) -> np.ndarray:
    s2 = np.empty_like(r)
    s2[0] = np.var(r[: min(len(r), 50)])
    for i in range(1, len(r)):
        s2[i] = lam * s2[i - 1] + (1 - lam) * r[i - 1] ** 2
    return np.sqrt(s2)


def simulate(close: pd.Series, horizon: int, n_paths: int = 10000, drift_per_bar: float = 0.0,
             block: int = 5, seed: int | None = None) -> tuple[np.ndarray, float]:
    """Gefilterte historische Simulation mit Block-Bootstrap.

    Echte Renditen dieser Aktie werden durch ihre damalige Volatilität geteilt (so bleiben dicke Enden
    und Muster erhalten), in Blöcken neu zusammengesetzt und mit der heutigen, sich weiterentwickelnden
    Volatilität skaliert. Rückgabe: (Pfade relativ zum aktuellen Kurs, Form n_paths x horizon; sigma pro Kerze)
    """
    rng = np.random.default_rng(seed)
    r = np.diff(np.log(close.to_numpy()))
    r = r[np.isfinite(r)]
    sig = ewma_sigma(r)
    z = r / np.where(sig > 0, sig, np.nan)
    z = z[np.isfinite(z)][50:]
    z = (z - z.mean()) / z.std()
    sigma_now = float(np.sqrt(LAMBDA * sig[-1] ** 2 + (1 - LAMBDA) * r[-1] ** 2))

    n_blocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, len(z) - block, size=(n_paths, n_blocks))
    shocks = z[starts[..., None] + np.arange(block)].reshape(n_paths, -1)[:, :horizon]

    logp = np.zeros((n_paths, horizon))
    s2 = np.full(n_paths, sigma_now ** 2)
    acc = np.zeros(n_paths)
    for t in range(horizon):
        step = np.sqrt(s2) * shocks[:, t]
        acc += drift_per_bar - 0.5 * s2 + step
        logp[:, t] = acc
        s2 = LAMBDA * s2 + (1 - LAMBDA) * step ** 2
    return np.exp(logp), sigma_now


def _first_hit(mask: np.ndarray) -> np.ndarray:
    h = mask.shape[1]
    return np.where(mask.any(axis=1), mask.argmax(axis=1), h)


def optimize(paths: np.ndarray, direction: int, sigma: float, cost: float = 0.0005, min_p_tp: float = 0.35,
             multiples=(0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)) -> dict:
    """Testet alle Ziel/Stop-Kombinationen auf allen Pfaden (wer zuerst erreicht wird, entscheidet).

    Gewählt wird die Kombination mit dem höchsten Erwartungswert, bei der das Ziel in mindestens
    `min_p_tp` der Pfade wirklich erreicht wird (sonst wäre es kein kurzfristiger Swing-Trade).
    """
    pnl = direction * (paths - 1)
    h = pnl.shape[1]
    scale = sigma * np.sqrt(h)
    levels = np.array(multiples) * scale
    tp_hit = [_first_hit(pnl >= x) for x in levels]
    sl_hit = [_first_hit(pnl <= -x) for x in levels]
    final = pnl[:, -1]
    ev = np.zeros((len(levels), len(levels)))
    pw = np.zeros_like(ev)
    best, fallback = None, None
    for i, tp in enumerate(levels):
        for j, sl in enumerate(levels):
            win = tp_hit[i] < np.minimum(sl_hit[j], h)
            loss = (sl_hit[j] <= tp_hit[i]) & (sl_hit[j] < h)
            out = np.where(win, tp, np.where(loss, -sl, final)) - cost
            ev[i, j] = out.mean()
            pw[i, j] = (out > 0).mean()
            cand = dict(tp=tp, sl=sl, ev=ev[i, j], p_win=pw[i, j], p_tp=win.mean(), p_sl=loss.mean(),
                        p_timeout=1 - win.mean() - loss.mean(),
                        avg_bars=float(np.where(win | loss, np.minimum(tp_hit[i], sl_hit[j]), h).mean()))
            score = (cand["ev"], cand["p_win"])
            if fallback is None or score > (fallback["ev"], fallback["p_win"]):
                fallback = cand
            if cand["p_tp"] >= min_p_tp and (best is None or score > (best["ev"], best["p_win"])):
                best = cand
    best = best or fallback
    best.update(direction=direction, levels=levels, ev_grid=ev, pwin_grid=pw)
    return best


def summary(paths: np.ndarray) -> dict:
    final = paths[:, -1]
    q = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    return dict(p_up=float((final > 1).mean()), median=float(np.median(final) - 1),
                p5=float(np.percentile(final, 5) - 1), p95=float(np.percentile(final, 95) - 1), bands=q)
