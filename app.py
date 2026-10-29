"""Streamlit-Oberfläche. Start: streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tradingapp import broker, config, recommend

config.load_env()
st.set_page_config(page_title="Swing-Trading-Radar", layout="wide")
config.load_streamlit_secrets()
st.title("📈 Swing-Trading-Radar")

with st.sidebar:
    st.header("Watchlist")
    watch = st.text_area("Symbole (Komma-getrennt)", "AAPL, NVDA, TSLA, AMD, BTC-USD")
    interval = st.selectbox("Kerzen", ["1h", "15m", "1d"], index=0)
    horizon = st.slider("Horizont (Kerzen)", 5, 60, recommend.DEFAULT_HORIZON[interval])
    n_paths = st.select_slider("Simulationen", [2000, 5000, 10000, 25000, 50000], 10000)
    news_days = st.slider("News-Historie (Tage)", 30, 730, 365)
    allow_short = st.checkbox("Short-Trades erlauben", True)
    with st.expander("API-Keys"):
        env = config.get_keys()
        overrides = {k: st.text_input(label, env[k], type="password") for k, label in config.KEY_NAMES.items()}
    keys = config.get_keys(overrides)
    connected = [label for k, label in config.KEY_NAMES.items() if keys[k]]
    st.caption("Verbunden: " + (", ".join(connected) if connected else "nur Yahoo (kostenlos)"))
    run = st.button("Analysieren", type="primary", width="stretch")


@st.cache_data(ttl=600, show_spinner=False)
def cached_analyze(sym, interval, horizon, n_paths, news_days, allow_short, keys_tuple):
    return recommend.analyze(sym, interval, horizon, n_paths, dict(keys_tuple), news_days, allow_short)


if run:
    syms = [s.strip().upper() for s in watch.replace(";", ",").split(",") if s.strip()]
    results, prog = {}, st.progress(0.0)
    for i, s in enumerate(syms):
        prog.progress(i / len(syms), f"Analysiere {s} …")
        try:
            results[s] = cached_analyze(s, interval, horizon, n_paths, news_days, allow_short, tuple(sorted(keys.items())))
        except Exception as e:
            st.error(f"{s}: {e}")
    prog.empty()
    st.session_state["results"] = results

results = st.session_state.get("results", {})
if not results:
    st.info("Watchlist eintragen und auf **Analysieren** klicken.")
    st.stop()

# ---------- Übersicht ----------
order = {"KAUFEN": 0, "SHORT": 1, "WARTEN": 2}
ov = pd.DataFrame([recommend.overview_row(a) for a in results.values()])
pct_cols = ["Gewinnchance", "Ziel zuerst", "Erwartung/Trade", "News-Impuls"]
ov[pct_cols] = ov[pct_cols] * 100
ov = ov.sort_values(["Empfehlung", "Erwartung/Trade"], key=lambda c: c.map(order) if c.name == "Empfehlung" else -c)
st.subheader("Empfehlungen")
st.dataframe(ov, hide_index=True, width="stretch", column_config={
    "Gewinnchance": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
    "Ziel zuerst": st.column_config.NumberColumn(format="%.0f%%"),
    "Erwartung/Trade": st.column_config.NumberColumn(format="%.2f%%"),
    "News-Impuls": st.column_config.NumberColumn(format="%.2f%%"),
})

# ---------- Detail ----------
sym = st.selectbox("Details", list(results))
a = results[sym]
t = a.trade
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Empfehlung", a.action)
c2.metric("Einstieg", f"{a.entry:,.2f}")
c3.metric("Ziel", f"{a.target:,.2f}", f"{a.trade['direction'] * t['tp'] * 100:+.2f}%")
c4.metric("Stop", f"{a.stop:,.2f}", f"{-a.trade['direction'] * t['sl'] * 100:+.2f}%")
c5.metric("Gewinnchance", f"{t['p_win'] * 100:.0f}%", f"Erw. {t['ev'] * 100:+.2f}%/Trade")
st.caption(f"Ziel zuerst: {t['p_tp']:.0%} · Stop zuerst: {t['p_sl']:.0%} · Zeitablauf: {t['p_timeout']:.0%} · "
           f"Ø Haltedauer: {t['avg_bars']:.0f} Kerzen · Signalvorteil {a.signal_edge * 100:+.2f}% · "
           f"News-Impuls {a.news_drift * 100:+.2f}% · Daten: {a.source}")

tab_chart, tab_strat, tab_news, tab_mc, tab_order = st.tabs(["Chart", "Strategien", "News & Reaktionen", "Simulation", "Order"])

with tab_chart:
    d = a.prices.iloc[-300:]
    fig = go.Figure(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name=sym))
    for col, color in [("SMA50", "#f39c12"), ("BB_up", "#95a5a6"), ("BB_lo", "#95a5a6"), ("VWAP", "#8e44ad")]:
        fig.add_scatter(x=d.index, y=d[col], name=col, line=dict(width=1, color=color))
    for name in [c for c in d.columns if c.startswith("sig::")]:
        s = d[name]
        for direction, sym_, col in [(1, "triangle-up", "green"), (-1, "triangle-down", "red")]:
            m = s == direction
            if m.any():
                fig.add_scatter(x=d.index[m], y=d["Close"][m], mode="markers", name=name[5:],
                                marker=dict(symbol=sym_, size=9, color=col), legendgroup=name, showlegend=False)
    if a.action != "WARTEN":
        fig.add_hline(a.target, line_color="green", line_dash="dash", annotation_text="Ziel")
        fig.add_hline(a.stop, line_color="red", line_dash="dash", annotation_text="Stop")
    for _, n in a.news[a.news["time"] >= d.index[0]].iterrows():
        fig.add_vline(n["time"], line_width=1, line_color="rgba(52,152,219,0.25)")
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")

with tab_strat:
    st.write("Jede Strategie wurde auf der Historie **dieses** Werts getestet (Rendite nach Horizont in Signalrichtung).")
    s = a.strat_stats.copy()
    s["Aktiv"] = s["Aktiv"].map({1: "↑ Long", -1: "↓ Short", 0: ""})
    s["Trefferquote"] *= 100
    s["Ø_Rendite"] *= 100
    st.dataframe(s, hide_index=True, width="stretch", column_config={
        "Trefferquote": st.column_config.NumberColumn(format="%.1f%%"),
        "Ø_Rendite": st.column_config.NumberColumn("Ø Rendite", format="%.2f%%"),
        "t_Wert": st.column_config.NumberColumn(format="%.2f"),
    })

with tab_news:
    ns = a.news_stats
    st.write(f"**{len(a.news)}** Nachrichten · Reaktion messbar bei **{ns['n']}** · "
             f"Stimmung↔Kurs-Korrelation: **{ns.get('corr', float('nan')):.2f}** · "
             f"Richtung stimmt in **{ns.get('hit_rate', float('nan')):.0%}** der Fälle")
    if not ns["by_category"].empty:
        bc = ns["by_category"]
        fig = go.Figure([go.Bar(x=bc.index, y=bc["Ø_1d"] * 100, name="Ø 1 Tag (vs. Markt)"),
                         go.Bar(x=bc.index, y=bc["Ø_3d"] * 100, name="Ø 3 Tage (vs. Markt)")])
        fig.update_layout(barmode="group", yaxis_title="%", height=350, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(bc, width="stretch")
        dd = a.news.dropna(subset=["abn_1d"])
        sc = go.Figure(go.Scatter(x=dd["sentiment"], y=dd["abn_1d"] * 100, mode="markers", text=dd["headline"],
                                  marker=dict(color=dd["abn_1d"], colorscale="RdYlGn", cmid=0)))
        sc.update_layout(xaxis_title="Stimmung der Nachricht", yaxis_title="Reaktion 1 Tag vs. Markt (%)", height=350)
        st.plotly_chart(sc, width="stretch")
    if not a.recent_news.empty:
        st.subheader("Frische Nachrichten und erwartete Wirkung")
        st.dataframe(a.recent_news[["time", "headline", "category", "sentiment", "erwartet_3d", "bisher"]],
                     hide_index=True, width="stretch")
    st.subheader("Alle Nachrichten")
    st.dataframe(a.news.sort_values("time", ascending=False)[
        ["time", "headline", "source", "category", "sentiment", "r_1h", "abn_1d", "abn_3d", "url"]],
        hide_index=True, width="stretch",
        column_config={"url": st.column_config.LinkColumn("Link")})

with tab_mc:
    b = a.mc["bands"] * a.price
    x = np.arange(1, a.horizon + 1)
    fig = go.Figure()
    for p in a.paths_sample[:60]:
        fig.add_scatter(x=x, y=p * a.price, mode="lines", line=dict(width=0.5, color="rgba(120,120,120,0.25)"), showlegend=False)
    fig.add_scatter(x=x, y=b[4], line=dict(width=0), showlegend=False)
    fig.add_scatter(x=x, y=b[0], fill="tonexty", fillcolor="rgba(52,152,219,0.15)", line=dict(width=0), name="5–95 %")
    fig.add_scatter(x=x, y=b[3], line=dict(width=0), showlegend=False)
    fig.add_scatter(x=x, y=b[1], fill="tonexty", fillcolor="rgba(52,152,219,0.3)", line=dict(width=0), name="25–75 %")
    fig.add_scatter(x=x, y=b[2], line=dict(color="#2980b9", width=2), name="Median")
    if a.action != "WARTEN":
        fig.add_hline(a.target, line_color="green", line_dash="dash")
        fig.add_hline(a.stop, line_color="red", line_dash="dash")
    fig.update_layout(height=450, xaxis_title="Kerzen ab jetzt", yaxis_title="Kurs", margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.write(f"Wahrscheinlichkeit höher zu schließen: **{a.mc['p_up']:.0%}** · Median {a.mc['median'] * 100:+.2f}% · "
             f"5 %/95 %: {a.mc['p5'] * 100:+.2f}% / {a.mc['p95'] * 100:+.2f}%")
    for label, opt in [("Long", a.long), ("Short", a.short)]:
        lv = [f"{v * 100:.2f}%" for v in opt["levels"]]
        hm = go.Figure(go.Heatmap(z=opt["ev_grid"] * 100, x=lv, y=lv, colorscale="RdYlGn", zmid=0,
                                  colorbar=dict(title="Erw. %")))
        hm.update_layout(title=f"{label}: Erwartungswert je Ziel (Zeile) × Stop (Spalte)", height=380,
                         xaxis_title="Stop", yaxis_title="Ziel")
        st.plotly_chart(hm, width="stretch")

with tab_order:
    if not (keys["ALPACA_API_KEY"] and keys["ALPACA_SECRET_KEY"]):
        st.info("Alpaca-Keys eintragen, um Orders direkt aus der App zu senden.")
    elif a.action == "WARTEN":
        st.write("Aktuell kein Einstieg.")
    else:
        paper = st.toggle("Paper-Trading (Spielgeld)", True)
        budget = st.number_input("Einsatz ($)", 10.0, 1e7, 1000.0, step=100.0)
        qty = round(budget / a.price, 4 if "-USD" in sym else 0) or 1
        side = "buy" if a.action == "KAUFEN" else "sell"
        st.write(f"Bracket-Order: **{side} {qty} {sym}** · Ziel {a.target:.2f} · Stop {a.stop:.2f}")
        if st.button("Order senden"):
            try:
                st.success(broker.bracket_order(sym, qty, side, a.target, a.stop,
                                                keys["ALPACA_API_KEY"], keys["ALPACA_SECRET_KEY"], paper))
            except Exception as e:
                st.error(str(e))

st.caption("Wahrscheinlichkeiten stammen aus historischen Daten dieses Werts.")
