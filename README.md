# Swing-Trading-Radar

Kurzfristige Trading-Empfehlungen (Stunden bis wenige Tage) für deine Watchlist: Aktien und Krypto.

## Starten

1. Python 3.10+ installieren (python.org).
2. Ordner entpacken, `.env.example` nach `.env` kopieren und Keys eintragen (optional).
3. **Windows:** `start.bat` doppelklicken. **Mac/Linux:** `./start.sh`
4. Im Browser öffnet sich die App. Watchlist eintragen, „Analysieren“ klicken.

Ohne Keys läuft alles mit Yahoo Finance. Mit Keys werden Daten und News deutlich besser.

| Dienst | Wofür | Kostenlos? |
|---|---|---|
| [Finnhub](https://finnhub.io/register) | News-Historie je Firma (1 Jahr) | ja |
| [Alpaca](https://app.alpaca.markets/signup) | Kurse, News seit 2015, Orders (Paper & Live) | ja |
| [Polygon.io](https://polygon.io) | lange Intraday-Historie | Basis ja |
| [NewsAPI](https://newsapi.org/register) | zusätzliche Nachrichtenquellen (30 Tage) | ja |

## Was die App macht

1. **Kurse** holen (Polygon → Alpaca → Yahoo), in 15-Min-, Stunden- oder Tageskerzen.
2. **7 Trader-Strategien** prüfen, ob gerade ein Signal aktiv ist: Connors RSI(2), Bollinger Squeeze,
   MACD-Momentum, VWAP-Pullback, Opening Range Breakout, Minervini/O'Neil-Breakout, Turtle-Donchian.
   Jede wird auf der Historie *genau dieses Werts* getestet. Nur Strategien, die dort einen Vorteil hatten,
   fließen in die Prognose ein, gewichtet nach Anzahl Trades und Signifikanz.
3. **News**: alle Nachrichten aus den verbundenen Quellen, Stimmung (VADER + Finanz-Wörterbuch), Kategorie
   (Quartalszahlen, Analysten, Übernahmen, Recht …). Für jede Nachricht wird die Kursreaktion nach 1 Std.,
   1 Tag und 3 Tagen gemessen, auch bereinigt um den Markt (SPY bzw. BTC). Daraus lernt die App, wie stark
   dieser Wert auf welche Art von News reagiert, und schätzt, wie viel Bewegung frische News noch auslösen.
4. **Monte Carlo**: 10.000+ Kursverläufe per gefilterter historischer Simulation (echte Renditen dieses Werts,
   Block-Bootstrap, aktuelle Volatilität mit EWMA-Dynamik) plus Drift aus Signalen und News.
   Für 81 Kombinationen aus Ziel und Stop wird gezählt, was zuerst erreicht wird. Gewählt wird die Kombination
   mit dem höchsten Erwartungswert, bei der das Ziel in mindestens 35 % der Pfade erreicht wird.
5. **Empfehlung**: KAUFEN / SHORT / WARTEN mit Einstieg, Ziel, Stop, Gewinnchance und Erwartung pro Trade.
   Mit Alpaca-Keys kannst du die Bracket-Order direkt aus der App senden (Paper-Trading ist voreingestellt).

## Projektstruktur

```
app.py                     Streamlit-Oberfläche
tradingapp/data.py         Kursdaten
tradingapp/strategies.py   Indikatoren, Strategien, Backtest
tradingapp/news.py         News-Quellen, Stimmung, Kursreaktions-Analyse
tradingapp/montecarlo.py   Simulation und Ziel/Stop-Optimierung
tradingapp/recommend.py    Zusammenführung zur Empfehlung
tradingapp/broker.py       Alpaca-Orders
tests/test_offline.py      Test mit synthetischen Daten
```
