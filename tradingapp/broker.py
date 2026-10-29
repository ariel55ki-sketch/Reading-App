"""Order an Alpaca senden (Bracket-Order: Einstieg + Ziel + Stop in einem)."""
import requests

PAPER = "https://paper-api.alpaca.markets"
LIVE = "https://api.alpaca.markets"


def bracket_order(symbol: str, qty: float, side: str, take_profit: float, stop_loss: float,
                  key: str, secret: str, paper: bool = True) -> dict:
    body = {
        "symbol": symbol.replace("-", "/") if symbol.upper().endswith("-USD") else symbol,
        "qty": str(qty),
        "side": side,
        "type": "market",
        "time_in_force": "gtc",
        "order_class": "bracket",
        "take_profit": {"limit_price": f"{take_profit:.2f}"},
        "stop_loss": {"stop_price": f"{stop_loss:.2f}"},
    }
    r = requests.post(f"{PAPER if paper else LIVE}/v2/orders", json=body, timeout=30,
                      headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
    r.raise_for_status()
    return r.json()


def account(key: str, secret: str, paper: bool = True) -> dict:
    r = requests.get(f"{PAPER if paper else LIVE}/v2/account", timeout=30,
                     headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
    r.raise_for_status()
    return r.json()
