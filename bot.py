#!/usr/bin/env python3
"""
Comms routing:
- heartbeats webhook: bot ON/OFF + runtime errors
- alpha webhook: trade ENTER/EXIT notifications
"""

from __future__ import annotations

import logging
import math
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

# ============================
# edit me
# ============================
ALPACA_API_KEY = "key"
ALPACA_SECRET_KEY = "secert"
ALPACA_TRADING_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"

# Enter symbols to scalp.
SYMBOLS = ["AAPL"] # Enter Any ticker you like
ORDER_QTY = 1

# Strategy settings. - Can be edited accordingly 
LOOKBACK = 3
ERROR_MARGIN = 0.0015
TIMEFRAME = "1Min"
BAR_LIMIT = 150
POLL_SECONDS = 20

# Slack webhooks (edit)
HEARTBEAT_COMMS_WEBHOOK = "webhook1"
ALPHA_COMMS_WEBHOOK = "webhook2"


@dataclass
class PivotPoint:
    ts: str
    idx: int
    price: float
    kind: str  # "high" | "low"


class WebhookNotifier:
    def __init__(self, heartbeat_url: str, alpha_url: str) -> None:
        self.heartbeat_url = heartbeat_url
        self.alpha_url = alpha_url

    @staticmethod
    def _post(url: str, text: str) -> None:
        if not url.startswith("https://hooks.slack.com/services/"):
            return
        resp = requests.post(url, json={"text": text}, timeout=15)
        resp.raise_for_status()

    def heartbeat(self, text: str) -> None:
        try:
            self._post(self.heartbeat_url, text)
        except Exception as exc:
            logging.error("Heartbeat webhook failed: %s", exc)

    def alpha(self, text: str) -> None:
        try:
            self._post(self.alpha_url, text)
        except Exception as exc:
            logging.error("Alpha webhook failed: %s", exc)


class AlpacaClient:
    def __init__(self, api_key: str, secret_key: str, trading_url: str, data_url: str) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self.trading_url = trading_url.rstrip("/")
        self.data_url = data_url.rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.secret_key}

    def get_bars(self, symbol: str, timeframe: str, limit: int) -> List[dict]:
        url = f"{self.data_url}/v2/stocks/{symbol}/bars"
        params = {"timeframe": timeframe, "limit": limit, "feed": "iex"}
        r = requests.get(url, headers=self._headers(), params=params, timeout=15)
        r.raise_for_status()
        payload = r.json()
        bars = payload.get("bars")
        if bars is None:
            logging.warning("No bar payload returned for %s; using empty list.", symbol)
            return []
        if not isinstance(bars, list):
            logging.warning("Unexpected bars payload type for %s: %s", symbol, type(bars).__name__)
            return []
        return bars

    def get_position_qty(self, symbol: str) -> float:
        url = f"{self.trading_url}/v2/positions/{symbol}"
        r = requests.get(url, headers=self._headers(), timeout=15)
        if r.status_code == 404:
            return 0.0
        r.raise_for_status()
        return float(r.json().get("qty", 0.0))

    def submit_market_order(self, symbol: str, side: str, qty: int) -> dict:
        url = f"{self.trading_url}/v2/orders"
        payload = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }
        r = requests.post(url, headers={**self._headers(), "Content-Type": "application/json"}, json=payload, timeout=15)
        r.raise_for_status()
        return r.json()


class PatternScalperBot:
    def __init__(self, client: AlpacaClient, notifier: WebhookNotifier) -> None:
        self.client = client
        self.notifier = notifier
        self.running = True
        self.state: Dict[str, Dict[str, object]] = {
            s: {"pivots": [], "seen": set()} for s in SYMBOLS
        }

    @staticmethod
    def _pivot_high(highs: List[float], i: int, lb: int) -> bool:
        v = highs[i]
        left = highs[i - lb : i]
        right = highs[i + 1 : i + lb + 1]
        return all(v > x for x in left) and all(v >= x for x in right)

    @staticmethod
    def _pivot_low(lows: List[float], i: int, lb: int) -> bool:
        v = lows[i]
        left = lows[i - lb : i]
        right = lows[i + 1 : i + lb + 1]
        return all(v < x for x in left) and all(v <= x for x in right)

    def _extract_new_pivots(self, symbol: str, bars: Optional[List[dict]]) -> List[PivotPoint]:
        if not bars:
            return []

        if len(bars) < LOOKBACK * 2 + 1:
            return []

        highs = [float(b["h"]) for b in bars]
        lows = [float(b["l"]) for b in bars]
        seen = self.state[symbol]["seen"]
        pivots: List[PivotPoint] = []

        for i in range(LOOKBACK, len(bars) - LOOKBACK):
            ts = bars[i]["t"]
            if ts in seen:
                continue
            if self._pivot_high(highs, i, LOOKBACK):
                pivots.append(PivotPoint(ts=ts, idx=i, price=highs[i], kind="high"))
                seen.add(ts)
            elif self._pivot_low(lows, i, LOOKBACK):
                pivots.append(PivotPoint(ts=ts, idx=i, price=lows[i], kind="low"))
                seen.add(ts)

        return pivots

    @staticmethod
    def _is_close(a: float, b: float, margin: float) -> bool:
        return math.fabs(a - b) <= (a * margin)

    def _signal_from_state(self, symbol: str, pivot: PivotPoint) -> Optional[str]:
        pivots: List[PivotPoint] = self.state[symbol]["pivots"]
        pivots.append(pivot)
        if len(pivots) > 3:
            pivots.pop(0)
        if len(pivots) < 3:
            return None

        p3, p2, p1 = pivots[0], pivots[1], pivots[2]

        is_double_bottom = pivot.kind == "low" and self._is_close(p1.price, p3.price, ERROR_MARGIN)
        is_double_top = pivot.kind == "high" and self._is_close(p1.price, p3.price, ERROR_MARGIN)
        is_head_shoulders = pivot.kind == "high" and (p2.price > p1.price and p2.price > p3.price)
        is_inv_hs = pivot.kind == "low" and (p2.price < p1.price and p2.price < p3.price)

        if is_double_top or is_head_shoulders:
            return "SELL"
        if is_double_bottom or is_inv_hs:
            return "BUY"
        return None

    def _send_alpha(self, symbol: str, action: str, side: str, qty: int, reason: str, order_id: str = "n/a") -> None:
        self.notifier.alpha(
            "\n".join(
                [
                    "*Alpha Comms*",
                    f"Symbol: `{symbol}`",
                    f"Action: *{action}*",
                    f"Side: `{side}`",
                    f"Qty: `{qty}`",
                    f"Reason: {reason}",
                    f"Order ID: `{order_id}`",
                    f"Time (UTC): {datetime.now(timezone.utc).isoformat()}",
                ]
            )
        )

    def _execute_trade(self, symbol: str, signal: str, reason: str) -> None:
        pos_qty = self.client.get_position_qty(symbol)
        side = "buy" if signal == "BUY" else "sell"

        if signal == "BUY" and pos_qty < 0:
            exit_qty = int(abs(pos_qty))
            exit_order = self.client.submit_market_order(symbol, "buy", exit_qty)
            self._send_alpha(symbol, "EXIT", "buy", exit_qty, "Close short before long entry", exit_order.get("id", "n/a"))
        elif signal == "SELL" and pos_qty > 0:
            exit_qty = int(abs(pos_qty))
            exit_order = self.client.submit_market_order(symbol, "sell", exit_qty)
            self._send_alpha(symbol, "EXIT", "sell", exit_qty, "Close long before short entry", exit_order.get("id", "n/a"))

        entry_order = self.client.submit_market_order(symbol, side, ORDER_QTY)
        self._send_alpha(symbol, "ENTER", side, ORDER_QTY, reason, entry_order.get("id", "n/a"))
        logging.info("%s %s qty=%s (%s)", signal, symbol, ORDER_QTY, reason)

    def stop(self) -> None:
        self.running = False

    def run(self) -> None:
        self.notifier.heartbeat(":satellite: Scalper bot started.")
        while self.running:
            for symbol in SYMBOLS:
                try:
                    bars = self.client.get_bars(symbol, TIMEFRAME, BAR_LIMIT)
                    for pivot in self._extract_new_pivots(symbol, bars):
                        signal = self._signal_from_state(symbol, pivot)
                        if signal:
                            self._execute_trade(symbol, signal, f"{pivot.kind} pivot pattern")
                except Exception as exc:
                    logging.exception("Error processing %s: %s", symbol, exc)
                    self.notifier.heartbeat(f":warning: Bot error on {symbol}: `{exc}`")
            time.sleep(POLL_SECONDS)
        self.notifier.heartbeat(":octagonal_sign: Scalper bot stopped.")


def _install_signal_handlers(bot: PatternScalperBot) -> None:
    def _handler(signum, frame):
        logging.info("Signal %s received, stopping bot...", signum)
        bot.stop()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if "ENTER_YOUR_ALPACA" in ALPACA_API_KEY or "ENTER_YOUR_ALPACA" in ALPACA_SECRET_KEY:
        raise SystemExit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in bot.py before running.")

    client = AlpacaClient(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        trading_url=ALPACA_TRADING_URL,
        data_url=ALPACA_DATA_URL,
    )
    notifier = WebhookNotifier(HEARTBEAT_COMMS_WEBHOOK, ALPHA_COMMS_WEBHOOK)
    bot = PatternScalperBot(client, notifier)
    _install_signal_handlers(bot)

    try:
        bot.run()
    except KeyboardInterrupt:
        bot.stop()
        notifier.heartbeat(":octagonal_sign: Scalper bot stopped (KeyboardInterrupt).")
        sys.exit(0)
