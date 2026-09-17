import os

TARGET = r"C:\Users\GushEx\Documents\IQOPTIONS BOT\BreakingBadV3\callisto_zone_copier.py"

CODE = """\
\"\"\"
callisto_zone_copier.py - Zone-Based Gold Copier for CallistoFx Live

Listens to CallistoFx Live Telegram channel for BUY/SELL zone messages.
When price enters the zone, waits for 1M/3M/5M bullish/bearish candle
confirmation then executes on IQ Option Marginal CFD (Gold/XAUUSD).

Typical CallistoFx zone message format:
  BUY ZONE: 4376.37 - 4361.06 level
  SELL ZONE: 4410.00 - 4425.00 level
  BUY ZONE INVALIDATED
  towards the 4435 level  <-- TP target

Logic:
  1. Parse zone from Telegram message
  2. Poll IQ Option Gold price every 15s
  3. When price enters zone, poll every 5s for 1M/3M/5M candle confirmation
  4. On confirmed candle: execute trade, SL beyond zone edge, TP at target level
  5. Zone is consumed after trade execution or expires after 8h
\"\"\"

import os
import sys
import re
import time
import logging
import asyncio
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

sys.path.append(os.getcwd())
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

from telethon import TelegramClient, events
from forex_mcp_client import IQForexMCPClient
from gsheet_logger import gsheet_logger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CallistoZoneCopier")

API_ID            = os.getenv("TELEGRAM_API_ID")
API_HASH          = os.getenv("TELEGRAM_API_HASH")
SESSION_NAME      = "user_desktop_session"
CALLISTO_CHANNEL  = -1002848189989   # CallistoFx Live

GOLD_ASSET_ID     = 74
GOLD_INSTRUMENT   = "mcfd.74"

POLL_INTERVAL     = 15   # seconds between price polls while waiting for zone touch
CONFIRM_INTERVAL  = 5    # seconds between polls once inside zone (candle confirm)
MAX_ZONE_WAIT_HRS = 8    # abandon zone watch after this many hours
SL_BUFFER         = 3.0  # USD buffer beyond zone edge for stop loss


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class CallistoZoneParser:
    \"\"\"Parses CallistoFx Live zone update messages.\"\"\"

    @staticmethod
    def is_zone_message(text: str) -> bool:
        upper = text.upper()
        return "BUY ZONE" in upper or "SELL ZONE" in upper

    @staticmethod
    def parse_zone(text: str) -> Optional[Dict[str, Any]]:
        upper = text.upper()

        # Invalidation
        if "BUY ZONE INVALIDATED" in upper or "SELL ZONE INVALIDATED" in upper:
            side = "BUY" if "BUY ZONE INVALIDATED" in upper else "SELL"
            return {"type": "INVALIDATE", "side": side, "raw_text": text}

        # Zone range  e.g.  BUY ZONE: 4376.37 - 4361.06
        m = re.search(
            r"(BUY|SELL)\\s+ZONE[:\\s]+([0-9]+(?:\\.[0-9]+)?)\\s*[-\\u2013]\\s*([0-9]+(?:\\.[0-9]+)?)",
            upper
        )
        if not m:
            return None

        side     = m.group(1)
        price_a  = float(m.group(2))
        price_b  = float(m.group(3))
        zone_high = max(price_a, price_b)
        zone_low  = min(price_a, price_b)

        # Target level: "towards the 4435 level" or "towards 4435"
        t = re.search(r"towards\\s+(?:the\\s+)?([0-9]+(?:\\.[0-9]+)?)", text, re.IGNORECASE)
        target = float(t.group(1)) if t else None

        # Fallback: any standalone price mentioned after zone that is outside the zone
        if not target:
            for lv_m in re.finditer(r"([0-9]{4,}(?:\\.[0-9]+)?)\\s*level", text, re.IGNORECASE):
                lv = float(lv_m.group(1))
                if not (zone_low - 1 <= lv <= zone_high + 1):
                    target = lv
                    break

        return {
            "type":      "ZONE",
            "side":      side,
            "zone_high": zone_high,
            "zone_low":  zone_low,
            "target":    target,
            "raw_text":  text
        }


# ---------------------------------------------------------------------------
# Copier
# ---------------------------------------------------------------------------

class CallistoZoneCopier:

    def __init__(self, account_type="training", lots=1.0, leverage=100,
                 sl_buffer=SL_BUFFER, confirm_tfs: List[int] = None):
        self.account_type   = account_type.lower()
        self.lots           = lots
        self.leverage       = leverage
        self.sl_buffer      = sl_buffer
        self.confirm_tfs    = confirm_tfs or [1, 3, 5]

        self.mcp            = IQForexMCPClient(base_url="https://marginal-cfd.mcp.iqoption.com")
        self.balance_id     = None
        self.tele_client    = None
        self.active_zone: Optional[Dict[str, Any]] = None
        self.zone_task: Optional[asyncio.Task] = None
        self.active_positions: Dict[int, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    def init_iq(self) -> bool:
        logger.info("Connecting to IQ Option Marginal CFD MCP...")
        if not self.mcp.initialize():
            logger.error("Failed to initialize IQ Option MCP session.")
            return False
        bal = self.mcp.get_training_balance() if self.account_type == "training" \\
              else self.mcp.get_real_balance()
        if not bal:
            logger.error("Failed to retrieve balance.")
            return False
        self.balance_id = bal["balance_id"]
        logger.info(
            f"IQ Option Connected | Balance: {self.balance_id} | "
            f"Equity: ${bal['equity']:.2f} | Free Margin: ${bal['free_margin']:.2f}"
        )
        return True

    # ------------------------------------------------------------------
    def get_gold_price(self) -> Dict[str, float]:
        try:
            p = self.mcp.calculate_order_size(
                asset_id=GOLD_ASSET_ID, balance_currency="USD",
                lots=self.lots, leverage=self.leverage
            )
            buy  = float(p.get("buy_price",  0.0))
            sell = float(p.get("sell_price", 0.0))
            return {"buy": buy, "sell": sell, "mid": (buy + sell) / 2}
        except Exception as e:
            logger.warning(f"Price fetch error: {e}")
            return {"buy": 0.0, "sell": 0.0, "mid": 0.0}

    # ------------------------------------------------------------------
    def check_confirmation(self, side: str) -> bool:
        \"\"\"True if the latest closed candle on any confirm TF agrees with side.\"\"\"
        for tf in self.confirm_tfs:
            try:
                candles = self.mcp.get_candles(
                    asset_id=GOLD_ASSET_ID,
                    period=tf * 60,
                    count=3
                )
                if not candles or len(candles) < 2:
                    continue
                closed = candles[-2]
                o = float(closed.get("open",  0))
                c = float(closed.get("close", 0))
                if o == 0 or c == 0:
                    continue
                if side == "BUY" and c > o:
                    logger.info(f"  Bullish {tf}M candle confirmed: O={o:.2f} C={c:.2f}")
                    return True
                if side == "SELL" and c < o:
                    logger.info(f"  Bearish {tf}M candle confirmed: O={o:.2f} C={c:.2f}")
                    return True
            except Exception as e:
                logger.warning(f"  Candle error ({tf}M): {e}")
        return False

    # ------------------------------------------------------------------
    def execute_zone_trade(self, zone: Dict[str, Any], exec_price: float):
        side      = zone["side"]
        zone_high = zone["zone_high"]
        zone_low  = zone["zone_low"]
        target    = zone.get("target")
        side_l    = side.lower()

        # SL just beyond far zone edge
        if side == "BUY":
            sl = round(zone_low  - self.sl_buffer, 2)
            tp = target if target else round(zone_high + (zone_high - zone_low) * 2, 2)
        else:
            sl = round(zone_high + self.sl_buffer, 2)
            tp = target if target else round(zone_low  - (zone_high - zone_low) * 2, 2)

        # Enforce IQ Option minimum $1.00 distance
        if abs(exec_price - sl) < 1.0:
            sl = round(exec_price - 1.5 if side == "BUY" else exec_price + 1.5, 2)
        if tp and abs(exec_price - tp) < 1.0:
            tp = round(exec_price + 1.5 if side == "BUY" else exec_price - 1.5, 2)

        logger.info("=" * 60)
        logger.info(f"CALLISTO ZONE TRADE | Gold {side} @ {exec_price:.2f}")
        logger.info(f"  Zone: {zone_low:.2f} - {zone_high:.2f}")
        logger.info(f"  SL:   {sl:.2f}  |  TP: {tp:.2f}")
        logger.info(f"  Lots: {self.lots} | Leverage: {self.leverage}x")
        logger.info("=" * 60)

        res = self.mcp.place_market_order(
            side=side_l,
            balance_id=self.balance_id,
            instrument_id=GOLD_INSTRUMENT,
            asset_id=GOLD_ASSET_ID,
            lots=self.lots,
            leverage=self.leverage,
            stop_loss=sl,
            take_profit=tp,
            is_margin_isolated=True,
            keep_position_open=False
        )

        if "order_id" in res:
            logger.info(f"ORDER FILLED | Order ID: #{res['order_id']}")
            time.sleep(2)
            self._sync_positions()
            self.active_zone = None   # zone consumed
        else:
            logger.error(f"Order failed: {res}")

    # ------------------------------------------------------------------
    async def _watch_zone(self, zone: Dict[str, Any]):
        \"\"\"Background task: monitors price against zone and fires trade on confirmation.\"\"\"
        side      = zone["side"]
        zone_high = zone["zone_high"]
        zone_low  = zone["zone_low"]
        deadline  = asyncio.get_event_loop().time() + MAX_ZONE_WAIT_HRS * 3600
        in_zone   = False

        logger.info(
            f"ZONE WATCH | {side} {zone_low:.2f}-{zone_high:.2f} | "
            f"Target: {zone.get('target')} | Expires {MAX_ZONE_WAIT_HRS}h"
        )

        while asyncio.get_event_loop().time() < deadline:
            prices = self.get_gold_price()
            mid = prices["mid"]
            if mid <= 0:
                await asyncio.sleep(POLL_INTERVAL)
                continue

            price_in_zone = zone_low <= mid <= zone_high

            if not in_zone and price_in_zone:
                in_zone = True
                logger.info(f"PRICE IN {side} ZONE @ {mid:.2f} | Watching for candle confirm...")

            if in_zone:
                if not price_in_zone:
                    logger.info(f"Price left zone ({mid:.2f}) — still watching...")
                    in_zone = False
                    await asyncio.sleep(POLL_INTERVAL)
                    continue
                # Inside zone — check candle confirmation
                if self.check_confirmation(side):
                    exec_price = prices["buy"] if side == "BUY" else prices["sell"]
                    logger.info(f"CONFIRMED! Executing {side} @ {exec_price:.2f}")
                    self.execute_zone_trade(zone, exec_price)
                    return
                await asyncio.sleep(CONFIRM_INTERVAL)
            else:
                dist_above = mid - zone_high if mid > zone_high else 0
                dist_below = zone_low - mid  if mid < zone_low  else 0
                dist_str = f"+{dist_above:.2f} above zone" if dist_above else f"{dist_below:.2f} below zone"
                logger.info(f"  Watching Gold @ {mid:.2f} | {dist_str}")
                await asyncio.sleep(POLL_INTERVAL)

        logger.warning(f"ZONE EXPIRED after {MAX_ZONE_WAIT_HRS}h — no entry. Zone cleared.")
        self.active_zone = None

    # ------------------------------------------------------------------
    def set_zone(self, zone: Dict[str, Any]):
        if self.zone_task and not self.zone_task.done():
            self.zone_task.cancel()
            logger.info("Previous zone watcher cancelled.")
        self.active_zone = zone
        self.zone_task   = asyncio.ensure_future(self._watch_zone(zone))

    def invalidate_zone(self, side: str):
        if self.active_zone and self.active_zone["side"] == side:
            if self.zone_task and not self.zone_task.done():
                self.zone_task.cancel()
            self.active_zone = None
            logger.info(f"{side} zone invalidated — watcher cancelled.")
        else:
            logger.info(f"Invalidation received for {side} zone but none is active.")

    # ------------------------------------------------------------------
    def _sync_positions(self):
        positions   = self.mcp.list_positions(balance_id=self.balance_id)
        current_ids = set()
        for p in positions:
            if p.get("asset_id") == GOLD_ASSET_ID:
                pid = p.get("position_id") or p.get("id")
                current_ids.add(pid)
                if pid not in self.active_positions:
                    self.active_positions[pid] = p
        closed = [pid for pid in self.active_positions if pid not in current_ids]
        for pid in closed:
            self._log_closed(pid, self.active_positions.pop(pid, {}))

    def _log_closed(self, position_id, pos_data):
        history = self.mcp.get_trade_history(balance_id=self.balance_id, limit=10)
        matched    = next((h for h in history if h.get("position_id") == position_id), None)
        pnl        = float(matched.get("pnl", 0.0))       if matched else 0.0
        exit_price = float(matched.get("close_price", 0.0)) if matched else 0.0
        reason     = matched.get("close_reason", "closed")  if matched else "closed"
        entry_px   = float(pos_data.get("open_price", 0.0))
        side       = pos_data.get("type", "buy").upper()
        logger.info(f"[TRADE CLOSED] Gold #{position_id} | PnL: ${pnl:.2f} | {reason}")
        try:
            bal_fn = self.mcp.get_training_balance if self.account_type == "training" \\
                     else self.mcp.get_real_balance
            eq = bal_fn().get("equity", 0.0)
        except Exception:
            eq = 0.0
        gsheet_logger.log_forex_margin_trade({
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "asset":        "Gold (XAUUSD)",
            "side":         side,
            "lots":         self.lots,
            "entry_price":  entry_px,
            "stop_loss":    pos_data.get("stop_lose_price", 0.0),
            "take_profit":  pos_data.get("take_profit_price", 0.0),
            "exit_price":   exit_price,
            "pnl":          pnl,
            "pips":         round(abs(exit_price - entry_px) * 10, 1),
            "risk_reward":  "Callisto Zone",
            "exit_reason":  reason,
            "position_id":  position_id,
            "balance_equity": eq
        })

    # ------------------------------------------------------------------
    async def start_telegram_listener(self):
        logger.info(f"Initializing listener for CallistoFx Live ({CALLISTO_CHANNEL})...")
        while True:
            try:
                self.tele_client = TelegramClient(
                    SESSION_NAME, API_ID, API_HASH,
                    connection_retries=None, retry_delay=5, auto_reconnect=True
                )
                await self.tele_client.connect()
                if not await self.tele_client.is_user_authorized():
                    logger.error("Telegram session not authorized. Run telegram_auth.py first.")
                    return
                me = await self.tele_client.get_me()
                logger.info(f"Telegram: {me.first_name} (@{me.username}) | Listening CallistoFx Live...")

                @self.tele_client.on(events.NewMessage(chats=CALLISTO_CHANNEL))
                async def handler(event):
                    text = event.message.text
                    if not text:
                        return
                    logger.info(f"\\n[CallistoFx] {text}\\n")
                    if not CallistoZoneParser.is_zone_message(text):
                        logger.info("Not a zone message — ignoring.")
                        return
                    parsed = CallistoZoneParser.parse_zone(text)
                    if not parsed:
                        logger.info("Could not parse zone from message.")
                        return
                    if parsed["type"] == "INVALIDATE":
                        self.invalidate_zone(parsed["side"])
                    elif parsed["type"] == "ZONE":
                        logger.info(
                            f"NEW {parsed['side']} ZONE: "
                            f"{parsed['zone_low']:.2f} - {parsed['zone_high']:.2f} | "
                            f"Target: {parsed.get('target')}"
                        )
                        self.set_zone(parsed)

                await self.tele_client.run_until_disconnected()

            except (ConnectionError, OSError) as e:
                logger.warning(f"Network error: {e}. Reconnecting in 10s...")
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}. Reconnecting in 10s...")
                await asyncio.sleep(10)
            finally:
                if self.tele_client:
                    try: await self.tele_client.disconnect()
                    except Exception: pass


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CallistoFx Zone-Based Gold Copier")
    parser.add_argument("--account",   default="training", choices=["training", "regular"])
    parser.add_argument("--lots",      type=float, default=1.0)
    parser.add_argument("--leverage",  type=int,   default=100)
    parser.add_argument("--sl-buffer", type=float, default=SL_BUFFER,
                        help=f"USD buffer beyond zone edge for SL (default: {SL_BUFFER})")
    args = parser.parse_args()

    copier = CallistoZoneCopier(
        account_type=args.account,
        lots=args.lots,
        leverage=args.leverage,
        sl_buffer=args.sl_buffer
    )
    if copier.init_iq():
        asyncio.run(copier.start_telegram_listener())


if __name__ == "__main__":
    main()
"""

with open(TARGET, "w", encoding="utf-8") as f:
    f.write(CODE)
print(f"Written {len(CODE)} bytes to {TARGET}")
