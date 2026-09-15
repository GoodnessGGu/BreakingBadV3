"""
unified_copier.py - Unified Gold Signal Copier (Gold Pips Hunter + CallistoFx Zones)

Single process, single Telethon client, listens to BOTH channels simultaneously.
Avoids the SQLite session lock that occurs when running two separate Telethon processes.

Channels:
  - Gold Pips Hunter  (-1003679078163): direct entry signals -> execute immediately
  - CallistoFx Live   (-1002848189989): zone alerts -> watch price, confirm, then execute

Usage:
  python unified_copier.py --account training --lots 1.0 --leverage 100
"""

import os, sys, re, time, logging, asyncio, argparse
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

sys.path.append(os.getcwd())
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
load_dotenv()

from telethon import TelegramClient, events
from forex_mcp_client import IQForexMCPClient
from gsheet_logger import gsheet_logger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("UnifiedCopier")

API_ID   = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")
SESSION  = "user_desktop_session"

GOLD_PIPS_CHANNEL = -1003679078163
CALLISTO_CHANNEL  = -1002848189989
GOLD_ASSET_ID     = 74
GOLD_INSTRUMENT   = "mcfd.74"

POLL_INTERVAL     = 15
CONFIRM_INTERVAL  = 5
MAX_ZONE_WAIT_HRS = 8
SL_BUFFER         = 3.0

# ─────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────

class GoldPipsParser:
    @staticmethod
    def parse_signal(text):
        clean = text.lower()
        if not any(k in clean for k in ["gold", "xau", "xauusd"]):
            return None
        side = None
        if "buy" in clean or "long" in clean:
            side = "BUY"
        elif "sell" in clean or "short" in clean:
            side = "SELL"
        if not side:
            return None
        sl_m  = re.search(r"(?:sl|stop\s*loss)[\s:]*([0-9]+(?:\.[0-9]+)?)", clean)
        tp1_m = re.search(r"(?:tp1|take\s*profit\s*1?)[\s:]*([0-9]+(?:\.[0-9]+)?)", clean)
        tp2_m = re.search(r"(?:tp2|take\s*profit\s*2)[\s:]*([0-9]+(?:\.[0-9]+)?)", clean)
        tp3_m = re.search(r"(?:tp3|take\s*profit\s*3)[\s:]*([0-9]+(?:\.[0-9]+)?)", clean)
        tp_g  = re.search(r"(?:tp|take\s*profit)[\s:]*([0-9]+(?:\.[0-9]+)?)", clean)
        ent_m = re.search(r"(?:@|at|entry)[\s:]*([0-9]+(?:\.[0-9]+)?)(?:\s*-\s*([0-9]+(?:\.[0-9]+)?))?", clean)
        sl   = float(sl_m.group(1))  if sl_m  else None
        tp1  = float(tp1_m.group(1)) if tp1_m else (float(tp_g.group(1)) if tp_g else None)
        tp2  = float(tp2_m.group(1)) if tp2_m else None
        tp3  = float(tp3_m.group(1)) if tp3_m else None
        emin = float(ent_m.group(1)) if ent_m else None
        emax = float(ent_m.group(2)) if (ent_m and ent_m.group(2)) else emin
        return {"type": "SIGNAL", "side": side, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
                "entry_min": emin, "entry_max": emax, "raw": text}

    @staticmethod
    def parse_instruction(text):
        clean = text.lower()
        if any(k in clean for k in ["breakeven", "break even", "move sl to entry", "sl to be"]):
            return {"type": "BREAKEVEN"}
        if any(k in clean for k in ["close all", "close positions", "close gold", "exit all"]):
            return {"type": "CLOSE_ALL"}
        return None


class CallistoParser:
    @staticmethod
    def is_zone_message(text):
        upper = text.upper()
        return "BUY ZONE" in upper or "SELL ZONE" in upper

    @staticmethod
    def parse_all(text):
        upper   = text.upper()
        results = []
        for inv in ("BUY", "SELL"):
            if inv + " ZONE INVALIDATED" in upper:
                results.append({"type": "INVALIDATE", "side": inv})
        for m in re.finditer(
            r"(?:NEW\s+)?(BUY|SELL)\s+ZONE[:\s]+([0-9]+(?:\.[0-9]+)?)\s*[-\u2013]\s*([0-9]+(?:\.[0-9]+)?)",
            upper
        ):
            side = m.group(1)
            a, b = float(m.group(2)), float(m.group(3))
            zhi, zlo = max(a, b), min(a, b)
            t = re.search(r"towards\s+(?:the\s+)?([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
            target = float(t.group(1)) if t else None
            if not target:
                for lm in re.finditer(r"([0-9]{4,}(?:\.[0-9]+)?)\s*level", text, re.IGNORECASE):
                    lv = float(lm.group(1))
                    if not (zlo - 1 <= lv <= zhi + 1):
                        target = lv
                        break
            results.append({"type": "ZONE", "side": side, "zone_high": zhi, "zone_low": zlo, "target": target})
        return results


# ─────────────────────────────────────────────
# Unified Copier
# ─────────────────────────────────────────────

class UnifiedCopier:
    def __init__(self, account_type="training", lots=1.0, leverage=100,
                 tp_target=1, max_slippage=4.0, sl_buffer=SL_BUFFER,
                 lookback_mins: int = 10):
        self.account_type = account_type.lower()
        self.lots         = lots
        self.leverage     = leverage
        self.tp_target    = tp_target
        self.max_slippage = max_slippage
        self.sl_buffer    = sl_buffer
        self.lookback_mins = lookback_mins
        self.processed_msg_ids = set()

        self.mcp         = IQForexMCPClient(base_url="https://marginal-cfd.mcp.iqoption.com")
        self.balance_id  = None
        self.tele        = None

        # State
        self.gold_positions: Dict[int, Dict] = {}   # Gold Pips active positions
        self.callisto_zone: Optional[Dict]   = None  # Active Callisto zone
        self.zone_task: Optional[asyncio.Task] = None
        self.callisto_positions: Dict[int, Dict] = {}

    # ── IQ init ──────────────────────────────
    def init_iq(self):
        logger.info("Connecting to IQ Option Marginal CFD MCP...")
        if not self.mcp.initialize():
            return False
        bal = self.mcp.get_training_balance() if self.account_type == "training" \
              else self.mcp.get_real_balance()
        if not bal:
            return False
        self.balance_id = bal["balance_id"]
        logger.info(f"IQ Connected | Equity: ${bal['equity']:.2f} | Free Margin: ${bal['free_margin']:.2f}")
        return True

    # ── Price ─────────────────────────────────
    def get_price(self):
        try:
            p = self.mcp.calculate_order_size(
                asset_id=GOLD_ASSET_ID, balance_currency="USD",
                lots=self.lots, leverage=self.leverage
            )
            buy  = float(p.get("buy_price",  0.0))
            sell = float(p.get("sell_price", 0.0))
            return {"buy": buy, "sell": sell, "mid": (buy + sell) / 2}
        except Exception as e:
            logger.warning(f"Price error: {e}")
            return {"buy": 0.0, "sell": 0.0, "mid": 0.0}

    # ── Gold Pips: execute signal ─────────────
    def execute_gold_signal(self, sig):
        side = sig["side"].lower()
        px   = self.get_price()
        cur  = px["buy"] if side == "buy" else px["sell"]
        if cur <= 0:
            logger.error("Cannot get Gold price.")
            return
        emin, emax = sig.get("entry_min"), sig.get("entry_max")
        if emin and emax:
            lo, hi = min(emin, emax) - self.max_slippage, max(emin, emax) + self.max_slippage
            if not (lo <= cur <= hi):
                logger.warning(f"Slippage skip: {cur:.2f} outside {lo:.2f}-{hi:.2f}")
                return
        tp = sig.get(f"tp{self.tp_target}") or sig.get("tp1")
        sl = sig.get("sl")
        if sl and abs(cur - sl) < 1.0:
            sl = round(cur - 1.5 if side == "buy" else cur + 1.5, 2)
        if tp and abs(cur - tp) < 1.0:
            tp = round(cur + 1.5 if side == "buy" else cur - 1.5, 2)
        logger.info(f"[GOLD PIPS] {side.upper()} @ {cur:.2f} | SL:{sl} TP:{tp}")
        res = self.mcp.place_market_order(
            side=side, balance_id=self.balance_id, instrument_id=GOLD_INSTRUMENT,
            asset_id=GOLD_ASSET_ID, lots=self.lots, leverage=self.leverage,
            stop_loss=sl, take_profit=tp, is_margin_isolated=True, keep_position_open=False
        )
        if "order_id" in res:
            logger.info(f"ORDER FILLED #{res['order_id']}")
            time.sleep(2)
            self._sync(self.gold_positions, "GoldPips")
        else:
            logger.error(f"Order failed: {res}")

    def apply_breakeven(self):
        self._sync(self.gold_positions, "GoldPips")
        for pid, p in list(self.gold_positions.items()):
            ep = float(p.get("open_price", 0.0))
            if ep > 0:
                self.mcp.change_position_stop_loss(position_id=pid, level=ep)
                logger.info(f"Breakeven set on #{pid} @ {ep:.2f}")
                p["stop_loss"] = ep

    def close_all_gold(self):
        self._sync(self.gold_positions, "GoldPips")
        for pid in list(self.gold_positions.keys()):
            self.mcp.close_position(position_id=pid)
            logger.info(f"Closed Gold Pips position #{pid}")
        time.sleep(2)
        self._sync(self.gold_positions, "GoldPips")

    # ── Callisto: zone trade ──────────────────
    def execute_zone_trade(self, zone, exec_price):
        side = zone["side"]
        zhi, zlo = zone["zone_high"], zone["zone_low"]
        target = zone.get("target")
        if side == "BUY":
            sl = round(zlo - self.sl_buffer, 2)
            tp = target if target else round(zhi + (zhi - zlo) * 2, 2)
        else:
            sl = round(zhi + self.sl_buffer, 2)
            tp = target if target else round(zlo - (zhi - zlo) * 2, 2)
        if abs(exec_price - sl) < 1.0:
            sl = round(exec_price - 1.5 if side == "BUY" else exec_price + 1.5, 2)
        if tp and abs(exec_price - tp) < 1.0:
            tp = round(exec_price + 1.5 if side == "BUY" else exec_price - 1.5, 2)
        logger.info(f"[CALLISTO] {side} @ {exec_price:.2f} | Zone:{zlo:.2f}-{zhi:.2f} | SL:{sl} TP:{tp}")
        res = self.mcp.place_market_order(
            side=side.lower(), balance_id=self.balance_id, instrument_id=GOLD_INSTRUMENT,
            asset_id=GOLD_ASSET_ID, lots=self.lots, leverage=self.leverage,
            stop_loss=sl, take_profit=tp, is_margin_isolated=True, keep_position_open=False
        )
        if "order_id" in res:
            logger.info(f"CALLISTO ORDER FILLED #{res['order_id']}")
            time.sleep(2)
            self._sync(self.callisto_positions, "Callisto")
            self.callisto_zone = None
        else:
            logger.error(f"Callisto order failed: {res}")

    def check_confirmation(self, side):
        # IQ Option CFD MCP supports: 60 (1m), 120 (2m), 300 (5m)
        for tf_sec in [60, 120, 300]:
            tf_label = f"{tf_sec // 60}M"
            try:
                candles = self.mcp.get_candles(asset_id=GOLD_ASSET_ID, size=tf_sec, count=3)
                if not candles or len(candles) < 2:
                    continue
                c = candles[-2]
                o_p = float(c.get("open", 0))
                c_p = float(c.get("close", 0))
                if o_p == 0 or c_p == 0:
                    continue
                if side == "BUY" and c_p > o_p:
                    logger.info(f"Bullish {tf_label} confirmed: O={o_p:.2f} C={c_p:.2f}")
                    return True
                if side == "SELL" and c_p < o_p:
                    logger.info(f"Bearish {tf_label} confirmed: O={o_p:.2f} C={c_p:.2f}")
                    return True
            except Exception as e:
                logger.warning(f"Candle error {tf_label}: {e}")
        return False

    async def _watch_zone(self, zone):
        side = zone["side"]
        zhi, zlo = zone["zone_high"], zone["zone_low"]
        deadline = asyncio.get_event_loop().time() + MAX_ZONE_WAIT_HRS * 3600
        in_zone  = False
        logger.info(f"[CALLISTO WATCH] {side} Zone {zlo:.2f}-{zhi:.2f} | Target:{zone.get('target')} | Expires {MAX_ZONE_WAIT_HRS}h")
        while asyncio.get_event_loop().time() < deadline:
            px  = self.get_price()
            mid = px["mid"]
            if mid <= 0:
                await asyncio.sleep(POLL_INTERVAL)
                continue
            in_z = zlo <= mid <= zhi
            if not in_zone and in_z:
                in_zone = True
                logger.info(f"[CALLISTO] Price entered {side} zone @ {mid:.2f} — watching for confirmation...")
            if in_zone:
                if not in_z:
                    in_zone = False
                    logger.info(f"Price left zone ({mid:.2f}) — still watching...")
                    await asyncio.sleep(POLL_INTERVAL)
                    continue
                if self.check_confirmation(side):
                    ep = px["buy"] if side == "BUY" else px["sell"]
                    self.execute_zone_trade(zone, ep)
                    return
                await asyncio.sleep(CONFIRM_INTERVAL)
            else:
                dist = (mid - zhi) if mid > zhi else (zlo - mid)
                logger.info(f"  Gold @ {mid:.2f} | {side} zone {zlo:.2f}-{zhi:.2f} | {dist:.2f} pts away")
                await asyncio.sleep(POLL_INTERVAL)
        logger.warning(f"[CALLISTO] Zone expired after {MAX_ZONE_WAIT_HRS}h.")
        self.callisto_zone = None

    def set_zone(self, zone):
        if self.zone_task and not self.zone_task.done():
            self.zone_task.cancel()
        self.callisto_zone = zone
        self.zone_task = asyncio.ensure_future(self._watch_zone(zone))

    def invalidate_zone(self, side):
        if self.callisto_zone and self.callisto_zone["side"] == side:
            if self.zone_task and not self.zone_task.done():
                self.zone_task.cancel()
            self.callisto_zone = None
            logger.info(f"[CALLISTO] {side} zone invalidated.")

    # ── Position tracking ─────────────────────
    def _sync(self, store, label):
        positions = self.mcp.list_positions(balance_id=self.balance_id)
        cur_ids   = set()
        for p in positions:
            if p.get("asset_id") == GOLD_ASSET_ID:
                pid = p.get("position_id") or p.get("id")
                cur_ids.add(pid)
                if pid not in store:
                    store[pid] = p
        closed = [pid for pid in store if pid not in cur_ids]
        for pid in closed:
            self._log_closed(pid, store.pop(pid, {}), label)

    def _log_closed(self, pid, pos, label):
        hist    = self.mcp.get_trade_history(balance_id=self.balance_id, limit=10)
        matched = next((h for h in hist if h.get("position_id") == pid), None)
        pnl     = float(matched.get("pnl", 0.0))       if matched else 0.0
        ep      = float(matched.get("close_price", 0.0)) if matched else 0.0
        reason  = matched.get("close_reason", "closed")  if matched else "closed"
        entry   = float(pos.get("open_price", 0.0))
        side    = pos.get("type", "buy").upper()
        logger.info(f"[{label}] Trade #{pid} closed | PnL:${pnl:.2f} | {reason}")
        try:
            bal_fn = self.mcp.get_training_balance if self.account_type == "training" \
                     else self.mcp.get_real_balance
            eq = bal_fn().get("equity", 0.0)
        except Exception:
            eq = 0.0
        gsheet_logger.log_forex_margin_trade({
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "asset":        "Gold (XAUUSD)",
            "side":         side,
            "lots":         self.lots,
            "entry_price":  entry,
            "stop_loss":    pos.get("stop_lose_price", 0.0),
            "take_profit":  pos.get("take_profit_price", 0.0),
            "exit_price":   ep,
            "pnl":          pnl,
            "pips":         round(abs(ep - entry) * 10, 1),
            "risk_reward":  label,
            "exit_reason":  reason,
            "position_id":  pid,
            "balance_equity": eq
        })

    # ── Telegram Listener ─────────────────────
    async def start(self):
        logger.info("Starting Unified Copier...")
        logger.info(f"  Gold Pips Hunter : {GOLD_PIPS_CHANNEL}")
        logger.info(f"  CallistoFx Live  : {CALLISTO_CHANNEL}")

        while True:
            try:
                self.tele = TelegramClient(
                    SESSION, API_ID, API_HASH,
                    connection_retries=None, retry_delay=5, auto_reconnect=True
                )
                await self.tele.connect()
                if not await self.tele.is_user_authorized():
                    logger.error("Session not authorized. Run telegram_auth.py first.")
                    return
                me = await self.tele.get_me()
                logger.info(f"Telegram: {me.first_name} (@{me.username})")
                logger.info("Listening to both channels...")

                # ── Gold Pips Hunter handler ──
                @self.tele.on(events.NewMessage(chats=GOLD_PIPS_CHANNEL))
                async def gold_handler(event):
                    if event.message.id in self.processed_msg_ids:
                        return
                    self.processed_msg_ids.add(event.message.id)

                    text = event.message.message
                    if not text:
                        return
                    logger.info(f"\n[Gold Pips Hunter]\n{text}\n")
                    instr = GoldPipsParser.parse_instruction(text)
                    if instr:
                        if instr["type"] == "BREAKEVEN":
                            self.apply_breakeven()
                        elif instr["type"] == "CLOSE_ALL":
                            self.close_all_gold()
                        return
                    sig = GoldPipsParser.parse_signal(text)
                    if sig:
                        logger.info(f"Gold signal: {sig['side']} | SL:{sig['sl']} TP1:{sig['tp1']}")
                        self.execute_gold_signal(sig)
                    else:
                        logger.info("Gold Pips: no actionable signal.")

                # ── CallistoFx Zone handler ──
                @self.tele.on(events.NewMessage(chats=CALLISTO_CHANNEL))
                async def callisto_handler(event):
                    if event.message.id in self.processed_msg_ids:
                        return
                    self.processed_msg_ids.add(event.message.id)

                    text = event.message.message
                    if not text:
                        return
                    logger.info(f"\n[CallistoFx Live]\n{text}\n")
                    if not CallistoParser.is_zone_message(text):
                        logger.info("Callisto: not a zone message.")
                        return
                    evts = CallistoParser.parse_all(text)
                    for ev in evts:
                        if ev["type"] == "INVALIDATE":
                            self.invalidate_zone(ev["side"])
                        elif ev["type"] == "ZONE":
                            logger.info(f"Callisto zone: {ev['side']} {ev['zone_low']:.2f}-{ev['zone_high']:.2f} Target:{ev.get('target')}")
                            self.set_zone(ev)

                # ── Startup Catch-Up Scan (Last N Minutes) ──
                if self.lookback_mins > 0:
                    logger.info(f"🔍 [STARTUP SCAN] Checking messages from the last {self.lookback_mins} minutes...")
                    now_utc = datetime.now(timezone.utc)

                    # 1. Check Gold Pips Hunter recent messages
                    try:
                        gp_recent = await self.tele.get_messages(GOLD_PIPS_CHANNEL, limit=30)
                        for m in reversed(gp_recent):
                            if not m.message:
                                continue
                            self.processed_msg_ids.add(m.id)
                            age_mins = (now_utc - m.date).total_seconds() / 60.0
                            if age_mins <= self.lookback_mins:
                                text = m.message
                                logger.info(f"📥 [RECENT GOLD PIPS MSG ({age_mins:.1f}m ago)] #{m.id}:\n{text}")
                                instr = GoldPipsParser.parse_instruction(text)
                                if instr:
                                    if instr["type"] == "BREAKEVEN":
                                        logger.info("🛡️ [STARTUP] Applying Breakeven instruction!")
                                        self.apply_breakeven()
                                    elif instr["type"] == "CLOSE_ALL":
                                        logger.info("🔒 [STARTUP] Applying Close All instruction!")
                                        self.close_all_gold()
                                    continue
                                sig = GoldPipsParser.parse_signal(text)
                                if sig:
                                    logger.info(f"🎯 [STARTUP CATCH-UP ({age_mins:.1f}m ago)] Found signal: {sig['side']} | SL: {sig['sl']} | TP1: {sig['tp1']}")
                                    self.execute_gold_signal(sig)
                    except Exception as e:
                        logger.warning(f"Could not scan recent Gold Pips messages on startup: {e}")

                    # 2. Check CallistoFx Live recent messages
                    try:
                        cal_recent = await self.tele.get_messages(CALLISTO_CHANNEL, limit=50)
                        for m in reversed(cal_recent):
                            if not m.message:
                                continue
                            self.processed_msg_ids.add(m.id)
                            age_mins = (now_utc - m.date).total_seconds() / 60.0
                            if age_mins <= self.lookback_mins:
                                if CallistoParser.is_zone_message(m.message):
                                    logger.info(f"📥 [RECENT CALLISTO MSG ({age_mins:.1f}m ago)] #{m.id}:\n{m.message}")
                                    evts = CallistoParser.parse_all(m.message)
                                    for ev in evts:
                                        if ev["type"] == "INVALIDATE":
                                            logger.info(f"❌ [STARTUP CATCH-UP] Invalidating {ev['side']} zone")
                                            self.invalidate_zone(ev["side"])
                                        elif ev["type"] == "ZONE":
                                            logger.info(f"🎯 [STARTUP CATCH-UP ({age_mins:.1f}m ago)] Loaded zone: {ev['side']} {ev['zone_low']:.2f}-{ev['zone_high']:.2f} Target:{ev.get('target')}")
                                            self.set_zone(ev)
                    except Exception as e:
                        logger.warning(f"Could not scan recent Callisto messages on startup: {e}")

                await self.tele.run_until_disconnected()

            except (ConnectionError, OSError) as e:
                logger.warning(f"Network error: {e}. Reconnecting in 10s...")
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error: {e}. Reconnecting in 10s...")
                await asyncio.sleep(10)
            finally:
                if self.tele:
                    try:
                        await self.tele.disconnect()
                    except Exception:
                        pass


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Unified Gold Copier (Gold Pips + CallistoFx)")
    p.add_argument("--account",    default="training", choices=["training", "regular"])
    p.add_argument("--lots",       type=float, default=1.0)
    p.add_argument("--leverage",   type=int,   default=100)
    p.add_argument("--tp-target",  type=int,   default=1, choices=[1, 2, 3])
    p.add_argument("--slippage",   type=float, default=4.0)
    p.add_argument("--sl-buffer",  type=float, default=SL_BUFFER)
    p.add_argument("--lookback-mins", type=int, default=10, help="Lookback window in minutes on startup to catch recent updates (default: 10)")
    args = p.parse_args()

    copier = UnifiedCopier(
        account_type=args.account,
        lots=args.lots,
        leverage=args.leverage,
        tp_target=args.tp_target,
        max_slippage=args.slippage,
        sl_buffer=args.sl_buffer,
        lookback_mins=args.lookback_mins
    )
    if copier.init_iq():
        asyncio.run(copier.start())

if __name__ == "__main__":
    main()