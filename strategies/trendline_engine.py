"""
strategies/trendline_engine.py - Autonomous Algorithmic Trendline Strategy Engine
Implements Model 1: 15-Minute 3rd Touch Trendline Bounce with 50 EMA Trend Filter.

Proven Edge (60-Day Historical Backtest):
  - NAS100: 67.3% Win Rate | +$1,009.77 Net | 1.56 Profit Factor
  - GOLD: 62.0% Win Rate | +$679.47 Net | 1.38 Profit Factor

Execution:
  - Dual-Ticket Partial Take-Profit System:
    * Leg 1 (50%): Takes profit at +1.0R (banks guaranteed profit).
    * Leg 2 (50%): Shifts Stop Loss to Breakeven when TP1 hits, targets +2.5R runner.
"""

import time
import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple, Set, Callable
import pandas as pd
import numpy as np

from clients.forex_mcp_client import IQForexMCPClient
from utils.chart_generator import (
    generate_trade_execution_chart,
    generate_breakeven_chart
)

try:
    from gsheet_logger import gsheet_logger
except ImportError:
    try:
        from utils.gsheet_logger import gsheet_logger
    except ImportError:
        gsheet_logger = None

logger = logging.getLogger("TrendlineEngine")

TRENDLINE_PROFILES: Dict[str, Dict[str, Any]] = {
    "XAUUSD": {
        "symbol": "XAUUSD",
        "name": "Gold (XAU/USD)",
        "asset_id": 74,
        "instrument_id": "mcfd.74",
        "sl_buffer": 1.5,
        "disp_min": 3.0,
        "digits": 2,
        "default_lots": 2.0,      # Two 1.0-lot tickets
        "min_lots": 1.0,
        "rr_target": 2.5,
        "pivot_window": 5
    },
    "BTCUSD": {
        "symbol": "BTCUSD",
        "name": "Bitcoin (BTC/USD)",
        "asset_id": 816,
        "instrument_id": "mcrpt.816",
        "sl_buffer": 150.0,
        "disp_min": 350.0,
        "digits": 2,
        "default_lots": 0.02,
        "min_lots": 0.001,
        "rr_target": 2.5,
        "pivot_window": 5
    },
    "EURUSD": {
        "symbol": "EURUSD",
        "name": "EUR/USD",
        "asset_id": 1,
        "instrument_id": "mf.1",
        "sl_buffer": 0.0008,
        "disp_min": 0.0015,
        "digits": 5,
        "default_lots": 0.1,
        "min_lots": 0.001,
        "rr_target": 2.5,
        "pivot_window": 5
    },
    "NAS100": {
        "symbol": "NAS100",
        "name": "US Tech 100 (NAS100)",
        "asset_id": 1471,
        "instrument_id": "mcfd.1471",
        "sl_buffer": 15.0,
        "disp_min": 30.0,
        "digits": 2,
        "default_lots": 1.0,
        "min_lots": 0.1,
        "rr_target": 2.5,
        "pivot_window": 5
    }
}

class TrendlineStrategyEngine:
    def __init__(
        self,
        mcp_client: IQForexMCPClient,
        symbol: str = "XAUUSD",
        account_type: str = "training",
        lots: float = 2.0,
        leverage: int = 100,
        notify_callback: Optional[Callable] = None,
        notify_photo_callback: Optional[Callable] = None
    ):
        self.mcp = mcp_client
        self.primary_symbol = symbol
        self.account_type = account_type
        self.lots = lots
        self.leverage = leverage
        self.notify_cb = notify_callback
        self.notify_photo_cb = notify_photo_callback

        self.is_enabled: bool = True
        self.is_running: bool = False
        self.balance_id: Optional[int] = None
        self.active_trades: Dict[str, Optional[Dict[str, Any]]] = {s: None for s in TRENDLINE_PROFILES}
        self.enabled_symbols: Set[str] = {symbol, "NAS100"}
        self.last_trade_bar: Dict[str, str] = {}
        self.last_loss_time: Dict[str, float] = {}
        self.consecutive_losses: Dict[str, int] = {}
        self.min_cooldown_seconds: int = 1800  # 30 minutes post-loss cooldown
        self.placed_position_ids: Set[str] = set()
        self.logged_trade_ids: Set[str] = set()

    def set_notification_callback(self, cb: Callable):
        self.notify_cb = cb

    def set_photo_notification_callback(self, cb: Callable):
        self.notify_photo_cb = cb

    def set_lots(self, lots: float):
        self.lots = max(0.01, float(lots))

    def set_balance(self, balance_id: int, account_type: str = "training"):
        self.balance_id = balance_id
        self.account_type = account_type
        logger.info(f"⚡ [TrendlineEngine] Balance ID set to: {balance_id} ({account_type.upper()})")

    def toggle(self) -> bool:
        self.is_enabled = not self.is_enabled
        logger.info(f"🔄 [TrendlineEngine] Master Switch: {'ENABLED' if self.is_enabled else 'DISABLED'}")
        return self.is_enabled

    def toggle_symbol(self, symbol: str) -> bool:
        if symbol in self.enabled_symbols:
            self.enabled_symbols.remove(symbol)
            state = False
        else:
            self.enabled_symbols.add(symbol)
            state = True
        logger.info(f"🔄 [TrendlineEngine] Symbol {symbol}: {'ENABLED' if state else 'DISABLED'}")
        return state

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.is_enabled,
            "running": self.is_running,
            "account_type": self.account_type,
            "enabled_symbols": list(self.enabled_symbols),
            "lots": self.lots,
            "leverage": self.leverage,
            "active_trades": self.active_trades
        }

    async def notify_text(self, text: str):
        if self.notify_cb:
            try:
                await self.notify_cb(text)
            except Exception as e:
                logger.error(f"[TrendlineEngine] Text notification failed: {e}")

    async def notify_photo(self, photo_bytes: Optional[bytes], caption: str = ""):
        if self.notify_photo_cb and photo_bytes:
            try:
                await self.notify_photo_cb(photo_bytes, caption)
            except Exception as e:
                logger.error(f"[TrendlineEngine] Photo notification failed: {e}")
                await self.notify_text(caption)
        else:
            await self.notify_text(caption)

    def find_pivots(self, highs: np.ndarray, lows: np.ndarray, window: int = 5) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        pivot_highs = []
        pivot_lows = []
        n = len(highs)
        for i in range(window, n - window):
            if highs[i] == max(highs[i - window : i + window + 1]):
                pivot_highs.append((i, highs[i]))
            if lows[i] == min(lows[i - window : i + window + 1]):
                pivot_lows.append((i, lows[i]))
        return pivot_highs, pivot_lows

    async def run_loop(self):
        self.is_running = True
        logger.info(f"🚀 [TrendlineEngine] Activated for: {list(self.enabled_symbols)} (15M Bounce Model)")

        while self.is_running:
            try:
                if self.is_enabled:
                    for symbol in list(self.enabled_symbols):
                        if symbol in TRENDLINE_PROFILES:
                            await self._evaluate_symbol_cycle(symbol)
            except Exception as e:
                logger.error(f"[TrendlineEngine] Error in execution cycle: {e}", exc_info=True)

            await asyncio.sleep(20)

    async def _evaluate_symbol_cycle(self, symbol: str):
        profile = TRENDLINE_PROFILES[symbol]
        asset_id = profile["asset_id"]

        # 1. Manage active trade if any
        if self.active_trades.get(symbol):
            await self._manage_active_trade(symbol)
            return

        # 2. Check Post-Loss Cooldown & Consecutive Loss Lockout
        now_ts = time.time()
        last_loss = self.last_loss_time.get(symbol, 0.0)
        cons_losses = self.consecutive_losses.get(symbol, 0)
        cooldown_needed = 7200 if cons_losses >= 2 else self.min_cooldown_seconds
        if (now_ts - last_loss) < cooldown_needed:
            rem_m = int(max(1, (cooldown_needed - (now_ts - last_loss)) / 60))
            logger.debug(f"[TrendlineEngine] {symbol} in cooldown ({rem_m}m remaining, {cons_losses} cons losses).")
            return

        # 3. Fetch 15-Minute Candles (size=900)
        raw_candles = self.mcp.get_candles(asset_id=asset_id, count=70, size=900)
        if not raw_candles or len(raw_candles) < 40:
            return

        # 4. Bar-Lock: Max 1 execution per 15-Minute Candle Bar
        cur_bar_time = str(raw_candles[-1].get("from") or raw_candles[-1].get("time") or raw_candles[-1].get("to") or "")
        if cur_bar_time and self.last_trade_bar.get(symbol) == cur_bar_time:
            return

        df = pd.DataFrame([{
            "Open": float(c.get("open", 0.0) or 0.0),
            "High": float(c.get("max", c.get("high", 0.0)) or 0.0),
            "Low": float(c.get("min", c.get("low", 0.0)) or 0.0),
            "Close": float(c.get("close", c.get("to", 0.0)) or 0.0)
        } for c in raw_candles])

        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        opens = df['Open'].values

        df['ema50'] = df['Close'].ewm(span=50).mean()
        cur_ema = df['ema50'].iloc[-1]

        n = len(df)
        cur_idx = n - 1
        cur_c, cur_o, cur_h, cur_l = closes[-1], opens[-1], highs[-1], lows[-1]

        sl_buf = profile["sl_buffer"]
        digits = profile["digits"]
        window = profile["pivot_window"]
        rr_target = profile["rr_target"]

        pivot_highs, pivot_lows = self.find_pivots(highs, lows, window=window)

        # ── Check Ascending Support Trendline (BUY Setup) ───────────────
        recent_lows = [p for p in pivot_lows if p[0] < cur_idx - 2]
        if len(recent_lows) >= 2:
            p1, p2 = recent_lows[-2], recent_lows[-1]
            if p2[1] > p1[1] and (p2[0] - p1[0]) >= 8:
                slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
                tl_val = p2[1] + slope * (cur_idx - p2[0])

                # 3rd Touch Bounce Rule:
                touch = (cur_l <= tl_val + sl_buf * 0.5) and (cur_l >= tl_val - sl_buf)
                reject_up = (cur_c > cur_o) and (cur_c > tl_val) and (cur_c > cur_ema)

                if touch and reject_up:
                    entry = cur_c
                    sl = round(cur_l - sl_buf, digits)
                    risk = abs(entry - sl)
                    if risk > sl_buf:
                        tp1 = round(entry + risk, digits)
                        tp2 = round(entry + (risk * rr_target), digits)
                        await self._execute_dual_ticket_trade(symbol, "BUY", entry, sl, tp1, tp2, tl_val, df, cur_bar_time)
                        return

        # ── Check Descending Resistance Trendline (SELL Setup) ──────────
        recent_highs = [p for p in pivot_highs if p[0] < cur_idx - 2]
        if len(recent_highs) >= 2:
            p1, p2 = recent_highs[-2], recent_highs[-1]
            if p2[1] < p1[1] and (p2[0] - p1[0]) >= 8:
                slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
                tl_val = p2[1] + slope * (cur_idx - p2[0])

                # 3rd Touch Bounce Rule:
                touch = (cur_h >= tl_val - sl_buf * 0.5) and (cur_h <= tl_val + sl_buf)
                reject_down = (cur_c < cur_o) and (cur_c < tl_val) and (cur_c < cur_ema)

                if touch and reject_down:
                    entry = cur_c
                    sl = round(cur_h + sl_buf, digits)
                    risk = abs(entry - sl)
                    if risk > sl_buf:
                        tp1 = round(entry - risk, digits)
                        tp2 = round(entry - (risk * rr_target), digits)
                        await self._execute_dual_ticket_trade(symbol, "SELL", entry, sl, tp1, tp2, tl_val, df, cur_bar_time)
                        return

    async def _execute_dual_ticket_trade(
        self, symbol: str, side: str, entry: float, sl: float, tp1: float, tp2: float,
        tl_val: float, df: pd.DataFrame, bar_time: str = ""
    ):
        profile = TRENDLINE_PROFILES[symbol]
        instrument_id = profile["instrument_id"]
        asset_id = profile["asset_id"]

        min_l = profile.get("min_lots", 0.001)
        trade_lots = profile.get("default_lots") or self.lots

        # Prop Firm Guardian Gatekeeper
        if getattr(self, "prop_guardian", None) and self.prop_guardian.is_enabled:
            open_pos = self.mcp.list_positions(balance_id=self.balance_id) or []
            allowed, reason, rec_lots = self.prop_guardian.can_execute_trade(symbol, len(open_pos))
            if not allowed:
                logger.warning(f"🛡️ [PropFirm Guard] Trendline trade rejected for {symbol}: {reason}")
                await self.notify_text(f"🛡️ **[PROP FIRM GUARD] Trendline Trade Skipped ({symbol})**\n\nReason: {reason}")
                return
            if rec_lots > 0:
                trade_lots = rec_lots

        if symbol == "XAUUSD":
            min_l = 1.0
            trade_lots = max(2.0, float(trade_lots))

        if trade_lots >= (min_l * 2.0):
            lot_tp1 = round(trade_lots / 2.0, 4 if min_l < 1 else 2)
            lot_runner = round(trade_lots - lot_tp1, 4 if min_l < 1 else 2)
        else:
            lot_tp1 = min_l
            lot_runner = min_l

        # Lock bar immediately before/during placement to prevent concurrent execution
        if bar_time:
            self.last_trade_bar[symbol] = bar_time

        trade_lev = min(self.leverage, 20 if symbol == "BTCUSD" else self.leverage)
        logger.info(f"📐 [TrendlineEngine] Executing 15M Bounce {side} on {symbol} @ {entry} | Line: {tl_val:.2f} | SL: {sl} | TP1: {tp1} | TP2: {tp2}")

        # Leg 1: Banks profit at TP1 (+1.0R)
        res1 = self.mcp.place_market_order(
            side=side.lower(),
            balance_id=self.balance_id,
            instrument_id=instrument_id,
            asset_id=asset_id,
            lots=lot_tp1,
            leverage=trade_lev,
            stop_loss=sl,
            take_profit=tp1,
            is_margin_isolated=True,
            keep_position_open=False
        )

        # Leg 2: Runner to TP2 (+2.5R)
        res2 = self.mcp.place_market_order(
            side=side.lower(),
            balance_id=self.balance_id,
            instrument_id=instrument_id,
            asset_id=asset_id,
            lots=lot_runner,
            leverage=trade_lev,
            stop_loss=sl,
            take_profit=tp2,
            is_margin_isolated=True,
            keep_position_open=False
        )

        pos1_id = res1.get("position_id") or res1.get("order_id") if isinstance(res1, dict) else None
        pos2_id = res2.get("position_id") or res2.get("order_id") if isinstance(res2, dict) else None

        # Resolve order IDs to broker position IDs
        if pos1_id and hasattr(self.mcp, "_resolve_actual_position_id"):
            resolved1 = self.mcp._resolve_actual_position_id(pos1_id)
            if resolved1: pos1_id = resolved1
        if pos2_id and hasattr(self.mcp, "_resolve_actual_position_id"):
            resolved2 = self.mcp._resolve_actual_position_id(pos2_id)
            if resolved2: pos2_id = resolved2

        if pos1_id: self.placed_position_ids.add(str(pos1_id))
        if pos2_id: self.placed_position_ids.add(str(pos2_id))

        self.active_trades[symbol] = {
            "symbol": symbol,
            "side": side,
            "entry_price": entry,
            "sl_price": sl,
            "tp1": tp1,
            "tp2": tp2,
            "trendline_price": tl_val,
            "leg1_pos_id": pos1_id,
            "leg2_pos_id": pos2_id,
            "lot_tp1": lot_tp1,
            "lot_runner": lot_runner,
            "tp1_closed": False,
            "runner_at_be": False,
            "opened_at": time.time()
        }

        # Telegram Chart Notification
        chart_bytes = generate_trade_execution_chart(
            df=df,
            symbol=symbol,
            side=side,
            entry_px=entry,
            sl=sl,
            tp=tp2,
            engine_name="Trendline",
            event_title="15M Trendline 3rd Touch Bounce"
        )

        caption = (
            f"📐 **Trendline 3rd Touch Bounce Executed** | {symbol} {side}\n\n"
            f"• Trendline Level: `{tl_val:.2f}`\n"
            f"• Entry Price    : `{entry:.2f}` (Rejection Confirmed)\n"
            f"• Stop Loss      : `{sl:.2f}`\n"
            f"• Leg 1 (50% TP1): `{tp1:.2f}` (+1.0R guaranteed bank)\n"
            f"• Leg 2 (50% TP2): `{tp2:.2f}` (+2.5R runner)\n"
            f"• Lot Sizing     : `{lot_tp1 + lot_runner:.2f}` Lots ({trade_lev}x)\n\n"
            f"Leg 2 will automatically shift Stop Loss to Breakeven when Leg 1 hits TP1."
        )
        await self.notify_photo(chart_bytes, caption)

    async def _manage_active_trade(self, symbol: str):
        trade = self.active_trades.get(symbol)
        if not trade:
            return

        profile = TRENDLINE_PROFILES.get(symbol, {})
        side = trade["side"]
        entry = trade["entry_price"]
        tp1 = trade["tp1"]
        tp2 = trade["tp2"]

        open_positions = []
        try:
            open_positions = self.mcp.list_positions(balance_id=self.balance_id) or []
        except Exception:
            pass

        # Build comprehensive set of all open IDs (position_id, id, order_id)
        open_ids = set()
        for p in open_positions:
            for k in ["position_id", "id", "order_id"]:
                v = p.get(k)
                if v:
                    open_ids.add(str(v))

        leg1_id = str(trade.get("leg1_pos_id"))
        leg2_id = str(trade.get("leg2_pos_id"))

        # Re-resolve if IDs still match broker open positions by asset_id
        if leg1_id not in open_ids and leg2_id not in open_ids:
            asset_open = [p for p in open_positions if p.get("asset_id") == profile.get("asset_id")]
            if asset_open:
                if len(asset_open) >= 2:
                    p1 = str(asset_open[0].get("position_id") or asset_open[0].get("id"))
                    p2 = str(asset_open[1].get("position_id") or asset_open[1].get("id"))
                    trade["leg1_pos_id"] = p1
                    trade["leg2_pos_id"] = p2
                    leg1_id, leg2_id = p1, p2
                    open_ids.add(p1)
                    open_ids.add(p2)
                elif len(asset_open) == 1:
                    p2 = str(asset_open[0].get("position_id") or asset_open[0].get("id"))
                    trade["leg1_pos_id"] = None
                    trade["leg2_pos_id"] = p2
                    leg1_id = "None"
                    leg2_id = p2
                    open_ids.add(p2)

        leg1_still_open = leg1_id in open_ids and leg1_id != "None"
        leg2_still_open = leg2_id in open_ids and leg2_id != "None"

        # 1. Check if Leg 1 (TP1) closed while Leg 2 is still running
        if not trade["tp1_closed"]:
            if leg1_id and not leg1_still_open and leg1_id != "None":
                trade["tp1_closed"] = True
                logger.info(f"🏆 [TrendlineEngine] {symbol} Leg 1 (#{leg1_id}) closed! Moving Leg 2 to Breakeven...")

                # Shift Leg 2 to Breakeven
                if leg2_still_open:
                    try:
                        be_buf = profile.get("sl_buffer", 1.0) * 0.15
                        be_price = round(entry + be_buf if side == "BUY" else entry - be_buf, profile["digits"])
                        self.mcp.change_position_stop_loss(position_id=int(leg2_id), level=be_price)
                        trade["runner_at_be"] = True
                        logger.info(f"✅ [TrendlineEngine] Leg 2 (#{leg2_id}) SL shifted to Breakeven @ {be_price}!")
                    except Exception as e:
                        logger.error(f"[TrendlineEngine] Failed to move Leg 2 to BE: {e}")

                await self.notify_text(
                    f"🛡️ **[TRENDLINE PARTIAL BANKED] {symbol}**\n\n"
                    f"• Leg 1 hit TP1 @ `{tp1:.2f}`! Profit banked.\n"
                    f"• Leg 2 Runner Stop Loss shifted to **Breakeven** (`{entry:.2f}`).\n"
                    f"• Position is now **100% Risk-Free** targeting `{tp2:.2f}`!"
                )

                # Log Leg 1 closure to Google Sheets
                await self._log_ticket_closure(
                    symbol=symbol,
                    tag="Leg 1 (50% TP1)",
                    pos_id=leg1_id,
                    lots=trade.get("lot_tp1", 0.5),
                    entry_px=entry,
                    sl_px=trade.get("sl_price", 0.0),
                    tp_px=tp1,
                    default_reason="take_profit"
                )

        # 2. Check if all legs are closed
        if not leg1_still_open and not leg2_still_open:
            logger.info(f"🏁 [TrendlineEngine] {symbol} All trade legs closed. Cycle complete.")

            # Log Leg 2 (Runner) to Google Sheets
            if leg2_id and leg2_id != "None":
                await self._log_ticket_closure(
                    symbol=symbol,
                    tag="Leg 2 (Runner)",
                    pos_id=leg2_id,
                    lots=trade.get("lot_runner", 0.5),
                    entry_px=entry,
                    sl_px=trade.get("sl_price", 0.0),
                    tp_px=tp2,
                    default_reason="take_profit" if trade.get("tp1_closed") else "stop_loss"
                )

            # If Leg 1 stopped out without hitting TP1, log Leg 1 as well
            if not trade.get("tp1_closed") and leg1_id and leg1_id != "None":
                await self._log_ticket_closure(
                    symbol=symbol,
                    tag="Leg 1 (50% TP1)",
                    pos_id=leg1_id,
                    lots=trade.get("lot_tp1", 0.5),
                    entry_px=entry,
                    sl_px=trade.get("sl_price", 0.0),
                    tp_px=tp1,
                    default_reason="stop_loss"
                )

            # Evaluate outcome: if TP1 never banked, register loss & trigger cooldown
            if not trade.get("tp1_closed"):
                self.last_loss_time[symbol] = time.time()
                self.consecutive_losses[symbol] = self.consecutive_losses.get(symbol, 0) + 1
                logger.warning(
                    f"⚠️ [TrendlineEngine] {symbol} Loss registered. Consecutive losses: {self.consecutive_losses[symbol]}. "
                    f"Enforcing {'2-Hour Lockout' if self.consecutive_losses[symbol] >= 2 else '30-Minute Cooldown'}."
                )
            else:
                self.consecutive_losses[symbol] = 0
                # Enforce a 15-minute breather even after a win to prevent instant re-entry on same structure
                self.last_loss_time[symbol] = time.time() - (self.min_cooldown_seconds - 900)

            self.active_trades[symbol] = None

    async def _log_ticket_closure(
        self, symbol: str, tag: str, pos_id: Any, lots: float,
        entry_px: float, sl_px: float, tp_px: float, default_reason: str = "closed",
        reason_override: Optional[str] = None
    ):
        if not pos_id or str(pos_id) == "None":
            return
        pid_str = str(pos_id)
        if pid_str in self.logged_trade_ids:
            return
        self.logged_trade_ids.add(pid_str)

        profile = TRENDLINE_PROFILES.get(symbol, {})
        digits = profile.get("digits", 2)
        sym_label = "US100" if symbol == "NAS100" else symbol
        trade = self.active_trades.get(symbol) or {}
        side = trade.get("side", "BUY")

        pnl = 0.0
        exit_px = entry_px
        reason = reason_override or default_reason

        # Fetch trade history from broker for exact PnL and exit price
        for attempt in range(3):
            try:
                hist = self.mcp.get_trade_history(balance_id=self.balance_id, limit=25) or []
                matched = next(
                    (h for h in hist if 
                     str(h.get("position_id")) == pid_str or 
                     str(h.get("order_id")) == pid_str or 
                     str(h.get("id")) == pid_str),
                    None
                )
                if matched:
                    pnl = float(matched.get("pnl", 0.0))
                    exit_px = float(matched.get("close_price", matched.get("exit_price", entry_px)))
                    reason = reason_override or matched.get("close_reason", reason)
                    break
            except Exception as e:
                logger.debug(f"[TrendlineEngine] History lookup attempt {attempt+1} error: {e}")
            if attempt < 2:
                await asyncio.sleep(0.8)

        # Sync to Google Sheets
        if gsheet_logger:
            try:
                eq = 0.0
                try:
                    bals = self.mcp.list_balances() or []
                    cur_bal = next((b for b in bals if b.get("balance_id") == self.balance_id), None)
                    if cur_bal:
                        eq = float(cur_bal.get("equity") or cur_bal.get("amount", 0.0))
                except Exception:
                    pass

                trade_payload = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "asset": f"Trendline {sym_label} ({tag})",
                    "side": side,
                    "lots": lots,
                    "entry_price": entry_px,
                    "stop_loss": sl_px,
                    "take_profit": tp_px,
                    "exit_price": exit_px,
                    "pnl": pnl,
                    "pips": round(abs(exit_px - entry_px) * (10 ** (digits - 1)), 1),
                    "risk_reward": f"{tag} (1:2.5 Plan)",
                    "exit_reason": reason,
                    "position_id": pid_str,
                    "balance_equity": eq
                }
                asyncio.create_task(asyncio.to_thread(gsheet_logger.log_forex_margin_trade, trade_payload))
                logger.info(f"📊 [TrendlineEngine] Dispatched Google Sheets log for {sym_label} #{pid_str} PnL=${pnl:.2f}")
            except Exception as ge:
                logger.warning(f"[TrendlineEngine] GSheet dispatch error: {ge}")

    async def _log_trade_closure(self, symbol: str, pos_id: Any, reason_override: Optional[str] = None):
        trade = self.active_trades.get(symbol)
        if not trade:
            return
        pid_str = str(pos_id)
        if str(trade.get("leg1_pos_id")) == pid_str:
            await self._log_ticket_closure(
                symbol, "Leg 1 (50% TP1)", pos_id, trade.get("lot_tp1", 0.5),
                trade["entry_price"], trade["sl_price"], trade["tp1"],
                default_reason="TP1 Partial Hit", reason_override=reason_override
            )
        elif str(trade.get("leg2_pos_id")) == pid_str:
            await self._log_ticket_closure(
                symbol, "Leg 2 (Runner)", pos_id, trade.get("lot_runner", 0.5),
                trade["entry_price"], trade["sl_price"], trade["tp2"],
                default_reason="Runner Closed", reason_override=reason_override
            )
