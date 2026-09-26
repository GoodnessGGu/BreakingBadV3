"""
strategies/news_straddle_engine.py - Experimental High-Impact News Straddle Spike Engine

Dedicated exclusively to trading high-volatility news events (NFP, CPI, FOMC, Rate Decisions):
  1. Pre-News Arming: Captures pre-news tight consolidation range at T-2 mins (13:28 WAT).
  2. High-Speed Spike Breakout Triggers: Watches live sub-second price ticks at T-0 (13:30 WAT).
  3. Instant One-Cancels-Other (OCO) Execution: Enters the breakout direction instantly.
  4. Hyper-Fast Risk Management: Micro-second Breakeven lock + dynamic profit trailing.
  5. Auto-Disarm Safety: Cleans up and disarms if no breakout triggers within 3 minutes.
"""

import time
import logging
import asyncio
import inspect
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable, Set

from clients.forex_mcp_client import IQForexMCPClient
from bot.news_engine import EconomicNewsEngine
from utils.gsheet_logger import gsheet_logger

logger = logging.getLogger("NewsStraddle")

GOLD_INSTRUMENT = "front.marginal-cfd.instrument.xauusd"
GOLD_ASSET_ID = 74

class NewsStraddleEngine:
    def __init__(self, forex_mcp: IQForexMCPClient, news_engine: EconomicNewsEngine,
                 lots: float = 1.0, leverage: int = 100):
        self.mcp = forex_mcp
        self.news_engine = news_engine
        self.lots = float(lots)
        self.leverage = int(leverage)
        self.is_enabled: bool = False  # Experimental — default disabled until toggled
        self.is_running: bool = False

        self.symbol: str = "XAUUSD"
        self.asset_id: int = GOLD_ASSET_ID
        self.instrument_id: str = GOLD_INSTRUMENT

        # Straddle Parameters (Optimized for Gold CFD)
        self.buffer_distance: float = 1.50       # $1.50 above high / below low
        self.sl_distance: float = 3.00           # $3.00 initial Stop Loss
        self.tp_distance: float = 6.00           # $6.00 initial Take Profit (1:2 RR)
        self.be_trigger_distance: float = 2.00   # Move to BE once in +$2.00 profit
        self.trail_offset: float = 1.50          # Trail $1.50 behind peak price once in BE

        # State tracking
        self.active_straddle: Optional[Dict[str, Any]] = None
        self.active_trade: Optional[Dict[str, Any]] = None
        self.handled_events: Set[str] = set()

        self.balance_id: Optional[int] = None
        self.notify_cb: Optional[Callable] = None

    def set_notification_callback(self, cb: Callable):
        self.notify_cb = cb

    async def notify(self, message: str):
        if self.notify_cb:
            try:
                res = self.notify_cb(message)
                if inspect.isawaitable(res):
                    await res
            except Exception as e:
                logger.warning(f"[NewsStraddle] Notification error: {e}")

    def set_balance(self, balance_id: Optional[int], account_type: str = "training"):
        self.balance_id = int(balance_id) if balance_id else None

    def set_lots(self, lots: float):
        self.lots = max(0.01, round(float(lots), 2))

    def set_leverage(self, leverage: int):
        self.leverage = int(leverage)

    def set_enabled(self, enabled: bool):
        self.is_enabled = bool(enabled)
        logger.info(f"⚡ [NewsStraddle] Straddle Engine set to: {'ENABLED' if self.is_enabled else 'DISABLED'}")

    def toggle_enabled(self) -> bool:
        self.is_enabled = not self.is_enabled
        logger.info(f"⚡ [NewsStraddle] Straddle Engine state toggled to: {'ENABLED' if self.is_enabled else 'DISABLED'}")
        return self.is_enabled

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": "News Straddle Spike Engine",
            "enabled": self.is_enabled,
            "symbol": self.symbol,
            "lots": self.lots,
            "leverage": self.leverage,
            "is_armed": self.active_straddle is not None,
            "has_active_trade": self.active_trade is not None,
            "buffer": self.buffer_distance,
            "tp": self.tp_distance,
            "sl": self.sl_distance
        }

    def _get_market_price(self) -> Dict[str, float]:
        try:
            return self.mcp.get_market_price(self.instrument_id)
        except Exception as e:
            logger.debug(f"[NewsStraddle] Error fetching market price: {e}")
            return {"buy": 0.0, "sell": 0.0, "mid": 0.0}

    async def check_upcoming_events_to_arm(self):
        """Checks for High-Impact USD news scheduled in ~2 minutes (120 seconds)."""
        if not self.is_enabled or self.active_straddle or self.active_trade:
            return

        now_utc = datetime.now(timezone.utc)
        for e in self.news_engine.events:
            if e.get("impact") != "High" or e.get("country") != "USD":
                continue

            event_id = e["id"]
            if event_id in self.handled_events:
                continue

            dt = e["dt_utc"]
            diff_secs = (dt - now_utc).total_seconds()

            # Arm between 150s and 45s before the release (T-2 minutes)
            if 45.0 <= diff_secs <= 150.0:
                self.handled_events.add(event_id)
                await self.arm_straddle(e, diff_secs)
                break

    async def arm_straddle(self, event: Dict[str, Any], seconds_to_news: float):
        """Captures pre-news range and arms breakout triggers."""
        logger.info(f"⚡ [NewsStraddle] Arming NFP/News Straddle for {event['title']} (In {int(seconds_to_news)}s)...")

        # Fetch recent M1 candles to find consolidation bounds
        candles = self.mcp.get_candles(self.instrument_id, count=15)
        prices = self._get_market_price()
        cur_mid = prices.get("mid", 0.0)

        if candles and len(candles) >= 5:
            highs = [c["max"] for c in candles]
            lows = [c["min"] for c in candles]
            range_high = max(highs)
            range_low = min(lows)
        elif cur_mid > 0:
            range_high = cur_mid + 0.80
            range_low = cur_mid - 0.80
        else:
            logger.error("[NewsStraddle] Could not retrieve prices/candles to arm straddle.")
            return

        buy_trigger = round(range_high + self.buffer_distance, 2)
        sell_trigger = round(range_low - self.buffer_distance, 2)

        self.active_straddle = {
            "event_title": event["title"],
            "event_time_utc": event["dt_utc"],
            "armed_at": time.time(),
            "pre_high": range_high,
            "pre_low": range_low,
            "buy_trigger": buy_trigger,
            "sell_trigger": sell_trigger,
            "status": "ARMED"
        }

        mins_to_news = int(seconds_to_news // 60)
        secs_to_news = int(seconds_to_news % 60)
        time_tag = f"{mins_to_news}m {secs_to_news}s" if mins_to_news > 0 else f"{secs_to_news}s"

        await self.notify(
            f"⚡ **[NEWS STRADDLE ARMED] {event['title']} in {time_tag}**\n\n"
            f"• Asset      : `{self.symbol}`\n"
            f"• Pre-Range  : `{range_low:.2f}` – `{range_high:.2f}`\n"
            f"• BUY Trigger: Above `{buy_trigger:.2f}` (+$1.50)\n"
            f"• SELL Trigger: Below `{sell_trigger:.2f}` (-$1.50)\n"
            f"• Lots & Lev : `{self.lots}` Lots | `{self.leverage}x`\n"
            f"🎯 _Listening for high-velocity breakout spike..._"
        )

    async def check_straddle_triggers(self):
        """High-frequency polling loop during armed window to execute on breakout spike."""
        if not self.active_straddle or self.active_trade:
            return

        straddle = self.active_straddle
        now_ts = time.time()
        armed_duration = now_ts - straddle["armed_at"]

        # Safety Timeout: Disarm if 4 minutes passed without breakout (whipsaw / flat event)
        if armed_duration > 240.0:
            logger.info("⏰ [NewsStraddle] Straddle disarmed after 4-minute timeout without clean breakout.")
            await self.notify(f"ℹ️ **[NEWS STRADDLE DISARMED]** No breakout triggered for {straddle['event_title']}.")
            self.active_straddle = None
            return

        prices = self._get_market_price()
        mid = prices.get("mid", 0.0)
        if mid <= 0:
            return

        buy_trig = straddle["buy_trigger"]
        sell_trig = straddle["sell_trigger"]

        # 1. Bullish Spike Breakout Triggered
        if mid >= buy_trig:
            logger.info(f"🚀 [NewsStraddle] BULLISH BREAKOUT SPIKE DETECTED at {mid:.2f} >= {buy_trig:.2f}!")
            self.active_straddle = None
            await self._execute_straddle_order(side="BUY", trigger_px=mid, pre_range=straddle)
            return

        # 2. Bearish Spike Breakout Triggered
        if mid <= sell_trig:
            logger.info(f"🚀 [NewsStraddle] BEARISH BREAKOUT SPIKE DETECTED at {mid:.2f} <= {sell_trig:.2f}!")
            self.active_straddle = None
            await self._execute_straddle_order(side="SELL", trigger_px=mid, pre_range=straddle)
            return

    async def _execute_straddle_order(self, side: str, trigger_px: float, pre_range: Dict[str, Any]):
        """Executes instant market order with tight SL & TP."""
        prices = self._get_market_price()
        exec_px = prices["buy"] if side == "BUY" else prices["sell"]
        if exec_px <= 0:
            exec_px = trigger_px

        sl = round(exec_px - self.sl_distance if side == "BUY" else exec_px + self.sl_distance, 2)
        tp = round(exec_px + self.tp_distance if side == "BUY" else exec_px - self.tp_distance, 2)

        await self.notify(
            f"🚀 **[NEWS STRADDLE TRIGGERED] {self.symbol} {side}**\n\n"
            f"• Event  : `{pre_range.get('event_title', 'High-Impact News')}`\n"
            f"• Entry  : `{exec_px:.2f}`\n"
            f"• SL / TP: `{sl:.2f}` / `{tp:.2f}`\n"
            f"• Lots   : `{self.lots}` (`{self.leverage}x`)"
        )

        res = self.mcp.place_market_order(
            side=side.lower(),
            balance_id=self.balance_id,
            instrument_id=self.instrument_id,
            asset_id=self.asset_id,
            lots=self.lots,
            leverage=self.leverage,
            stop_loss=sl,
            take_profit=tp,
            is_margin_isolated=True,
            keep_position_open=False
        )

        if "order_id" in res:
            order_id = res["order_id"]
            logger.info(f"✅ [NewsStraddle] Order placed! ID: #{order_id}")
            self.active_trade = {
                "order_id": order_id,
                "position_id": None,
                "side": side,
                "entry_price": exec_px,
                "initial_sl": sl,
                "current_sl": sl,
                "tp": tp,
                "moved_to_be": False,
                "highest_gain": 0.0,
                "opened_at": time.time(),
                "event_title": pre_range.get("event_title", "News")
            }
            await self.notify(f"✅ **[NEWS STRADDLE FILLED] #{order_id} {side} @ {exec_px:.2f}**")
            asyncio.create_task(self._monitor_straddle_trade(order_id))
        else:
            logger.error(f"❌ [NewsStraddle] Order failed: {res}")
            await self.notify(f"❌ **[NEWS STRADDLE FAILED]**: {res.get('error', res)}")

    async def _monitor_straddle_trade(self, order_id: int):
        """High-frequency trade monitoring with rapid Breakeven lock & dynamic trailing."""
        await asyncio.sleep(2)
        trade = self.active_trade
        if not trade:
            return

        # Resolve position_id
        for _ in range(8):
            if trade.get("position_id"):
                break
            try:
                positions = self.mcp.list_positions(balance_id=self.balance_id)
                for p in positions:
                    if p.get("asset_id") == self.asset_id:
                        trade["position_id"] = p.get("position_id") or p.get("id")
                        break
            except Exception as e:
                logger.warning(f"[NewsStraddle] Position lookup error: {e}")
            if not trade.get("position_id"):
                await asyncio.sleep(1)

        pos_id = trade.get("position_id")
        side = trade["side"]
        entry = trade["entry_price"]

        logger.info(f"🛡️ [NewsStraddle] Monitoring trade #{pos_id or order_id} with rapid BE & spike trailing.")

        while self.active_trade and self.active_trade.get("order_id") == order_id:
            await asyncio.sleep(1)  # Sub-second / 1s fast tick loop

            # Check if still open
            try:
                open_positions = self.mcp.list_positions(balance_id=self.balance_id)
                is_still_open = any((p.get("position_id") or p.get("id")) == pos_id for p in open_positions) if pos_id else True
            except Exception:
                continue

            if not is_still_open:
                logger.info(f"📊 [NewsStraddle] Position #{pos_id} closed! Resolving settlement...")
                await self._log_trade_closure(order_id)
                self.active_trade = None
                break

            # Fast Breakeven & Spike Trailing
            if pos_id:
                prices = self._get_market_price()
                mid = prices.get("mid", 0.0)
                if mid > 0:
                    gain = (mid - entry) if side == "BUY" else (entry - mid)
                    if gain > trade["highest_gain"]:
                        trade["highest_gain"] = gain

                    # 1. Fast Breakeven (+0.30 buffer) once price gains >= $2.00
                    if gain >= self.be_trigger_distance and not trade["moved_to_be"]:
                        be_buf = 0.30
                        be_level = round(entry + be_buf if side == "BUY" else entry - be_buf, 2)
                        res = self.mcp.change_position_stop_loss(position_id=pos_id, level=be_level)
                        if not res.get("error"):
                            trade["current_sl"] = be_level
                            trade["moved_to_be"] = True
                            logger.info(f"🛡️ [NewsStraddle +$2.00] Breakeven activated on #{pos_id}! SL: {be_level}")
                            await self.notify(
                                f"🛡️ **[NEWS STRADDLE BREAKEVEN] #{pos_id} ({side})**\n\n"
                                f"• Trade is now Risk-Free! SL shifted to: `{be_level:.2f}`"
                            )

                    # 2. Dynamic Trailing ($1.50 behind peak price once past $3.50 gain)
                    if gain >= 3.50 and trade["moved_to_be"]:
                        trail_sl = round(mid - self.trail_offset if side == "BUY" else mid + self.trail_offset, 2)
                        cur_sl = trade.get("current_sl", 0.0)
                        should_update = (trail_sl > cur_sl + 0.60) if side == "BUY" else (trail_sl < cur_sl - 0.60)
                        if should_update:
                            res = self.mcp.change_position_stop_loss(position_id=pos_id, level=trail_sl)
                            if not res.get("error"):
                                trade["current_sl"] = trail_sl
                                logger.info(f"🚀 [NewsStraddle Trailing] SL ratcheted to {trail_sl} on #{pos_id}")

    async def _log_trade_closure(self, order_id: int):
        trade = self.active_trade or {}
        pos_id = trade.get("position_id")
        side = trade.get("side", "BUY")
        entry_px = trade.get("entry_price", 0.0)
        tp = trade.get("tp", 0.0)
        sl = trade.get("current_sl", 0.0)
        event_name = trade.get("event_title", "NFP / News Event")

        pnl = 0.0
        exit_px = 0.0
        reason = "closed"

        try:
            history = self.mcp.get_trade_history(balance_id=self.balance_id, limit=5)
            matched = next((h for h in history if (pos_id and h.get("position_id") == pos_id) or h.get("asset_id") == self.asset_id), None)
            if matched:
                pnl = float(matched.get("pnl", 0.0))
                exit_px = float(matched.get("close_price", 0.0))
                reason = matched.get("close_reason", "closed")
        except Exception as e:
            logger.warning(f"[NewsStraddle] History fetch error: {e}")

        bal = self.mcp.get_training_balance()
        eq = bal.get("equity", 0.0) if bal else 0.0

        try:
            gsheet_logger.log_forex_margin_trade({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "asset": f"News Straddle ({self.symbol})",
                "side": side,
                "lots": self.lots,
                "entry_price": entry_px,
                "stop_loss": sl,
                "take_profit": tp,
                "exit_price": exit_px,
                "pnl": pnl,
                "pips": round(abs(exit_px - entry_px) * 10, 1) if exit_px else 0.0,
                "risk_reward": f"News Spike ({event_name})",
                "exit_reason": reason,
                "position_id": pos_id or order_id,
                "balance_equity": eq
            })
        except Exception as e:
            logger.warning(f"[NewsStraddle] GSheet log error: {e}")

        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        if pnl > 0:
            header_line = f"🏆 **[NEWS STRADDLE WON] {self.symbol} {pnl_str}**"
        elif pnl == 0:
            header_line = f"🛡️ **[NEWS STRADDLE BREAKEVEN] {self.symbol} $0.00**"
        else:
            header_line = f"❌ **[NEWS STRADDLE CLOSED] {self.symbol} {pnl_str}**"

        await self.notify(
            f"{header_line}\n\n"
            f"• Event   : `{event_name}`\n"
            f"• Position: `#{pos_id}` ({side})\n"
            f"• Prices  : `{entry_px:.2f}` ➔ `{exit_px:.2f}`\n"
            f"• Net PnL : `{pnl_str}`\n"
            f"• Reason  : `{reason}`"
        )

    async def run_loop(self):
        """Main background loop checking for upcoming high-impact news events to arm/trade."""
        self.is_running = True
        logger.info("⚡ [NewsStraddle] Dedicated High-Impact News Straddle Engine started.")
        while self.is_running:
            try:
                if self.is_enabled:
                    # 1. Check if an armed straddle is active (poll triggers every 500ms)
                    if self.active_straddle:
                        await self.check_straddle_triggers()
                        await asyncio.sleep(0.5)
                        continue

                    # 2. Check if there's an upcoming event in ~2m to arm
                    await self.check_upcoming_events_to_arm()

            except Exception as e:
                logger.error(f"[NewsStraddle] Error in loop: {e}")

            await asyncio.sleep(1.0)

    def stop(self):
        self.is_running = False
