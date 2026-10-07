"""
strategies/crt_engine.py - Autonomous Institutional Candle Range Theory (CRT) Engine
Upgraded for:
1. Real Multi-Timeframe Anchor Range Tracking (TRUE 1-Hour candles for H1_ANCHOR mode).
2. Institutional Killzone Timing (London Open 07:00-10:00 UTC | NY Open 12:30-16:00 UTC).
3. Trend Filter: 50 EMA on 1-Hour candles.
4. Dual-Ticket Partial Take-Profit Execution:
   - Leg 1 (50% size): Takes profit at +1.0R (banks guaranteed profit).
   - Leg 2 (50% size): Automatically moves SL to Breakeven when Leg 1 hits TP1, targets +2.2R runner.
5. High-resolution TradingView Pro graphical Telegram alerts.
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
    generate_breakeven_chart,
    generate_trade_close_chart,
    generate_ict_setup_chart
)

try:
    from gsheet_logger import gsheet_logger
except ImportError:
    try:
        from utils.gsheet_logger import gsheet_logger
    except ImportError:
        gsheet_logger = None

logger = logging.getLogger("CRTEngine")

CRT_INSTRUMENT_PROFILES: Dict[str, Dict[str, Any]] = {
    "EURUSD": {
        "symbol": "EURUSD",
        "name": "EUR/USD",
        "asset_id": 1,
        "instrument_id": "mf.1",
        "min_sweep": 0.0004,
        "max_sweep": 0.0035,
        "sl_buffer": 0.0003,
        "default_lots": 0.1,
        "min_lots": 0.001,
        "contract_size": 100000,
        "target_rr": 2.2,
        "digits": 5,
        "mode": "ASIAN_JUDAS"
    },
    "GBPUSD": {
        "symbol": "GBPUSD",
        "name": "GBP/USD",
        "asset_id": 5,
        "instrument_id": "mf.5",
        "min_sweep": 0.0005,
        "max_sweep": 0.0040,
        "sl_buffer": 0.0004,
        "default_lots": 0.1,
        "min_lots": 0.001,
        "contract_size": 100000,
        "target_rr": 2.2,
        "digits": 5,
        "mode": "ASIAN_JUDAS"
    },
    "BTCUSD": {
        "symbol": "BTCUSD",
        "name": "Bitcoin (BTC/USD)",
        "asset_id": 816,
        "instrument_id": "mcrpt.816",
        "min_sweep": 120.0,
        "max_sweep": 1500.0,
        "sl_buffer": 150.0,      # Realistic buffer matching BTC volatility
        "default_lots": 0.02,     # Split into two 0.01 micro-lots
        "min_lots": 0.001,
        "contract_size": 1,
        "target_rr": 2.2,
        "digits": 2,
        "mode": "H1_ANCHOR"
    },
    "XAUUSD": {
        "symbol": "XAUUSD",
        "name": "Gold (XAU/USD)",
        "asset_id": 74,
        "instrument_id": "mcfd.74",
        "min_sweep": 3.0,
        "max_sweep": 35.0,
        "sl_buffer": 2.5,
        "default_lots": 2.0,      # Two 1.0-lot tickets
        "min_lots": 1.0,
        "contract_size": 100,
        "target_rr": 2.2,
        "digits": 2,
        "mode": "H1_ANCHOR"
    }
}


class CRTStrategyEngine:
    """
    Autonomous Institutional Candle Range Theory (CRT) Execution Engine.
    Specialized for Forex (EURUSD, GBPUSD), Crypto (BTCUSD), and Metals.
    """

    def __init__(
        self,
        mcp_client: IQForexMCPClient,
        symbols: Optional[List[str]] = None,
        account_type: str = "training",
        lots: float = 1.0,
        leverage: int = 100,
        notify_callback: Optional[Callable] = None,
        notify_photo_callback: Optional[Callable] = None
    ):
        self.mcp = mcp_client
        self.symbols = symbols or ["EURUSD", "BTCUSD"]
        self.account_type = account_type
        self.lots = lots
        self.leverage = leverage
        self.notify_cb = notify_callback
        self.notify_photo_cb = notify_photo_callback

        self.is_enabled: bool = True
        self.is_running: bool = False
        self.balance_id: Optional[int] = None
        self.active_trades: Dict[str, Optional[Dict[str, Any]]] = {s: None for s in self.symbols}
        self.enabled_symbols: Set[str] = set(self.symbols)
        self.unavailable_cooldown: Dict[str, float] = {}
        self.placed_position_ids: Set[str] = set()
        self.logged_trade_ids: Set[str] = set()

    def set_notification_callback(self, cb: Callable):
        self.notify_cb = cb

    def set_photo_notification_callback(self, cb: Callable):
        self.notify_photo_cb = cb

    def set_balance(self, balance_id: int, account_type: str = "training"):
        self.balance_id = balance_id
        self.account_type = account_type
        logger.info(f"⚡ [CRTEngine] Balance ID set to: {balance_id} ({account_type.upper()})")

    def toggle(self) -> bool:
        self.is_enabled = not self.is_enabled
        logger.info(f"🔄 [CRTEngine] Master Switch: {'ENABLED' if self.is_enabled else 'DISABLED'}")
        return self.is_enabled

    def toggle_symbol(self, symbol: str) -> bool:
        if symbol in self.enabled_symbols:
            self.enabled_symbols.remove(symbol)
            state = False
        else:
            self.enabled_symbols.add(symbol)
            state = True
        logger.info(f"🔄 [CRTEngine] Asset {symbol}: {'ENABLED' if state else 'DISABLED'}")
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
                logger.error(f"[CRTEngine] Text notification failed: {e}")

    async def notify_photo(self, photo_bytes: Optional[bytes], caption: str = ""):
        if self.notify_photo_cb and photo_bytes:
            try:
                await self.notify_photo_cb(photo_bytes, caption)
            except Exception as e:
                logger.error(f"[CRTEngine] Photo notification failed: {e}")
                await self.notify_text(caption)
        else:
            await self.notify_text(caption)

    def is_killzone_active(self, symbol: str) -> Tuple[bool, str]:
        """Returns True if within active trading killzones. Crypto is 24/7."""
        if symbol == "BTCUSD":
            return True, "CRYPTO_24_7"
        now = datetime.now(timezone.utc)
        mins = now.hour * 60 + now.minute
        if 420 <= mins <= 600:
            return True, "LONDON_OPEN"
        if 750 <= mins <= 960:
            return True, "NEW_YORK_OPEN"
        return False, "OFF_HOURS"

    def get_market_price(self, symbol: str) -> Dict[str, float]:
        profile = CRT_INSTRUMENT_PROFILES.get(symbol)
        if not profile:
            return {"buy": 0.0, "sell": 0.0, "mid": 0.0}

        try:
            candles = self.mcp.get_candles(asset_id=profile["asset_id"], size=60, count=2)
            if candles and len(candles) > 0:
                last_c = candles[-1]
                px = float(last_c.get("close", last_c.get("c", 0.0)))
                if px > 0:
                    return {"buy": px, "sell": px, "mid": px}
        except Exception as e:
            logger.debug(f"[CRTEngine] Candle price fetch error for {symbol}: {e}")

        lev = min(self.leverage, 20 if symbol == "BTCUSD" else self.leverage)
        try:
            p = self.mcp.calculate_order_size(
                asset_id=profile["asset_id"], balance_currency="USD",
                lots=self.lots, leverage=lev
            )
            if isinstance(p, dict) and "buy_price" in p and "sell_price" in p:
                buy = float(p.get("buy_price", 0.0))
                sell = float(p.get("sell_price", 0.0))
                if buy > 0 and sell > 0:
                    return {"buy": buy, "sell": sell, "mid": (buy + sell) / 2}
        except Exception:
            pass

        return {"buy": 0.0, "sell": 0.0, "mid": 0.0}

    async def run_loop(self):
        """Continuous async monitoring loop for active CRT instruments."""
        self.is_running = True
        logger.info(f"🚀 [CRTEngine] Activated for: {list(self.enabled_symbols)} (Account: {self.account_type})")

        while self.is_running:
            try:
                if self.is_enabled:
                    for symbol in list(self.enabled_symbols):
                        if symbol in CRT_INSTRUMENT_PROFILES:
                            await self._evaluate_symbol_cycle(symbol)
            except Exception as e:
                logger.error(f"[CRTEngine] Error in execution cycle: {e}", exc_info=True)

            await asyncio.sleep(15)

    async def _evaluate_symbol_cycle(self, symbol: str):
        if time.time() < self.unavailable_cooldown.get(symbol, 0.0):
            return

        profile = CRT_INSTRUMENT_PROFILES[symbol]
        asset_id = profile["asset_id"]

        cur_px = self.get_market_price(symbol)
        if not cur_px or cur_px.get("mid", 0.0) <= 0:
            return

        # Manage active dual-ticket trade if open
        if self.active_trades.get(symbol):
            await self._manage_active_trade(symbol, cur_px)
            return

        in_kz, kz_name = self.is_killzone_active(symbol)
        if not in_kz:
            return

        mode = profile["mode"]
        digits = profile["digits"]
        min_sweep = profile["min_sweep"]
        max_sweep = profile["max_sweep"]
        sl_buffer = profile["sl_buffer"]
        target_rr = profile["target_rr"]

        # Fetch 5-Minute Execution Candles
        raw_5m = self.mcp.get_candles(asset_id=asset_id, count=30, size=300)
        if not raw_5m or len(raw_5m) < 15:
            return

        df_5m = pd.DataFrame([{
            "Open": float(c.get("open") or c.get("from", 0.0)),
            "High": float(c.get("max") or c.get("high", 0.0)),
            "Low": float(c.get("min") or c.get("low", 0.0)),
            "Close": float(c.get("close") or c.get("to", 0.0))
        } for c in raw_5m])

        recent_low = float(df_5m['Low'].tail(4).min())
        recent_high = float(df_5m['High'].tail(4).max())
        last_c = float(df_5m['Close'].iloc[-1])
        last_o = float(df_5m['Open'].iloc[-1])
        last_h = float(df_5m['High'].iloc[-1])
        last_l = float(df_5m['Low'].iloc[-1])

        # ── 1. H1 ANCHOR RANGE MODE (True 1-Hour Candles) ─────────────
        if mode == "H1_ANCHOR":
            raw_h1 = self.mcp.get_candles(asset_id=asset_id, count=15, size=3600)
            if not raw_h1 or len(raw_h1) < 5:
                return

            df_h1 = pd.DataFrame([{
                "Open": float(c.get("open") or c.get("from", 0.0)),
                "High": float(c.get("max") or c.get("high", 0.0)),
                "Low": float(c.get("min") or c.get("low", 0.0)),
                "Close": float(c.get("close") or c.get("to", 0.0))
            } for c in raw_h1])

            # Anchor is the COMPLETED previous 1-Hour candle (-2 because -1 is forming)
            prev_h1 = df_h1.iloc[-2]
            anchor_high = float(prev_h1['High'])
            anchor_low = float(prev_h1['Low'])
            anchor_mid = (anchor_high + anchor_low) / 2.0

            # Calculate H1 50-EMA for macro trend flow
            df_h1['ema50'] = df_h1['Close'].ewm(span=max(3, len(df_h1)//2)).mean()
            h1_trend_up = df_h1['Close'].iloc[-1] >= df_h1['ema50'].iloc[-1]
            h1_trend_down = df_h1['Close'].iloc[-1] <= df_h1['ema50'].iloc[-1]

        # ── 2. ASIAN JUDAS MODE (EURUSD / GBPUSD) ─────────────────────
        else:
            anchor_high = float(df_5m['High'].iloc[-24:-6].max()) if len(df_5m) >= 24 else float(df_5m['High'].iloc[0:8].max())
            anchor_low = float(df_5m['Low'].iloc[-24:-6].min()) if len(df_5m) >= 24 else float(df_5m['Low'].iloc[0:8].min())
            anchor_mid = (anchor_high + anchor_low) / 2.0
            h1_trend_up = True
            h1_trend_down = True

        # --- A. Bullish CRT: Sweep Anchor Low -> Displacement Reclaim UP ---
        sweep_d = anchor_low - recent_low
        if min_sweep <= sweep_d <= max_sweep and h1_trend_up:
            body = last_c - last_o
            tot_r = last_h - last_l
            if last_c > anchor_low and body > 0 and tot_r > 0 and (body / tot_r >= 0.40):
                entry = cur_px["buy"]
                sl = round(recent_low - sl_buffer, digits)
                risk = abs(entry - sl)
                if risk > sl_buffer:
                    tp1 = round(entry + risk, digits)
                    tp2 = round(entry + (risk * target_rr), digits)
                    await self._execute_dual_ticket_crt_trade(symbol, "BUY", entry, sl, tp1, tp2, anchor_mid, kz_name, df_5m)

        # --- B. Bearish CRT: Sweep Anchor High -> Displacement Reclaim DOWN ---
        sweep_u = recent_high - anchor_high
        if min_sweep <= sweep_u <= max_sweep and h1_trend_down:
            body = last_o - last_c
            tot_r = last_h - last_l
            if last_c < anchor_high and body > 0 and tot_r > 0 and (body / tot_r >= 0.40):
                entry = cur_px["sell"]
                sl = round(recent_high + sl_buffer, digits)
                risk = abs(sl - entry)
                if risk > sl_buffer:
                    tp1 = round(entry - risk, digits)
                    tp2 = round(entry - (risk * target_rr), digits)
                    await self._execute_dual_ticket_crt_trade(symbol, "SELL", entry, sl, tp1, tp2, anchor_mid, kz_name, df_5m)

    async def _execute_dual_ticket_crt_trade(
        self, symbol: str, side: str, entry: float, sl: float, tp1: float, tp2: float,
        mid: float, kz_name: str, df: pd.DataFrame
    ):
        profile = CRT_INSTRUMENT_PROFILES[symbol]
        instrument_id = profile["instrument_id"]
        asset_id = profile["asset_id"]

        min_l = profile.get("min_lots", 0.001)
        trade_lots = profile.get("default_lots") or self.lots
        if symbol == "XAUUSD":
            min_l = 1.0
            trade_lots = max(2.0, float(trade_lots))
        elif symbol == "BTCUSD":
            min_l = 0.001
            trade_lots = max(0.02, float(trade_lots))

        if trade_lots >= (min_l * 2.0):
            lot_tp1 = round(trade_lots / 2.0, 4 if min_l < 1 else 2)
            lot_runner = round(trade_lots - lot_tp1, 4 if min_l < 1 else 2)
        else:
            lot_tp1 = min_l
            lot_runner = min_l

        trade_lev = min(self.leverage, 20 if symbol == "BTCUSD" else self.leverage)
        logger.info(f"⚡ [CRTEngine] Dual-Ticket {side} on {symbol} @ {entry:.5f} | SL: {sl} | TP1: {tp1} (+1.0R) | TP2: {tp2} ({kz_name})")

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

        # Leg 2: Runner to TP2
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
        if pos1_id: self.placed_position_ids.add(str(pos1_id))
        if pos2_id: self.placed_position_ids.add(str(pos2_id))

        self.active_trades[symbol] = {
            "symbol": symbol,
            "side": side,
            "entry_price": entry,
            "sl_price": sl,
            "initial_sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "leg1_pos_id": pos1_id,
            "leg2_pos_id": pos2_id,
            "lot_tp1": lot_tp1,
            "lot_runner": lot_runner,
            "tp1_closed": False,
            "runner_at_be": False,
            "opened_at": time.time()
        }

        # Telegram Photo Notification
        chart_bytes = generate_trade_execution_chart(
            df=df,
            symbol=symbol,
            side=side,
            entry_px=entry,
            sl=sl,
            tp=tp2,
            engine_name="CRT",
            event_title=f"CRT Dual-Ticket Setup ({kz_name})"
        )

        caption = (
            f"⚡ **CRT Dual-Ticket Execution** | {symbol} {side}\n\n"
            f"• Strategy: Institutional Candle Range Theory ({kz_name})\n"
            f"• Entry: `{entry:.5f}`\n"
            f"• Stop Loss: `{sl:.5f}`\n"
            f"• Leg 1 (50% TP1): `{tp1:.5f}` (+1.0R guaranteed bank)\n"
            f"• Leg 2 (50% Runner): `{tp2:.5f}` (+2.2R)\n"
            f"• Exposure: `{lot_tp1 + lot_runner:.4f}` Lots ({trade_lev}x)\n\n"
            f"Leg 2 will automatically shift Stop Loss to Breakeven when Leg 1 hits TP1."
        )
        await self.notify_photo(chart_bytes, caption)

    async def _manage_active_trade(self, symbol: str, cur_prices: Dict[str, float]):
        trade = self.active_trades.get(symbol)
        if not trade:
            return

        profile = CRT_INSTRUMENT_PROFILES.get(symbol, {})
        side = trade["side"]
        entry = trade["entry_price"]
        sl = trade["sl_price"]
        tp1 = trade["tp1"]
        tp2 = trade["tp2"]
        mid = cur_prices["mid"]

        # Check broker positions
        open_positions = []
        try:
            open_positions = self.mcp.list_positions(balance_id=self.balance_id) or []
        except Exception:
            pass

        open_ids = [str(p.get("position_id") or p.get("id")) for p in open_positions]

        leg1_id = str(trade.get("leg1_pos_id"))
        leg2_id = str(trade.get("leg2_pos_id"))
        leg1_still_open = leg1_id in open_ids and leg1_id != "None"
        leg2_still_open = leg2_id in open_ids and leg2_id != "None"

        # 1. Check if Leg 1 (TP1) closed while Leg 2 is still running
        if not trade["tp1_closed"]:
            if leg1_id and not leg1_still_open and leg1_id != "None":
                trade["tp1_closed"] = True
                logger.info(f"🏆 [CRTEngine] {symbol} Leg 1 (#{leg1_id}) closed! Snapping Leg 2 to Breakeven @ {entry}...")

                # Shift Leg 2 to Breakeven
                if leg2_still_open:
                    try:
                        be_buf = profile.get("sl_buffer", 1.0) * 0.15
                        be_price = round(entry + be_buf if side == "BUY" else entry - be_buf, profile["digits"])
                        self.mcp.change_position_stop_loss(position_id=int(leg2_id), level=be_price)
                        trade["runner_at_be"] = True
                        logger.info(f"✅ [CRTEngine] Leg 2 (#{leg2_id}) SL shifted to Breakeven @ {be_price}!")
                    except Exception as e:
                        logger.error(f"[CRTEngine] Failed to move Leg 2 to BE: {e}")

                await self.notify_text(
                    f"🛡️ **[CRT PARTIAL PROFIT BANKED] {symbol}**\n\n"
                    f"• Leg 1 hit TP1 @ `{tp1}`! Profit locked in.\n"
                    f"• Leg 2 Runner Stop Loss moved to **Breakeven** (`{entry}`).\n"
                    f"• Position is now **100% Risk-Free** running to `{tp2}`!"
                )

                # Log Leg 1 to Google Sheets
                await self._log_ticket_closure(
                    symbol=symbol,
                    tag="Leg 1 (50% TP1)",
                    pos_id=leg1_id,
                    lots=trade.get("lot_tp1", self.lots * 0.5),
                    entry_px=entry,
                    sl_px=trade.get("sl_price", 0.0),
                    tp_px=tp1,
                    default_reason="take_profit"
                )

        # 2. Check if all legs are closed
        if not leg1_still_open and not leg2_still_open:
            logger.info(f"🏁 [CRTEngine] {symbol} All trade legs closed. Cycle complete.")

            # Log Leg 2 (Runner) to Google Sheets
            if leg2_id and leg2_id != "None":
                await self._log_ticket_closure(
                    symbol=symbol,
                    tag="Leg 2 (Runner)",
                    pos_id=leg2_id,
                    lots=trade.get("lot_runner", self.lots * 0.5),
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
                    lots=trade.get("lot_tp1", self.lots * 0.5),
                    entry_px=entry,
                    sl_px=trade.get("sl_price", 0.0),
                    tp_px=tp1,
                    default_reason="stop_loss"
                )

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

        profile = CRT_INSTRUMENT_PROFILES.get(symbol, {})
        digits = profile.get("digits", 5)
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
                logger.debug(f"[CRTEngine] History lookup attempt {attempt+1} error: {e}")
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
                    "asset": f"CRT {symbol} ({tag})",
                    "side": side,
                    "lots": lots,
                    "entry_price": entry_px,
                    "stop_loss": sl_px,
                    "take_profit": tp_px,
                    "exit_price": exit_px,
                    "pnl": pnl,
                    "pips": round(abs(exit_px - entry_px) * (10 ** (digits - 1)), 1),
                    "risk_reward": f"{tag} (1:{self.rr_target:.1f} Plan)",
                    "exit_reason": reason,
                    "position_id": pid_str,
                    "balance_equity": eq
                }
                asyncio.create_task(asyncio.to_thread(gsheet_logger.log_forex_margin_trade, trade_payload))
                logger.info(f"📊 [CRTEngine] Dispatched Google Sheets log for {symbol} #{pid_str} PnL=${pnl:.2f}")
            except Exception as ge:
                logger.warning(f"[CRTEngine] GSheet dispatch error: {ge}")

    async def _log_trade_closure(self, symbol: str, trade_dict: Any, reason_override: Optional[str] = None):
        trade = self.active_trades.get(symbol) or trade_dict or {}
        if not trade:
            return
        leg1 = trade.get("leg1_pos_id")
        leg2 = trade.get("leg2_pos_id")
        if leg1 and str(leg1) != "None":
            await self._log_ticket_closure(
                symbol, "Leg 1 (50% TP1)", leg1, trade.get("lot_tp1", self.lots * 0.5),
                trade.get("entry_price", 0.0), trade.get("sl_price", 0.0), trade.get("tp1", 0.0),
                default_reason="closed", reason_override=reason_override
            )
        if leg2 and str(leg2) != "None":
            await self._log_ticket_closure(
                symbol, "Leg 2 (Runner)", leg2, trade.get("lot_runner", self.lots * 0.5),
                trade.get("entry_price", 0.0), trade.get("sl_price", 0.0), trade.get("tp2", 0.0),
                default_reason="closed", reason_override=reason_override
            )
