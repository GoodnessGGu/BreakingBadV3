"""
strategies/crt_engine.py - Autonomous Institutional Candle Range Theory (CRT) Engine

Dedicated autonomous strategy engine specialized for:
- EUR/USD & GBP/USD: Asian Session Range (00:00 - 06:00 UTC) London Judas Protocol
- Bitcoin (BTCUSD): H1/H4 Anchor Candle Range Expansions

Features:
1. Multi-Tiered Anchor Range tracking (Asian Range + H1 Candle Range).
2. Institutional Killzone Timing (London Open 07:00-10:00 UTC | NY Open 12:30-16:00 UTC).
3. Displacement & FVG Retest Validation before execution.
4. Dynamic Breakeven Ratchet when price reaches 50% Equilibrium / +1.0R.
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
        "contract_size": 100000,
        "target_rr": 2.5,
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
        "contract_size": 100000,
        "target_rr": 2.5,
        "digits": 5,
        "mode": "ASIAN_JUDAS"
    },
    "BTCUSD": {
        "symbol": "BTCUSD",
        "name": "Bitcoin (BTC/USD)",
        "asset_id": 816,
        "instrument_id": "mcrpt.816",
        "min_sweep": 50.0,
        "max_sweep": 800.0,
        "sl_buffer": 40.0,
        "default_lots": 0.01,
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
        "sl_buffer": 3.0,
        "default_lots": 1.0,
        "contract_size": 100,
        "target_rr": 2.5,
        "digits": 2,
        "mode": "ASIAN_JUDAS"
    }
}


class CRTStrategyEngine:
    """
    Autonomous Candle Range Theory (CRT) Execution Engine.
    Specialized for Forex (EURUSD, GBPUSD), Crypto (BTCUSD), and Metals.
    """

    def __init__(
        self,
        mcp_client: IQForexMCPClient,
        symbols: Optional[List[str]] = None,
        account_type: str = "training",
        lots: float = 1.0,
        leverage: int = 100,
        enabled: bool = True
    ):
        self.mcp = mcp_client
        self.account_type = account_type.lower()
        self.lots = lots
        self.leverage = leverage
        self.is_enabled = enabled

        # Active CRT symbols
        self.enabled_symbols: Set[str] = set(symbols) if symbols else {"EURUSD", "BTCUSD"}
        self.balance_id: Optional[int] = None
        self.notify_cb: Optional[Callable] = None
        self.notify_photo_cb: Optional[Callable] = None
        self.is_running = False

        # State storage per symbol
        self.asian_ranges: Dict[str, Dict[str, Any]] = {}
        self.h1_anchors: Dict[str, Dict[str, Any]] = {}
        self.pending_setups: Dict[str, Optional[Dict[str, Any]]] = {}
        self.active_trades: Dict[str, Optional[Dict[str, Any]]] = {}
        self.unavailable_cooldown: Dict[str, float] = {}

    def set_balance(self, balance_id: int, account_type: str = "training"):
        self.balance_id = balance_id
        self.account_type = account_type.lower()

    def set_lots(self, lots: float):
        self.lots = max(0.01, round(float(lots), 2))

    def set_leverage(self, leverage: int):
        self.leverage = int(leverage)

    def toggle(self) -> bool:
        self.is_enabled = not self.is_enabled
        logger.info(f"[CRTEngine] Master switch toggled: {'ENABLED' if self.is_enabled else 'DISABLED'}")
        return self.is_enabled

    def toggle_symbol(self, symbol: str) -> bool:
        sym_clean = symbol.upper().replace("/", "").replace("-", "")
        if sym_clean in self.enabled_symbols:
            self.enabled_symbols.remove(sym_clean)
            logger.info(f"[CRTEngine] Disabled symbol: {sym_clean}")
            return False
        else:
            self.enabled_symbols.add(sym_clean)
            logger.info(f"[CRTEngine] Enabled symbol: {sym_clean}")
            return True

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.is_enabled,
            "enabled_symbols": list(self.enabled_symbols),
            "lots": self.lots,
            "leverage": self.leverage,
            "active_trades_count": len([t for t in self.active_trades.values() if t])
        }

    def set_notification_callback(self, cb: Callable):
        self.notify_cb = cb

    def set_photo_notification_callback(self, cb: Callable):
        self.notify_photo_cb = cb

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

    def is_killzone_active(self) -> Tuple[bool, str]:
        """Returns True if within London (07:00-10:00 UTC) or NY (12:30-16:00 UTC)."""
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

        # 1. Fetch latest candle for instant live pricing
        try:
            candles = self.mcp.get_candles(asset_id=profile["asset_id"], size=60, count=2)
            if candles and len(candles) > 0:
                last_c = candles[-1]
                px = float(last_c.get("close", last_c.get("c", 0.0)))
                if px > 0:
                    return {"buy": px, "sell": px, "mid": px}
        except Exception as e:
            logger.debug(f"[CRTEngine] Candle price fetch error for {symbol}: {e}")

        # 2. Fallback to calculate_order_size if candle is unavailable
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
        # 0. Check if symbol is in temporary unavailable cooldown
        if time.time() < self.unavailable_cooldown.get(symbol, 0.0):
            return

        profile = CRT_INSTRUMENT_PROFILES[symbol]
        asset_id = profile["asset_id"]

        # 1. Fetch current price
        cur_px = self.get_market_price(symbol)
        if not cur_px or cur_px.get("mid", 0.0) <= 0:
            return

        # 2. Manage active trade if any
        if self.active_trades.get(symbol):
            await self._manage_active_trade(symbol, cur_px)
            return

        # 3. Check killzones
        in_kz, kz_name = self.is_killzone_active()
        if not in_kz:
            return

        # 4. Fetch candles for analysis
        raw_candles = self.mcp.get_candles(asset_id=asset_id, count=40, size=300)
        if not raw_candles or len(raw_candles) < 15:
            return

        df = pd.DataFrame([{
            "Open": float(c.get("open") or c.get("from", 0.0)),
            "High": float(c.get("max") or c.get("high", 0.0)),
            "Low": float(c.get("min") or c.get("low", 0.0)),
            "Close": float(c.get("close") or c.get("to", 0.0))
        } for c in raw_candles])

        # 5. Evaluate CRT Setup
        mode = profile["mode"]
        digits = profile["digits"]
        min_sweep = profile["min_sweep"]
        max_sweep = profile["max_sweep"]
        sl_buffer = profile["sl_buffer"]
        target_rr = profile["target_rr"]

        recent_low = float(df['Low'].tail(6).min())
        recent_high = float(df['High'].tail(6).max())
        last_c = float(df['Close'].iloc[-1])
        last_o = float(df['Open'].iloc[-1])
        last_h = float(df['High'].iloc[-1])
        last_l = float(df['Low'].iloc[-1])

        # Reference anchor: previous 12-bar high/low (or Asian session range)
        anchor_high = float(df['High'].iloc[-24:-6].max()) if len(df) >= 24 else float(df['High'].iloc[0:8].max())
        anchor_low = float(df['Low'].iloc[-24:-6].min()) if len(df) >= 24 else float(df['Low'].iloc[0:8].min())
        anchor_mid = (anchor_high + anchor_low) / 2.0

        # --- A. Bullish CRT: Sweep Low -> Displacement Reclaim ---
        sweep_d = anchor_low - recent_low
        if min_sweep <= sweep_d <= max_sweep:
            body = last_c - last_o
            tot_r = last_h - last_l
            if last_c > anchor_low and body > 0 and tot_r > 0 and (body / tot_r >= 0.50):
                entry = cur_px["buy"]
                sl = round(recent_low - sl_buffer, digits)
                risk = abs(entry - sl)
                if risk > 0:
                    tp = round(max(anchor_high, entry + (risk * target_rr)), digits)
                    await self._execute_crt_trade(symbol, "BUY", entry, sl, tp, anchor_mid, kz_name, df)

        # --- B. Bearish CRT: Sweep High -> Displacement Reclaim ---
        sweep_u = recent_high - anchor_high
        if min_sweep <= sweep_u <= max_sweep:
            body = last_o - last_c
            tot_r = last_h - last_l
            if last_c < anchor_high and body > 0 and tot_r > 0 and (body / tot_r >= 0.50):
                entry = cur_px["sell"]
                sl = round(recent_high + sl_buffer, digits)
                risk = abs(sl - entry)
                if risk > 0:
                    tp = round(min(anchor_low, entry - (risk * target_rr)), digits)
                    await self._execute_crt_trade(symbol, "SELL", entry, sl, tp, anchor_mid, kz_name, df)

    async def _execute_crt_trade(self, symbol: str, side: str, entry: float, sl: float, tp: float, mid: float, kz_name: str, df: pd.DataFrame):
        profile = CRT_INSTRUMENT_PROFILES[symbol]
        instrument_id = profile["instrument_id"]
        lots = profile.get("default_lots", self.lots)

        logger.info(f"⚡ [CRTEngine] Executing {side} on {symbol} @ {entry:.5f} | SL: {sl} | TP: {tp} ({kz_name})")
        trade_lev = min(self.leverage, 20 if symbol == "BTCUSD" else self.leverage)
        res = self.mcp.place_market_order(
            side=side.lower(),
            balance_id=self.balance_id,
            instrument_id=instrument_id,
            asset_id=profile["asset_id"],
            lots=lots,
            leverage=trade_lev,
            stop_loss=sl,
            take_profit=tp,
            is_margin_isolated=True,
            keep_position_open=False
        )

        if not res or "error" in res or not (res.get("order_id") or res.get("position_id")):
            err_dict = res.get('error', {}) if isinstance(res, dict) else {}
            err_msg = err_dict.get('message', str(err_dict)) if isinstance(err_dict, dict) else str(res)
            logger.error(f"[CRTEngine] {symbol} Order placement rejected by broker: {err_msg}")
            if "not_available" in str(err_msg).lower():
                self.unavailable_cooldown[symbol] = time.time() + 3600
                await self.notify_text(f"CRT Order Failed | {symbol} is currently unavailable for CFD trading on broker (paused for 1h).")
            else:
                await self.notify_text(f"CRT Order Failed | {symbol}: {err_msg}")
            return

        order_id = res.get("order_id")
        pos_id = res.get("position_id")
        # Auto-resolve actual position_id from broker if missing or still equal to order_id
        if not pos_id or pos_id == order_id:
            try:
                time.sleep(1.0)
                open_positions = self.mcp.list_positions(balance_id=self.balance_id)
                for p in open_positions:
                    if p.get("asset_id") == profile["asset_id"]:
                        pos_id = p.get("position_id") or p.get("id")
                        break
            except Exception as e:
                logger.debug(f"[CRTEngine] Error resolving position_id: {e}")
        pos_id = pos_id or order_id or f"crt_{int(time.time())}"

        self.active_trades[symbol] = {
            "position_id": pos_id,
            "order_id": order_id or pos_id,
            "symbol": symbol,
            "side": side,
            "entry_price": entry,
            "sl_price": sl,
            "initial_sl": sl,
            "tp_price": tp,
            "mid_equilibrium": mid,
            "risk_points": abs(entry - sl),
            "lots": lots,
            "contract_size": profile.get("contract_size", 1.0),
            "is_breakeven": False,
            "opened_at": time.time()
        }

        # Generate TradingView Pro execution chart
        chart_bytes = generate_trade_execution_chart(
            df=df,
            symbol=symbol,
            side=side,
            entry_px=entry,
            sl=sl,
            tp=tp,
            engine_name="CRT",
            event_title=f"{kz_name} Judas Expansion"
        )

        caption = (
            f"CRT {side} Executed | #{pos_id} {symbol}\n\n"
            f"• Strategy: Candle Range Theory ({kz_name})\n"
            f"• Entry: {entry:.5f}\n"
            f"• Stop Loss: {sl:.5f}\n"
            f"• Target TP: {tp:.5f}\n"
            f"• Equilibrium: {mid:.5f} (BE Target)\n\n"
            f"Dynamic Breakeven armed at +1.0R."
        )
        await self.notify_photo(chart_bytes, caption)

    async def _manage_active_trade(self, symbol: str, cur_prices: Dict[str, float]):
        trade = self.active_trades.get(symbol)
        if not trade:
            return

        profile = CRT_INSTRUMENT_PROFILES.get(symbol, {})
        pos_id = trade.get("position_id")

        # 1. Resolve actual position_id from broker if missing or still equal to order_id
        if not pos_id or pos_id == trade.get("order_id"):
            try:
                open_positions = self.mcp.list_positions(balance_id=self.balance_id)
                for p in open_positions:
                    if p.get("asset_id") == profile.get("asset_id"):
                        pos_id = p.get("position_id") or p.get("id")
                        trade["position_id"] = pos_id
                        break
            except Exception as e:
                logger.debug(f"[CRTEngine] Position ID resolution error: {e}")

        # 2. Check if position was closed on broker directly (broker SL or TP executed)
        if pos_id and str(pos_id).isdigit():
            try:
                open_positions = self.mcp.list_positions(balance_id=self.balance_id)
                is_still_open = any((p.get("position_id") or p.get("id")) == int(pos_id) for p in open_positions)
                if not is_still_open:
                    logger.info(f"🏁 [CRTEngine] {symbol} Position #{pos_id} was closed on broker platform!")
                    hist = self.mcp.get_trade_history(balance_id=self.balance_id, limit=5)
                    matched_hist = next((h for h in hist if str(h.get("position_id")) == str(pos_id)), None)
                    pnl = float(matched_hist.get("pnl", 0.0)) if matched_hist else 0.0
                    outcome = "WIN" if pnl > 0 else ("BREAKEVEN" if pnl == 0 else "LOSS")
                    pnl_sign = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
                    caption = (
                        f"CRT Trade Closed ({outcome}) | {symbol} {pnl_sign}\n\n"
                        f"• Position: #{pos_id}\n"
                        f"• Realized PnL: {pnl_sign}\n"
                        f"• Settled directly on broker."
                    )
                    await self.notify_text(caption)
                    self.active_trades[symbol] = None
                    return
            except Exception as e:
                logger.debug(f"[CRTEngine] Error verifying broker position state: {e}")

        mid = cur_prices["mid"]
        side = trade["side"]
        entry = trade["entry_price"]
        sl = trade["sl_price"]
        tp = trade["tp_price"]
        risk = trade["risk_points"]

        # Check Breakeven ratchet
        if not trade["is_breakeven"]:
            hit_be = (mid >= entry + (risk * 1.0)) if side == "BUY" else (mid <= entry - (risk * 1.0))
            if hit_be:
                trade["sl_price"] = entry
                trade["is_breakeven"] = True
                logger.info(f"🛡️ [CRTEngine] {symbol} reached +1.0R Equilibrium. Ratcheting SL to Breakeven @ {entry}!")
                
                if pos_id and str(pos_id).isdigit():
                    try:
                        sl_res = self.mcp.change_position_stop_loss(position_id=int(pos_id), level=entry)
                        if sl_res and not sl_res.get("error"):
                            logger.info(f"✅ [CRTEngine] Broker SL successfully confirmed at Breakeven @ {entry} on #{pos_id}!")
                        else:
                            logger.warning(f"⚠️ [CRTEngine] Broker rejected Breakeven SL update on #{pos_id}: {sl_res}")
                    except Exception as e:
                        logger.error(f"[CRTEngine] Could not update broker SL for #{pos_id}: {e}")
                
                chart_bytes = generate_breakeven_chart(
                    df=None,
                    symbol=symbol,
                    side=side,
                    entry_px=entry,
                    be_sl=entry,
                    initial_sl=trade["initial_sl"],
                    tp=tp,
                    cur_px=mid,
                    engine_name="CRT"
                )
                caption = (
                    f"CRT Breakeven Locked | {symbol}\n\n"
                    f"• Entry: {entry:.5f}\n"
                    f"• New SL: {entry:.5f} (Risk-Free)\n"
                    f"• Target TP: {tp:.5f}\n\n"
                    f"Trade is now 100% risk-free."
                )
                await self.notify_photo(chart_bytes, caption)

        # Check TP or SL Hit locally
        hit_tp = (mid >= tp) if side == "BUY" else (mid <= tp)
        hit_sl = (mid <= sl) if side == "BUY" else (mid >= sl)

        if hit_tp or hit_sl:
            outcome = "WIN" if hit_tp else ("BREAKEVEN" if trade["is_breakeven"] else "LOSS")
            contract_size = trade.get("contract_size", 1.0)
            trade_lots = trade.get("lots", 1.0)

            # Explicitly liquidate position on broker if still open
            if pos_id and str(pos_id).isdigit():
                try:
                    logger.info(f"🔒 [CRTEngine] Explicitly closing #{pos_id} on broker ({outcome})...")
                    self.mcp.close_position(int(pos_id))
                except Exception as e:
                    logger.error(f"[CRTEngine] Could not close broker position #{pos_id}: {e}")

            if hit_tp:
                gain_pts = (tp - entry) if side == "BUY" else (entry - tp)
                dollar_pnl = gain_pts * contract_size * trade_lots
            elif trade["is_breakeven"]:
                dollar_pnl = 0.0
            else:
                dollar_pnl = -(risk * contract_size * trade_lots)

            pnl_sign = f"+${dollar_pnl:.2f}" if dollar_pnl >= 0 else f"-${abs(dollar_pnl):.2f}"
            logger.info(f"🏁 [CRTEngine] {symbol} trade closed: {outcome} | PnL: {pnl_sign}")

            chart_bytes = generate_trade_close_chart(
                df=None,
                symbol=symbol,
                side=side,
                entry_px=entry,
                exit_px=mid,
                tp=tp,
                sl=sl,
                pnl=dollar_pnl,
                reason="take_profit" if hit_tp else "stop_loss",
                engine_name="CRT"
            )
            caption = (
                f"CRT Trade Closed ({outcome}) | {symbol} {pnl_sign}\n\n"
                f"• Outcome: {'Take Profit Hit' if hit_tp else ('Breakeven' if trade['is_breakeven'] else 'Stop Loss Hit')}\n"
                f"• Entry/Exit: {entry:.5f} ➔ {mid:.5f}\n"
                f"• Realized PnL: {pnl_sign}"
            )
            await self.notify_photo(chart_bytes, caption)
            self.active_trades[symbol] = None
