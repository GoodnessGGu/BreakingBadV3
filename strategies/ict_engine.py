"""
strategies/ict_engine.py - Concurrent Multi-Asset Autonomous ICT / SMC Strategy Engine

Trades Gold (XAUUSD), Bitcoin (BTCUSD), and Forex pairs simultaneously on IQ Option Marginal CFD engine.
Strategy:
  1. 15M Candlestick Orderflow & Liquidity Sweeps per active asset.
  2. Change in State of Delivery (CISD) Confirmation.
  3. Displacement + Fair Value Gap (FVG) creation.
  4. Dynamic Entry Zone on FVG retest.
  5. Auto-Breakeven at 1.0R profit.
  6. Calibrated Risk-to-Reward Ratio (1:2.0 / 1:2.2 / 1:2.5).
  7. Live Google Sheets logging to "Forex_Margin_Trades".
"""

import time
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List, Set, Callable
import pandas as pd
from clients.forex_mcp_client import IQForexMCPClient
from gsheet_logger import gsheet_logger

logger = logging.getLogger("ICTEngine")

CANDLE_SIZE = 900  # 15 minutes (900s)

INSTRUMENT_PROFILES = {
    "XAUUSD": {
        "symbol": "XAUUSD",
        "name": "Gold (XAU/USD)",
        "asset_id": 74,
        "instrument_id": "mcfd.74",
        "sl_buffer": 3.0,
        "disp_threshold": 2.0,
        "min_fvg_gap": 0.3,
        "body_ratio_req": 0.50,
        "default_lots": 1.0,
        "digits": 2
    },
    "BTCUSD": {
        "symbol": "BTCUSD",
        "name": "Bitcoin (BTC/USD)",
        "asset_id": 816,
        "instrument_id": "mcfd.816",
        "sl_buffer": 50.0,
        "disp_threshold": 160.0,
        "min_fvg_gap": 20.0,
        "body_ratio_req": 0.50,
        "default_lots": 1.0,
        "digits": 2
    },
    "EURUSD": {
        "symbol": "EURUSD",
        "name": "EUR/USD",
        "asset_id": 1,
        "instrument_id": "mcfd.1",
        "sl_buffer": 0.0003,
        "disp_threshold": 0.0004,
        "min_fvg_gap": 0.0001,
        "body_ratio_req": 0.50,
        "default_lots": 1.0,
        "digits": 5
    },
    "GBPUSD": {
        "symbol": "GBPUSD",
        "name": "GBP/USD",
        "asset_id": 5,
        "instrument_id": "mcfd.5",
        "sl_buffer": 0.0004,
        "disp_threshold": 0.0005,
        "min_fvg_gap": 0.0001,
        "body_ratio_req": 0.50,
        "default_lots": 1.0,
        "digits": 5
    },
    "USDJPY": {
        "symbol": "USDJPY",
        "name": "USD/JPY",
        "asset_id": 6,
        "instrument_id": "mcfd.6",
        "sl_buffer": 0.04,
        "disp_threshold": 0.05,
        "min_fvg_gap": 0.01,
        "body_ratio_req": 0.50,
        "default_lots": 1.0,
        "digits": 3
    },
    "AUDUSD": {
        "symbol": "AUDUSD",
        "name": "AUD/USD",
        "asset_id": 99,
        "instrument_id": "mcfd.99",
        "sl_buffer": 0.0003,
        "disp_threshold": 0.0004,
        "min_fvg_gap": 0.0001,
        "body_ratio_req": 0.50,
        "default_lots": 1.0,
        "digits": 5
    }
}

class ICTStrategyEngine:
    def __init__(self, mcp_client: IQForexMCPClient, symbol: Optional[str] = None,
                 account_type: str = "training", lots: float = 1.0,
                 leverage: int = 100, rr_ratio: float = 2.2, enabled: bool = True):
        self.mcp = mcp_client
        self.account_type = account_type.lower()
        self.lots = lots
        self.leverage = leverage
        self.rr_ratio = rr_ratio
        self.is_enabled = enabled

        # Multi-asset state: Defaults to Gold + Bitcoin active
        self.enabled_symbols: Set[str] = {"XAUUSD", "BTCUSD"}
        if symbol:
            sym_clean = symbol.upper().replace("/", "").replace("-", "")
            if sym_clean in INSTRUMENT_PROFILES:
                self.enabled_symbols.add(sym_clean)
        self.pending_fvgs: Dict[str, Optional[Dict[str, Any]]] = {}
        self.active_trades: Dict[str, Optional[Dict[str, Any]]] = {}
        
        self.balance_id: Optional[int] = None
        self.notify_cb: Optional[Callable] = None
        self.is_running = False

    @property
    def pending_fvg(self) -> Optional[Dict[str, Any]]:
        for s, f in self.pending_fvgs.items():
            if f:
                return f
        return None

    @property
    def active_trade(self) -> Optional[Dict[str, Any]]:
        for s, t in self.active_trades.items():
            if t:
                return t
        return None

    def set_notification_callback(self, cb: Callable):
        self.notify_cb = cb

    async def notify(self, message: str):
        if self.notify_cb:
            try:
                await self.notify_cb(message)
            except Exception as e:
                logger.warning(f"[ICTEngine] Notification error: {e}")

    def toggle_symbol(self, symbol: str) -> bool:
        sym = symbol.upper().replace("/", "").replace("-", "")
        if sym == "BTC":
            sym = "BTCUSD"
        if sym == "GOLD" or sym == "XAU":
            sym = "XAUUSD"

        if sym not in INSTRUMENT_PROFILES:
            logger.warning(f"⚠️ Unknown instrument: {symbol}")
            return False

        if sym in self.enabled_symbols:
            self.enabled_symbols.remove(sym)
            self.pending_fvgs.pop(sym, None)
            logger.info(f"🔴 [ICTEngine] Disabled asset: {sym}")
            return False
        else:
            self.enabled_symbols.add(sym)
            logger.info(f"🟢 [ICTEngine] Enabled asset: {sym}")
            return True

    def set_instrument(self, symbol: str) -> bool:
        """Helper to toggle or ensure an instrument is active."""
        return self.toggle_symbol(symbol)

    def set_balance(self, balance_id: int, account_type: str = "training"):
        self.balance_id = balance_id
        self.account_type = account_type

    def set_lots(self, lots: float):
        self.lots = max(0.01, round(float(lots), 2))
        logger.info(f"📊 [ICTEngine] Lot size set to: {self.lots}")

    def set_leverage(self, leverage: int):
        self.leverage = int(leverage)
        logger.info(f"⚡ [ICTEngine] Leverage set to: {self.leverage}x")

    def enable(self):
        self.is_enabled = True
        logger.info("🟢 [ICTEngine] Master switch Enabled")

    def disable(self):
        self.is_enabled = False
        logger.info("🔴 [ICTEngine] Master switch Disabled")

    def toggle(self) -> bool:
        self.is_enabled = not self.is_enabled
        logger.info(f"🔄 [ICTEngine] Toggled -> {'ENABLED' if self.is_enabled else 'DISABLED'}")
        return self.is_enabled

    def get_market_price(self, symbol: str) -> Dict[str, float]:
        profile = INSTRUMENT_PROFILES.get(symbol)
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
            logger.debug(f"[ICTEngine] Candle price fetch error for {symbol}: {e}")

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

    def fetch_recent_candles(self, symbol: str, count: int = 50) -> Optional[pd.DataFrame]:
        profile = INSTRUMENT_PROFILES.get(symbol)
        if not profile:
            return None
        try:
            candles = self.mcp.get_candles(asset_id=profile["asset_id"], size=CANDLE_SIZE, count=count)
            if not candles or len(candles) < 15:
                return None
            df = pd.DataFrame(candles)
            df.rename(columns={"open": "Open", "close": "Close", "min": "Low", "max": "High"}, inplace=True)
            for col in ["Open", "Close", "Low", "High"]:
                df[col] = df[col].astype(float)
            return df
        except Exception as e:
            logger.error(f"[ICTEngine] Error fetching candles for {symbol}: {e}")
            return None

    async def scan_for_setups(self, symbol: str, df: pd.DataFrame):
        if not self.is_enabled or self.active_trades.get(symbol) or self.pending_fvgs.get(symbol):
            return

        profile = INSTRUMENT_PROFILES[symbol]
        sl_buffer = profile["sl_buffer"]
        disp_threshold = profile["disp_threshold"]
        min_fvg_gap = profile["min_fvg_gap"]
        body_ratio_req = profile.get("body_ratio_req", 0.50)
        digits = profile["digits"]

        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        opens = df['Open'].values

        recent_high = max(highs[-25:-5])
        recent_low = min(lows[-25:-5])

        # 1. Bearish Liquidity Sweep (High swept + CISD + Bearish FVG + Solid Displacement)
        swept_h = (highs[-3] > recent_high and closes[-3] < recent_high) or \
                  (highs[-2] > recent_high and closes[-2] < recent_high)
        has_bearish_fvg = lows[-3] > (highs[-1] + min_fvg_gap)
        disp_range_down = highs[-2] - lows[-2]
        disp_body_down = abs(closes[-2] - opens[-2])
        disp_down = closes[-2] < opens[-2] and disp_range_down > disp_threshold
        body_ok_down = (disp_body_down / max(0.0001, disp_range_down)) >= body_ratio_req

        # Explicit ICT CISD: Displacement closes below the Open of the high-forming candle
        sweep_open_h = opens[-3] if highs[-3] >= highs[-2] else opens[-2]
        cisd_down = (closes[-2] < sweep_open_h) or (closes[-1] < sweep_open_h)

        if swept_h and has_bearish_fvg and disp_down and body_ok_down and cisd_down:
            sweep_peak = max(highs[-3], highs[-2])
            sl = round(sweep_peak + sl_buffer, digits)
            fvg_h = round(lows[-3], digits)
            fvg_l = round(highs[-1], digits)

            logger.info("=" * 60)
            logger.info(f"🔥 [ICT CISD SETUP DETECTED] {symbol} Bearish Sweep at {sweep_peak} | CISD Open: {sweep_open_h}!")
            logger.info(f"   Bearish FVG Zone : {fvg_l} - {fvg_h} | SL: {sl}")
            logger.info("=" * 60)

            await self.notify(
                f"🔥 [ICT CISD SETUP DETECTED — {symbol} SELL]\n"
                f"Sweep Peak : {sweep_peak}\n"
                f"CISD Shift : Broken below {sweep_open_h}\n"
                f"FVG Zone   : {fvg_l} – {fvg_h}\n"
                f"Stop Loss  : {sl}\n"
                f"⏳ Waiting for FVG retest..."
            )
            self.pending_fvgs[symbol] = {
                "symbol": symbol,
                "side": "SELL",
                "fvg_high": fvg_h,
                "fvg_low": fvg_l,
                "sl": sl,
                "detected_at": time.time()
            }
            return

        # 2. Bullish Liquidity Sweep (Low swept + CISD + Bullish FVG + Solid Displacement)
        swept_l = (lows[-3] < recent_low and closes[-3] > recent_low) or \
                  (lows[-2] < recent_low and closes[-2] > recent_low)
        has_bullish_fvg = highs[-3] < (lows[-1] - min_fvg_gap)
        disp_range_up = highs[-2] - lows[-2]
        disp_body_up = abs(closes[-2] - opens[-2])
        disp_up = closes[-2] > opens[-2] and disp_range_up > disp_threshold
        body_ok_up = (disp_body_up / max(0.0001, disp_range_up)) >= body_ratio_req

        # Explicit ICT CISD: Displacement closes above the Open of the low-forming candle
        sweep_open_l = opens[-3] if lows[-3] <= lows[-2] else opens[-2]
        cisd_up = (closes[-2] > sweep_open_l) or (closes[-1] > sweep_open_l)

        if swept_l and has_bullish_fvg and disp_up and body_ok_up and cisd_up:
            sweep_trough = min(lows[-3], lows[-2])
            sl = round(sweep_trough - sl_buffer, digits)
            fvg_l = round(highs[-3], digits)
            fvg_h = round(lows[-1], digits)

            logger.info("=" * 60)
            logger.info(f"🔥 [ICT CISD SETUP DETECTED] {symbol} Bullish Sweep at {sweep_trough} | CISD Open: {sweep_open_l}!")
            logger.info(f"   Bullish FVG Zone : {fvg_l} - {fvg_h} | SL: {sl}")
            logger.info("=" * 60)

            await self.notify(
                f"🔥 [ICT CISD SETUP DETECTED — {symbol} BUY]\n"
                f"Sweep Trough: {sweep_trough}\n"
                f"CISD Shift  : Broken above {sweep_open_l}\n"
                f"FVG Zone    : {fvg_l} – {fvg_h}\n"
                f"Stop Loss   : {sl}\n"
                f"⏳ Waiting for FVG retest..."
            )
            self.pending_fvgs[symbol] = {
                "symbol": symbol,
                "side": "BUY",
                "fvg_high": fvg_h,
                "fvg_low": fvg_l,
                "sl": sl,
                "detected_at": time.time()
            }

    async def check_fvg_retest_and_enter(self, symbol: str, cur_prices: Dict[str, float]):
        pending = self.pending_fvgs.get(symbol)
        if not self.is_enabled or not pending or self.active_trades.get(symbol):
            return

        profile = INSTRUMENT_PROFILES[symbol]
        digits = profile["digits"]
        min_fvg_gap = profile["min_fvg_gap"]

        mid = cur_prices["mid"]
        side = pending["side"]
        fvg_h = pending["fvg_high"]
        fvg_l = pending["fvg_low"]
        sl = pending["sl"]

        if time.time() - pending["detected_at"] > (3 * 3600):
            logger.info(f"⏰ [ICTEngine] {symbol} FVG expired without retest. Resetting.")
            self.pending_fvgs.pop(symbol, None)
            return

        in_fvg = (fvg_l <= mid <= fvg_h)
        if in_fvg:
            exec_px = cur_prices["buy"] if side == "BUY" else cur_prices["sell"]
            risk_dist = abs(exec_px - sl)
            if risk_dist < (min_fvg_gap * 0.5):
                return

            tp = round(exec_px + (risk_dist * self.rr_ratio) if side == "BUY" else exec_px - (risk_dist * self.rr_ratio), digits)

            await self.notify(
                f"⚡ [ICT EXECUTION — {symbol} {side}]\n"
                f"Entry : {exec_px} (FVG Retest)\n"
                f"SL    : {sl}\n"
                f"TP    : {tp} (1:{self.rr_ratio:.1f} RR)\n"
                f"Lots  : {self.lots}"
            )

            trade_lev = min(self.leverage, 20 if symbol == "BTCUSD" else self.leverage)
            res = self.mcp.place_market_order(
                side=side.lower(),
                balance_id=self.balance_id,
                instrument_id=profile["instrument_id"],
                asset_id=profile["asset_id"],
                lots=self.lots,
                leverage=trade_lev,
                stop_loss=sl,
                take_profit=tp,
                is_margin_isolated=True,
                keep_position_open=False
            )

            if "order_id" in res:
                order_id = res["order_id"]
                logger.info(f"✅ [ICTEngine] {symbol} Order filled! ID: #{order_id}")
                self.active_trades[symbol] = {
                    "order_id": order_id,
                    "position_id": None,
                    "symbol": symbol,
                    "side": side,
                    "entry_price": exec_px,
                    "initial_sl": sl,
                    "current_sl": sl,
                    "tp": tp,
                    "moved_to_be": False,
                    "opened_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                self.pending_fvgs.pop(symbol, None)
                await self.notify(f"✅ [ICTEngine] {symbol} Order Filled! ID: #{order_id}")
            else:
                logger.error(f"❌ [ICTEngine] {symbol} Order placement failed: {res}")
                await self.notify(f"❌ [ICTEngine] {symbol} Order Failed: {res.get('error', res)}")

    async def manage_active_trade(self, symbol: str, cur_prices: Dict[str, float]):
        trade = self.active_trades.get(symbol)
        if not trade:
            return

        profile = INSTRUMENT_PROFILES[symbol]
        digits = profile["digits"]
        asset_id = profile["asset_id"]

        if not trade["position_id"]:
            positions = self.mcp.list_positions(balance_id=self.balance_id)
            for p in positions:
                if p.get("asset_id") == asset_id:
                    trade["position_id"] = p.get("position_id") or p.get("id")
                    break

        pos_id = trade["position_id"]
        if not pos_id:
            return

        open_positions = self.mcp.list_positions(balance_id=self.balance_id)
        is_still_open = any((p.get("position_id") or p.get("id")) == pos_id for p in open_positions)

        if not is_still_open:
            logger.info(f"ICT {symbol} Trade #{pos_id} closed! Syncing Google Sheets...")
            await self._log_trade_closure(symbol, pos_id)
            self.active_trades.pop(symbol, None)
            return

        # Breakeven check
        if not trade["moved_to_be"]:
            mid = cur_prices["mid"]
            entry = trade["entry_price"]
            risk_dist = abs(entry - trade["initial_sl"])
            side = trade["side"]

            hit_1r = (mid - entry >= risk_dist) if side == "BUY" else (entry - mid >= risk_dist)
            if hit_1r:
                be_buf = profile["min_fvg_gap"] * 0.5
                be_level = round(entry + be_buf if side == "BUY" else entry - be_buf, digits)
                logger.info(f"🛡️ [ICT BREAKEVEN] {symbol} Reached 1.0R profit! Moving SL to {be_level}")
                res = self.mcp.change_position_stop_loss(position_id=pos_id, level=be_level)
                if not res.get("error"):
                    trade["current_sl"] = be_level
                    trade["moved_to_be"] = True
                    await self.notify(f"🛡️ [ICT BREAKEVEN ACTIVATED — {symbol}]\nPosition #{pos_id} SL shifted to {be_level}")

    async def _log_trade_closure(self, symbol: str, pos_id: int):
        trade = self.active_trades.get(symbol)
        if not trade:
            return
        profile = INSTRUMENT_PROFILES[symbol]
        digits = profile["digits"]
        try:
            history = self.mcp.get_trade_history(balance_id=self.balance_id, limit=5)
            matched = next((h for h in history if h.get("position_id") == pos_id), None)
            pnl = float(matched.get("pnl", 0.0)) if matched else 0.0
            exit_px = float(matched.get("close_price", 0.0)) if matched else 0.0
            reason = matched.get("close_reason", "closed") if matched else "closed"
            entry_px = trade["entry_price"]

            bal = self.mcp.get_training_balance() if self.account_type == "training" else self.mcp.get_real_balance()
            eq = bal.get("equity", 0.0) if bal else 0.0

            gsheet_logger.log_forex_margin_trade({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "asset": f"ICT {symbol}",
                "side": trade["side"],
                "lots": self.lots,
                "entry_price": entry_px,
                "stop_loss": trade["current_sl"],
                "take_profit": trade["tp"],
                "exit_price": exit_px,
                "pnl": pnl,
                "pips": round(abs(exit_px - entry_px) * (10 ** (digits - 1)), 1),
                "risk_reward": f"1:{self.rr_ratio:.1f} (ICT Autonomous)",
                "exit_reason": reason,
                "position_id": pos_id,
                "balance_equity": eq
            })

            emoji = "🏆 WIN" if pnl > 0 else "❌ LOSS"
            await self.notify(
                f"{emoji} [ICT TRADE CLOSED — {symbol}]\n"
                f"Position   : #{pos_id}\n"
                f"PnL        : {'+' if pnl >= 0 else ''}${pnl:.2f}\n"
                f"Exit Price : {exit_px}\n"
                f"Reason     : {reason}"
            )
        except Exception as e:
            logger.error(f"[ICTEngine] Error logging trade closure for {symbol}: {e}")

    async def run_loop(self):
        self.is_running = True
        logger.info(f"🚀 [ICTEngine] Started concurrent multi-asset loop: {list(self.enabled_symbols)}")
        last_candle_scan: Dict[str, float] = {}

        while self.is_running:
            try:
                if self.is_enabled and self.enabled_symbols:
                    now = time.time()
                    for sym in list(self.enabled_symbols):
                        prices = self.get_market_price(sym)
                        if prices["mid"] > 0:
                            if now - last_candle_scan.get(sym, 0) > 60:
                                df = self.fetch_recent_candles(sym, count=50)
                                if df is not None:
                                    await self.scan_for_setups(sym, df)
                                last_candle_scan[sym] = now

                            await self.check_fvg_retest_and_enter(sym, prices)
                            await self.manage_active_trade(sym, prices)

                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[ICTEngine] Unexpected loop error: {e}")
                await asyncio.sleep(10)

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.is_enabled,
            "enabled_symbols": list(self.enabled_symbols),
            "symbol": ", ".join(self.enabled_symbols) if self.enabled_symbols else "None",
            "name": f"Multi-Asset ({len(self.enabled_symbols)} Active)",
            "lots": self.lots,
            "leverage": self.leverage,
            "rr_ratio": self.rr_ratio,
            "pending_fvgs": {k: v for k, v in self.pending_fvgs.items() if v},
            "active_trades": {k: v for k, v in self.active_trades.items() if v}
        }
