"""
strategies/ict_engine.py - Multi-Instrument Autonomous ICT / SMC Strategy Engine

Trades Gold (XAUUSD) or selected Forex pairs on IQ Option Marginal CFD engine.
Strategy:
  1. 15M Candlestick Orderflow & Liquidity Sweeps.
  2. Displacement + Fair Value Gap (FVG) creation.
  3. Dynamic Entry Zone on FVG retest.
  4. Auto-Breakeven at 1.0R profit.
  5. 1:2.0 Risk-to-Reward Ratio with tight SL at the sweep extreme.
  6. Live Google Sheets logging to "Forex_Margin_Trades".
"""

import time
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
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
        "sl_buffer": 2.5,
        "disp_threshold": 2.0,
        "min_fvg_gap": 0.3,
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
        "default_lots": 1.0,
        "digits": 5
    }
}

class ICTStrategyEngine:
    def __init__(self, mcp_client: IQForexMCPClient, symbol: str = "XAUUSD",
                 account_type: str = "training", lots: float = 1.0,
                 leverage: int = 100, rr_ratio: float = 2.0, enabled: bool = True):
        self.mcp = mcp_client
        self.account_type = account_type.lower()
        self.lots = lots
        self.leverage = leverage
        self.rr_ratio = rr_ratio
        self.is_enabled = enabled

        self.balance_id: Optional[int] = None
        self.active_trade: Optional[Dict[str, Any]] = None
        self.pending_fvg: Optional[Dict[str, Any]] = None
        self.notify_cb: Optional[Callable] = None
        self.is_running = False

        self.set_instrument(symbol)

    def set_notification_callback(self, cb: Callable):
        self.notify_cb = cb

    async def notify(self, message: str):
        if self.notify_cb:
            try:
                await self.notify_cb(message)
            except Exception as e:
                logger.warning(f"[ICTEngine] Notification error: {e}")

    def set_instrument(self, symbol: str) -> bool:
        sym = symbol.upper().replace("/", "").replace("-", "")
        if sym in INSTRUMENT_PROFILES:
            self.profile = INSTRUMENT_PROFILES[sym]
            self.symbol = sym
            self.asset_id = self.profile["asset_id"]
            self.instrument_id = self.profile["instrument_id"]
            self.sl_buffer = self.profile["sl_buffer"]
            self.disp_threshold = self.profile["disp_threshold"]
            self.min_fvg_gap = self.profile["min_fvg_gap"]
            self.digits = self.profile["digits"]
            self.pending_fvg = None
            logger.info(f"🎯 [ICTEngine] Switched instrument to {self.profile['name']} (ID: {self.asset_id})")
            return True
        logger.warning(f"⚠️ Unknown instrument: {symbol}")
        return False

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
        logger.info("🟢 [ICTEngine] Enabled")

    def disable(self):
        self.is_enabled = False
        logger.info("🔴 [ICTEngine] Disabled")

    def toggle(self) -> bool:
        self.is_enabled = not self.is_enabled
        logger.info(f"🔄 [ICTEngine] Toggled -> {'ENABLED' if self.is_enabled else 'DISABLED'}")
        return self.is_enabled

    def get_market_price(self) -> Dict[str, float]:
        try:
            p = self.mcp.calculate_order_size(
                asset_id=self.asset_id, balance_currency="USD",
                lots=self.lots, leverage=self.leverage
            )
            buy = float(p.get("buy_price", 0.0))
            sell = float(p.get("sell_price", 0.0))
            return {"buy": buy, "sell": sell, "mid": (buy + sell) / 2}
        except Exception as e:
            logger.warning(f"[ICTEngine] Price fetch error: {e}")
            return {"buy": 0.0, "sell": 0.0, "mid": 0.0}

    def fetch_recent_candles(self, count: int = 50) -> Optional[pd.DataFrame]:
        try:
            candles = self.mcp.get_candles(asset_id=self.asset_id, size=CANDLE_SIZE, count=count)
            if not candles or len(candles) < 15:
                return None
            df = pd.DataFrame(candles)
            df.rename(columns={"open": "Open", "close": "Close", "min": "Low", "max": "High"}, inplace=True)
            for col in ["Open", "Close", "Low", "High"]:
                df[col] = df[col].astype(float)
            return df
        except Exception as e:
            logger.error(f"[ICTEngine] Error fetching candles: {e}")
            return None

    async def scan_for_setups(self, df: pd.DataFrame):
        if not self.is_enabled or self.active_trade or self.pending_fvg:
            return

        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        opens = df['Open'].values

        recent_high = max(highs[-25:-5])
        recent_low = min(lows[-25:-5])

        # 1. Bearish Liquidity Sweep (High swept + Bearish FVG)
        swept_h = (highs[-3] > recent_high and closes[-3] < recent_high) or \
                  (highs[-2] > recent_high and closes[-2] < recent_high)
        has_bearish_fvg = lows[-3] > (highs[-1] + self.min_fvg_gap)
        disp_down = closes[-2] < opens[-2] and (highs[-2] - lows[-2]) > self.disp_threshold

        if swept_h and has_bearish_fvg and disp_down:
            sweep_peak = max(highs[-3], highs[-2])
            sl = round(sweep_peak + self.sl_buffer, self.digits)
            fvg_h = round(lows[-3], self.digits)
            fvg_l = round(highs[-1], self.digits)

            logger.info("=" * 60)
            logger.info(f"🔥 [ICT SETUP DETECTED] {self.symbol} Bearish Liquidity Sweep at {sweep_peak}!")
            logger.info(f"   Bearish FVG Zone : {fvg_l} - {fvg_h} | SL: {sl}")
            logger.info("=" * 60)

            await self.notify(
                f"🔥 [ICT SETUP DETECTED — {self.symbol} SELL]\n"
                f"Sweep Peak: {sweep_peak}\n"
                f"FVG Zone  : {fvg_l} – {fvg_h}\n"
                f"Stop Loss : {sl}\n"
                f"⏳ Waiting for price retest..."
            )
            self.pending_fvg = {
                "side": "SELL",
                "fvg_high": fvg_h,
                "fvg_low": fvg_l,
                "sl": sl,
                "detected_at": time.time()
            }
            return

        # 2. Bullish Liquidity Sweep (Low swept + Bullish FVG)
        swept_l = (lows[-3] < recent_low and closes[-3] > recent_low) or \
                  (lows[-2] < recent_low and closes[-2] > recent_low)
        has_bullish_fvg = highs[-3] < (lows[-1] - self.min_fvg_gap)
        disp_up = closes[-2] > opens[-2] and (highs[-2] - lows[-2]) > self.disp_threshold

        if swept_l and has_bullish_fvg and disp_up:
            sweep_trough = min(lows[-3], lows[-2])
            sl = round(sweep_trough - self.sl_buffer, self.digits)
            fvg_l = round(highs[-3], self.digits)
            fvg_h = round(lows[-1], self.digits)

            logger.info("=" * 60)
            logger.info(f"🔥 [ICT SETUP DETECTED] {self.symbol} Bullish Liquidity Sweep at {sweep_trough}!")
            logger.info(f"   Bullish FVG Zone : {fvg_l} - {fvg_h} | SL: {sl}")
            logger.info("=" * 60)

            await self.notify(
                f"🔥 [ICT SETUP DETECTED — {self.symbol} BUY]\n"
                f"Sweep Trough: {sweep_trough}\n"
                f"FVG Zone    : {fvg_l} – {fvg_h}\n"
                f"Stop Loss   : {sl}\n"
                f"⏳ Waiting for price retest..."
            )
            self.pending_fvg = {
                "side": "BUY",
                "fvg_high": fvg_h,
                "fvg_low": fvg_l,
                "sl": sl,
                "detected_at": time.time()
            }

    async def check_fvg_retest_and_enter(self, cur_prices: Dict[str, float]):
        if not self.is_enabled or not self.pending_fvg or self.active_trade:
            return

        mid = cur_prices["mid"]
        side = self.pending_fvg["side"]
        fvg_h = self.pending_fvg["fvg_high"]
        fvg_l = self.pending_fvg["fvg_low"]
        sl = self.pending_fvg["sl"]

        if time.time() - self.pending_fvg["detected_at"] > (3 * 3600):
            logger.info(f"⏰ [ICTEngine] {self.symbol} FVG expired without retest. Resetting.")
            self.pending_fvg = None
            return

        in_fvg = (fvg_l <= mid <= fvg_h)
        if in_fvg:
            exec_px = cur_prices["buy"] if side == "BUY" else cur_prices["sell"]
            risk_dist = abs(exec_px - sl)
            if risk_dist < (self.min_fvg_gap * 0.5):
                return

            tp = round(exec_px + (risk_dist * self.rr_ratio) if side == "BUY" else exec_px - (risk_dist * self.rr_ratio), self.digits)

            await self.notify(
                f"⚡ [ICT EXECUTION — {self.symbol} {side}]\n"
                f"Entry : {exec_px} (FVG Retest)\n"
                f"SL    : {sl}\n"
                f"TP    : {tp} (1:{self.rr_ratio:.1f} RR)\n"
                f"Lots  : {self.lots}"
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
                logger.info(f"✅ [ICTEngine] Order filled! ID: #{order_id}")
                self.active_trade = {
                    "order_id": order_id,
                    "position_id": None,
                    "symbol": self.symbol,
                    "side": side,
                    "entry_price": exec_px,
                    "initial_sl": sl,
                    "current_sl": sl,
                    "tp": tp,
                    "moved_to_be": False,
                    "opened_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                self.pending_fvg = None
                await self.notify(f"✅ [ICTEngine] Order Filled! ID: #{order_id}")
            else:
                logger.error(f"❌ [ICTEngine] Order placement failed: {res}")
                await self.notify(f"❌ [ICTEngine] Order Failed: {res.get('error', res)}")

    async def manage_active_trade(self, cur_prices: Dict[str, float]):
        if not self.active_trade:
            return

        if not self.active_trade["position_id"]:
            positions = self.mcp.list_positions(balance_id=self.balance_id)
            for p in positions:
                if p.get("asset_id") == self.asset_id:
                    self.active_trade["position_id"] = p.get("position_id") or p.get("id")
                    break

        pos_id = self.active_trade["position_id"]
        if not pos_id:
            return

        open_positions = self.mcp.list_positions(balance_id=self.balance_id)
        is_still_open = any((p.get("position_id") or p.get("id")) == pos_id for p in open_positions)

        if not is_still_open:
            logger.info(f"ICT Trade #{pos_id} closed! Syncing Google Sheets...")
            await self._log_trade_closure(pos_id)
            self.active_trade = None
            return

        # Breakeven check
        if not self.active_trade["moved_to_be"]:
            mid = cur_prices["mid"]
            entry = self.active_trade["entry_price"]
            risk_dist = abs(entry - self.active_trade["initial_sl"])
            side = self.active_trade["side"]

            hit_1r = (mid - entry >= risk_dist) if side == "BUY" else (entry - mid >= risk_dist)
            if hit_1r:
                be_buf = self.min_fvg_gap * 0.5
                be_level = round(entry + be_buf if side == "BUY" else entry - be_buf, self.digits)
                logger.info(f"🛡️ [ICT BREAKEVEN] Reached 1.0R profit! Moving SL to {be_level}")
                res = self.mcp.change_position_stop_loss(position_id=pos_id, level=be_level)
                if not res.get("error"):
                    self.active_trade["current_sl"] = be_level
                    self.active_trade["moved_to_be"] = True
                    await self.notify(f"🛡️ [ICT BREAKEVEN ACTIVATED]\nPosition #{pos_id} SL shifted to {be_level}")

    async def _log_trade_closure(self, pos_id: int):
        try:
            history = self.mcp.get_trade_history(balance_id=self.balance_id, limit=5)
            matched = next((h for h in history if h.get("position_id") == pos_id), None)
            pnl = float(matched.get("pnl", 0.0)) if matched else 0.0
            exit_px = float(matched.get("close_price", 0.0)) if matched else 0.0
            reason = matched.get("close_reason", "closed") if matched else "closed"
            entry_px = self.active_trade["entry_price"]

            bal = self.mcp.get_training_balance() if self.account_type == "training" else self.mcp.get_real_balance()
            eq = bal.get("equity", 0.0) if bal else 0.0

            gsheet_logger.log_forex_margin_trade({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "asset": f"ICT {self.symbol}",
                "side": self.active_trade["side"],
                "lots": self.lots,
                "entry_price": entry_px,
                "stop_loss": self.active_trade["current_sl"],
                "take_profit": self.active_trade["tp"],
                "exit_price": exit_px,
                "pnl": pnl,
                "pips": round(abs(exit_px - entry_px) * (10 ** (self.digits - 1)), 1),
                "risk_reward": f"1:{self.rr_ratio:.1f} (ICT Autonomous)",
                "exit_reason": reason,
                "position_id": pos_id,
                "balance_equity": eq
            })

            emoji = "🏆 WIN" if pnl > 0 else "❌ LOSS"
            await self.notify(
                f"{emoji} [ICT TRADE CLOSED]\n"
                f"Instrument : {self.symbol}\n"
                f"Position   : #{pos_id}\n"
                f"PnL        : {'+' if pnl >= 0 else ''}${pnl:.2f}\n"
                f"Exit Price : {exit_px}\n"
                f"Reason     : {reason}"
            )
        except Exception as e:
            logger.error(f"[ICTEngine] Error logging trade closure: {e}")

    async def run_loop(self):
        self.is_running = True
        logger.info(f"🚀 [ICTEngine] Started autonomous loop for {self.symbol}")
        last_candle_scan = 0

        while self.is_running:
            try:
                if self.is_enabled:
                    now = time.time()
                    prices = self.get_market_price()

                    if prices["mid"] > 0:
                        if now - last_candle_scan > 60:
                            df = self.fetch_recent_candles(count=50)
                            if df is not None:
                                await self.scan_for_setups(df)
                            last_candle_scan = now

                        await self.check_fvg_retest_and_enter(prices)
                        await self.manage_active_trade(prices)

                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[ICTEngine] Unexpected loop error: {e}")
                await asyncio.sleep(10)

    def get_status(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.profile["name"],
            "enabled": self.is_enabled,
            "lots": self.lots,
            "leverage": self.leverage,
            "rr_ratio": self.rr_ratio,
            "pending_fvg": self.pending_fvg,
            "active_trade": self.active_trade
        }
