"""
strategies/snd_engine.py - Autonomous Institutional Supply & Demand + Imbalance (Config E) Engine

Dedicated autonomous strategy engine specialized for:
- NZD/USD, USD/JPY, AUD/USD, USD/CAD
- Timeframe: 15-Minute Institutional Candlesticks
- Strategy Core (Config E):
    1. Base Structure: Drop-Base-Rally (Demand) / Rally-Base-Drop (Supply).
    2. Imbalance Validation: Displacement candle >= 1.2x ATR leaving confirmed 3-candle FVG.
    3. Macro Trend: Filtered by EMA100.
    4. Mitigation Entry: Proximal edge of imbalance.
    5. Invalidation: Distal edge of base candle + 0.2x ATR buffer.
    6. Target TP: Fixed 1:2.0 Risk-to-Reward.
    7. Dynamic Breakeven: Armed at +1.0R displacement (100% Risk-Free).
"""

import time
import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple, Set, Callable
import pandas as pd
import numpy as np

from clients.forex_mcp_client import IQForexMCPClient
from gsheet_logger import gsheet_logger
from utils.chart_generator import (
    generate_trade_execution_chart,
    generate_breakeven_chart,
    generate_trade_close_chart,
    generate_ict_setup_chart
)

logger = logging.getLogger("SNDEngine")

CANDLE_SIZE = 900  # 15 minutes (900 seconds)

SND_INSTRUMENT_PROFILES: Dict[str, Dict[str, Any]] = {
    "NZDUSD": {
        "symbol": "NZDUSD",
        "name": "NZD/USD",
        "asset_id": 8,
        "instrument_id": "mf.8",
        "digits": 5,
        "default_lots": 1.0,
        "contract_size": 100000,
        "target_rr": 2.0,
        "be_trigger_r": 1.0,
        "disp_mult": 1.2,
        "sl_buf_mult": 0.2,
        "min_fvg_mult": 0.1,
        "pip_unit": 0.0001
    },
    "USDJPY": {
        "symbol": "USDJPY",
        "name": "USD/JPY",
        "asset_id": 6,
        "instrument_id": "mf.6",
        "digits": 3,
        "default_lots": 1.0,
        "contract_size": 100000,
        "target_rr": 2.0,
        "be_trigger_r": 1.0,
        "disp_mult": 1.2,
        "sl_buf_mult": 0.2,
        "min_fvg_mult": 0.1,
        "pip_unit": 0.01
    },
    "AUDUSD": {
        "symbol": "AUDUSD",
        "name": "AUD/USD",
        "asset_id": 99,
        "instrument_id": "mf.99",
        "digits": 5,
        "default_lots": 1.0,
        "contract_size": 100000,
        "target_rr": 2.0,
        "be_trigger_r": 1.0,
        "disp_mult": 1.2,
        "sl_buf_mult": 0.2,
        "min_fvg_mult": 0.1,
        "pip_unit": 0.0001
    },
    "USDCAD": {
        "symbol": "USDCAD",
        "name": "USD/CAD",
        "asset_id": 100,
        "instrument_id": "mf.100",
        "digits": 5,
        "default_lots": 1.0,
        "contract_size": 100000,
        "target_rr": 2.0,
        "be_trigger_r": 1.0,
        "disp_mult": 1.2,
        "sl_buf_mult": 0.2,
        "min_fvg_mult": 0.1,
        "pip_unit": 0.0001
    }
}


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """True Range & ATR calculation."""
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


class SNDStrategyEngine:
    """
    Autonomous Supply & Demand + Imbalance Execution Engine.
    Implements verified Config E across NZDUSD, USDJPY, AUDUSD, and USDCAD.
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

        # Active S&D symbols (Defaults to all 4 elite pairs)
        default_syms = ["NZDUSD", "USDJPY", "AUDUSD", "USDCAD"]
        self.enabled_symbols: Set[str] = set(symbols) if symbols else set(default_syms)

        # State storage per symbol
        self.pending_zones: Dict[str, List[Dict[str, Any]]] = {s: [] for s in SND_INSTRUMENT_PROFILES}
        self.active_trades: Dict[str, Optional[Dict[str, Any]]] = {s: None for s in SND_INSTRUMENT_PROFILES}
        self.unavailable_cooldown: Dict[str, float] = {}

        self.balance_id: Optional[int] = None
        self.notify_cb: Optional[Callable] = None
        self.notify_photo_cb: Optional[Callable] = None
        self.is_running = False

    def set_balance(self, balance_id: int, account_type: str = "training"):
        self.balance_id = balance_id
        self.account_type = account_type.lower()

    def set_lots(self, lots: float):
        self.lots = max(0.01, round(float(lots), 2))
        logger.info(f"📊 [SNDEngine] Lot size updated to: {self.lots}")

    def set_leverage(self, leverage: int):
        self.leverage = int(leverage)
        logger.info(f"⚡ [SNDEngine] Leverage updated to: {self.leverage}x")

    def toggle(self) -> bool:
        self.is_enabled = not self.is_enabled
        logger.info(f"🔄 [SNDEngine] Master switch toggled: {'ENABLED' if self.is_enabled else 'DISABLED'}")
        return self.is_enabled

    def toggle_symbol(self, symbol: str) -> bool:
        sym_clean = symbol.upper().replace("/", "").replace("-", "")
        if sym_clean not in SND_INSTRUMENT_PROFILES:
            return False

        if sym_clean in self.enabled_symbols:
            self.enabled_symbols.remove(sym_clean)
            self.pending_zones[sym_clean] = []
            logger.info(f"🔴 [SNDEngine] Disabled symbol: {sym_clean}")
            return False
        else:
            self.enabled_symbols.add(sym_clean)
            logger.info(f"🟢 [SNDEngine] Enabled symbol: {sym_clean}")
            return True

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.is_enabled,
            "enabled_symbols": list(self.enabled_symbols),
            "lots": self.lots,
            "leverage": self.leverage,
            "active_trades_count": len([t for t in self.active_trades.values() if t]),
            "pending_zones_count": sum(len(z) for z in self.pending_zones.values())
        }

    def set_notification_callback(self, cb: Callable):
        self.notify_cb = cb

    def set_photo_notification_callback(self, cb: Callable):
        self.notify_photo_cb = cb

    async def notify_text(self, text: str):
        if self.notify_cb:
            try:
                res = self.notify_cb(text)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as e:
                logger.error(f"[SNDEngine] Text notification failed: {e}")

    async def notify_photo(self, photo_bytes: Optional[bytes], caption: str = ""):
        if self.notify_photo_cb and photo_bytes:
            try:
                res = self.notify_photo_cb(photo_bytes, caption)
                if asyncio.iscoroutine(res):
                    await res
                return
            except Exception as e:
                logger.error(f"[SNDEngine] Photo notification failed: {e}")
                await self.notify_text(caption)
        else:
            await self.notify_text(caption)

    def get_market_price(self, symbol: str) -> Dict[str, float]:
        profile = SND_INSTRUMENT_PROFILES.get(symbol)
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
            logger.debug(f"[SNDEngine] Candle price fetch error for {symbol}: {e}")

        # Fallback to calculate_order_size
        try:
            p = self.mcp.calculate_order_size(
                asset_id=profile["asset_id"],
                balance_currency="USD",
                lots=self.lots,
                leverage=self.leverage
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
        """Continuous async monitoring loop for active S&D instruments."""
        self.is_running = True
        logger.info(f"🚀 [SNDEngine] Activated for: {list(self.enabled_symbols)} (Account: {self.account_type})")

        while self.is_running:
            try:
                if self.is_enabled:
                    for symbol in list(self.enabled_symbols):
                        if symbol in SND_INSTRUMENT_PROFILES:
                            await self._evaluate_symbol_cycle(symbol)
                            await asyncio.sleep(2.5)  # Stagger requests between assets
            except Exception as e:
                logger.error(f"[SNDEngine] Error in execution cycle: {e}", exc_info=True)

            await asyncio.sleep(45)  # 15M candles only need checking every 45-60s

    async def _evaluate_symbol_cycle(self, symbol: str):
        if time.time() < self.unavailable_cooldown.get(symbol, 0.0):
            return

        profile = SND_INSTRUMENT_PROFILES[symbol]
        asset_id = profile["asset_id"]

        # 1. Manage Active Trade if any (Fast price check)
        if self.active_trades.get(symbol):
            cur_px = self.get_market_price(symbol)
            if cur_px and cur_px.get("mid", 0.0) > 0:
                await self._manage_active_trade(symbol, cur_px)
            return

        # 2. Fetch 15M candles (120 bars = ~30 hours) - 1 single API call for price + candles
        raw_candles = self.mcp.get_candles(asset_id=asset_id, count=120, size=CANDLE_SIZE)
        if not raw_candles or len(raw_candles) < 30:
            return

        df = pd.DataFrame([{
            "Open": float(c.get("open") or c.get("from", 0.0)),
            "High": float(c.get("max") or c.get("high", 0.0)),
            "Low": float(c.get("min") or c.get("low", 0.0)),
            "Close": float(c.get("close") or c.get("to", 0.0))
        } for c in raw_candles])

        latest_close = float(df['Close'].iloc[-1])
        cur_px = {"buy": latest_close, "sell": latest_close, "mid": latest_close}

        df['ATR'] = calculate_atr(df, period=14)
        df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
        df.dropna(inplace=True)

        if len(df) < 25:
            return

        # 3. Check Pending Zones for Mitigation or Invalidation
        await self._check_pending_zones(symbol, df, cur_px)

        # 4. Scan for New S&D Base + Imbalance (Config E)
        if not self.active_trades.get(symbol):
            self._detect_new_zones(symbol, df)

    def _detect_new_zones(self, symbol: str, df: pd.DataFrame):
        profile = SND_INSTRUMENT_PROFILES[symbol]
        digits = profile["digits"]
        disp_mult = profile["disp_mult"]
        sl_buf_mult = profile["sl_buf_mult"]
        min_fvg_mult = profile["min_fvg_mult"]

        highs = df['High'].values
        lows = df['Low'].values
        opens = df['Open'].values
        closes = df['Close'].values
        atrs = df['ATR'].values
        ema100 = df['EMA100'].values

        i = len(df) - 1
        c_base = i - 2
        c_disp = i - 1
        c_conf = i

        cur_atr = atrs[c_conf]
        if cur_atr <= 0:
            return

        disp_body = abs(closes[c_disp] - opens[c_disp])

        # Existing pending check (avoid duplicates)
        existing = self.pending_zones.get(symbol, [])

        if disp_body >= (disp_mult * atrs[c_disp]):
            # --- Bullish: Drop-Base-Rally + BISI ---
            if closes[c_disp] > opens[c_disp]:
                gap = lows[c_conf] - highs[c_base]
                if gap > (min_fvg_mult * cur_atr) and closes[c_conf] > ema100[c_conf]:
                    base_low = min(lows[c_base], lows[c_disp])
                    entry = round(lows[c_conf], digits)
                    sl = round(base_low - (sl_buf_mult * cur_atr), digits)
                    distal = round(base_low, digits)

                    # Check if already registered
                    if not any(z["entry"] == entry and z["side"] == "BUY" for z in existing):
                        new_zone = {
                            "side": "BUY",
                            "entry": entry,
                            "sl": sl,
                            "distal": distal,
                            "atr": cur_atr,
                            "created_bar": i,
                            "created_time": time.time()
                        }
                        self.pending_zones[symbol].append(new_zone)
                        logger.info(f"🏛️ [SNDEngine] New Demand Zone for {symbol} @ {entry:.5f} | Distal: {distal:.5f} | SL: {sl:.5f}")

            # --- Bearish: Rally-Base-Drop + SIBI ---
            elif closes[c_disp] < opens[c_disp]:
                gap = lows[c_base] - highs[c_conf]
                if gap > (min_fvg_mult * cur_atr) and closes[c_conf] < ema100[c_conf]:
                    base_high = max(highs[c_base], highs[c_disp])
                    entry = round(highs[c_conf], digits)
                    sl = round(base_high + (sl_buf_mult * cur_atr), digits)
                    distal = round(base_high, digits)

                    if not any(z["entry"] == entry and z["side"] == "SELL" for z in existing):
                        new_zone = {
                            "side": "SELL",
                            "entry": entry,
                            "sl": sl,
                            "distal": distal,
                            "atr": cur_atr,
                            "created_bar": i,
                            "created_time": time.time()
                        }
                        self.pending_zones[symbol].append(new_zone)
                        logger.info(f"🏛️ [SNDEngine] New Supply Zone for {symbol} @ {entry:.5f} | Distal: {distal:.5f} | SL: {sl:.5f}")

    async def _check_pending_zones(self, symbol: str, df: pd.DataFrame, cur_px: Dict[str, float]):
        zones = self.pending_zones.get(symbol, [])
        if not zones:
            return

        profile = SND_INSTRUMENT_PROFILES[symbol]
        digits = profile["digits"]
        target_rr = profile["target_rr"]

        last_h = float(df['High'].iloc[-1])
        last_l = float(df['Low'].iloc[-1])
        mid = cur_px["mid"]

        rem = []
        for z in zones:
            side = z["side"]
            entry = z["entry"]
            sl = z["sl"]
            distal = z["distal"]

            # Expiration (80 bars = ~20 hours)
            if time.time() - z["created_time"] > (80 * 900):
                continue

            # Invalidation: Price breached distal line before entry
            if (side == "BUY" and last_l < distal) or (side == "SELL" and last_h > distal):
                logger.info(f"🗑️ [SNDEngine] Zone invalidated for {symbol} ({side} distal breached)")
                continue

            # Mitigation / Proximal Touch Test
            is_mitigated = False
            if side == "BUY" and (last_l <= entry or mid <= entry) and mid > sl:
                is_mitigated = True
            elif side == "SELL" and (last_h >= entry or mid >= entry) and mid < sl:
                is_mitigated = True

            if is_mitigated:
                risk = abs(entry - sl)
                if risk > 0:
                    tp = round(entry + (risk * target_rr) if side == "BUY" else entry - (risk * target_rr), digits)
                    executed = await self._execute_snd_trade(symbol, side, entry, sl, tp, risk, df)
                    if executed:
                        continue  # Do not re-add, zone is fulfilled
            rem.append(z)

        self.pending_zones[symbol] = rem

    async def _execute_snd_trade(self, symbol: str, side: str, entry: float, sl: float, tp: float, risk: float, df: pd.DataFrame) -> bool:
        profile = SND_INSTRUMENT_PROFILES[symbol]
        instrument_id = profile["instrument_id"]
        lots = profile.get("default_lots", self.lots)

        logger.info(f"⚡ [SNDEngine] Executing {side} on {symbol} @ {entry:.5f} | SL: {sl} | TP: {tp} (1:{profile['target_rr']} R:R)")

        res = self.mcp.place_market_order(
            side=side.lower(),
            balance_id=self.balance_id,
            instrument_id=instrument_id,
            asset_id=profile["asset_id"],
            lots=lots,
            leverage=self.leverage,
            stop_loss=sl,
            take_profit=tp,
            is_margin_isolated=True,
            keep_position_open=False
        )

        if not res or "error" in res or not (res.get("order_id") or res.get("position_id")):
            err_dict = res.get('error', {}) if isinstance(res, dict) else {}
            err_msg = err_dict.get('message', str(err_dict)) if isinstance(err_dict, dict) else str(res)
            logger.error(f"[SNDEngine] {symbol} Order rejected: {err_msg}")
            if "not_available" in str(err_msg).lower():
                self.unavailable_cooldown[symbol] = time.time() + 3600
                await self.notify_text(f"⚠️ S&D Order Paused | {symbol} currently unavailable on broker (cooldown 1h).")
            else:
                await self.notify_text(f"❌ S&D Order Failed | {symbol}: {err_msg}")
            return False

        order_id = res.get("order_id")
        pos_id = res.get("position_id")

        if not pos_id or pos_id == order_id:
            try:
                time.sleep(1.0)
                open_positions = self.mcp.list_positions(balance_id=self.balance_id)
                for p in open_positions:
                    if p.get("asset_id") == profile["asset_id"]:
                        pos_id = p.get("position_id") or p.get("id")
                        break
            except Exception as e:
                logger.debug(f"[SNDEngine] Error resolving position_id: {e}")

        pos_id = pos_id or order_id or f"snd_{int(time.time())}"

        self.active_trades[symbol] = {
            "position_id": pos_id,
            "order_id": order_id or pos_id,
            "symbol": symbol,
            "side": side,
            "entry_price": entry,
            "sl_price": sl,
            "initial_sl": sl,
            "tp_price": tp,
            "risk_points": risk,
            "lots": lots,
            "contract_size": profile.get("contract_size", 100000),
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
            engine_name="S&D Imbalance",
            event_title="Config E Proximal Tap"
        )

        caption = (
            f"🏛️ S&D {side} Executed | #{pos_id} {symbol}\n\n"
            f"• Strategy: Supply & Demand + Imbalance (Config E)\n"
            f"• Entry: {entry:.5f}\n"
            f"• Stop Loss: {sl:.5f}\n"
            f"• Target TP: {tp:.5f} (1:2.0 R:R)\n"
            f"• Lots: {lots:.2f} | Leverage: {self.leverage}x\n\n"
            f"🛡️ Dynamic Breakeven armed at +1.0R displacement."
        )
        await self.notify_photo(chart_bytes, caption)

        # Log opening to Google Sheets
        try:
            if hasattr(gsheet_logger, "log_forex_trade"):
                gsheet_logger.log_forex_trade(
                    symbol=symbol,
                    side=side,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    lots=lots,
                    source="SND_ENGINE"
                )
        except Exception as e:
            logger.debug(f"[SNDEngine] GSheet open log error: {e}")

        return True

    async def _manage_active_trade(self, symbol: str, cur_prices: Dict[str, float]):
        trade = self.active_trades.get(symbol)
        if not trade:
            return

        profile = SND_INSTRUMENT_PROFILES.get(symbol, {})
        pos_id = trade.get("position_id")

        # 1. Resolve position_id if needed
        if not pos_id or pos_id == trade.get("order_id"):
            try:
                open_positions = self.mcp.list_positions(balance_id=self.balance_id)
                for p in open_positions:
                    if p.get("asset_id") == profile.get("asset_id"):
                        pos_id = p.get("position_id") or p.get("id")
                        trade["position_id"] = pos_id
                        break
            except Exception as e:
                logger.debug(f"[SNDEngine] Position ID resolution error: {e}")

        # 2. Check if closed on broker directly
        if pos_id and str(pos_id).isdigit():
            try:
                open_positions = self.mcp.list_positions(balance_id=self.balance_id)
                is_still_open = any(str(p.get("position_id") or p.get("id")) == str(pos_id) for p in open_positions)
                if not is_still_open:
                    logger.info(f"🏁 [SNDEngine] {symbol} Position #{pos_id} was settled on broker platform!")
                    await self._log_trade_closure(symbol, trade)
                    self.active_trades[symbol] = None
                    return
            except Exception as e:
                logger.debug(f"[SNDEngine] Broker position state check error: {e}")

        mid = cur_prices["mid"]
        side = trade["side"]
        entry = trade["entry_price"]
        sl = trade["sl_price"]
        tp = trade["tp_price"]
        risk = trade["risk_points"]

        # 3. Breakeven Ratchet @ +1.0R
        if not trade["is_breakeven"]:
            hit_be = (mid >= entry + (risk * 1.0)) if side == "BUY" else (mid <= entry - (risk * 1.0))
            if hit_be:
                # Add tiny buffer so commission/spread is neutralized
                pip_unit = profile.get("pip_unit", 0.0001)
                be_level = entry + (1.0 * pip_unit) if side == "BUY" else entry - (1.0 * pip_unit)
                be_level = round(be_level, profile.get("digits", 5))

                trade["sl_price"] = be_level
                trade["is_breakeven"] = True
                logger.info(f"🛡️ [SNDEngine] {symbol} reached +1.0R. Ratcheting SL to Breakeven @ {be_level}!")

                if pos_id and str(pos_id).isdigit():
                    try:
                        sl_res = self.mcp.change_position_stop_loss(position_id=int(pos_id), level=be_level)
                        if sl_res and not sl_res.get("error"):
                            logger.info(f"✅ [SNDEngine] Broker SL confirmed at Breakeven on #{pos_id}!")
                        else:
                            logger.warning(f"⚠️ [SNDEngine] Broker rejected Breakeven SL update: {sl_res}")
                    except Exception as e:
                        logger.error(f"[SNDEngine] Could not update broker SL: {e}")

                chart_bytes = generate_breakeven_chart(
                    df=None,
                    symbol=symbol,
                    side=side,
                    entry_px=entry,
                    be_sl=be_level,
                    initial_sl=trade["initial_sl"],
                    tp=tp,
                    cur_px=mid,
                    engine_name="S&D Imbalance"
                )
                caption = (
                    f"🛡️ S&D Breakeven Locked | {symbol}\n\n"
                    f"• Entry: {entry:.5f}\n"
                    f"• New SL: {be_level:.5f} (Risk-Free)\n"
                    f"• Target TP: {tp:.5f} (1:2.0 R:R)\n\n"
                    f"Position is now 100% risk-free."
                )
                await self.notify_photo(chart_bytes, caption)

        # 4. Check TP or SL Hit locally
        hit_tp = (mid >= tp) if side == "BUY" else (mid <= tp)
        hit_sl = (mid <= sl) if side == "BUY" else (mid >= sl)

        if hit_tp or hit_sl:
            close_reason = "take_profit" if hit_tp else ("breakeven" if trade["is_breakeven"] else "stop_loss")

            if pos_id and str(pos_id).isdigit():
                try:
                    logger.info(f"🔒 [SNDEngine] Explicitly closing #{pos_id} on broker ({close_reason})...")
                    self.mcp.close_position(int(pos_id))
                except Exception as e:
                    logger.error(f"[SNDEngine] Broker position closure error for #{pos_id}: {e}")

            await self._log_trade_closure(symbol, trade, reason_override=close_reason)
            self.active_trades[symbol] = None

    async def _log_trade_closure(self, symbol: str, trade: dict, reason_override: Optional[str] = None):
        """Broadcast settlement card and log to Google Sheets."""
        if not trade:
            return

        pos_id = trade.get("position_id") or trade.get("order_id")
        side = trade.get("side", "BUY")
        entry = float(trade.get("entry_price", 0.0))
        lots = float(trade.get("lots", self.lots))

        pnl = 0.0
        exit_px = 0.0
        reason = reason_override or "closed"

        for attempt in range(3):
            try:
                hist = self.mcp.get_trade_history(balance_id=self.balance_id, limit=15) or []
                matched = next(
                    (h for h in hist if
                     (pos_id and str(h.get("position_id")) == str(pos_id)) or
                     (pos_id and str(h.get("order_id")) == str(pos_id)) or
                     (pos_id and str(h.get("id")) == str(pos_id))),
                    None
                )
                if matched:
                    pnl = float(matched.get("pnl", 0.0))
                    exit_px = float(matched.get("close_price", matched.get("exit_price", 0.0)))
                    reason = reason_override or matched.get("close_reason", reason)
                    break
            except Exception as e:
                logger.debug(f"[SNDEngine] Hist lookup error (attempt {attempt+1}): {e}")
            if attempt < 2:
                await asyncio.sleep(0.8)

        if exit_px == 0.0:
            prices = self.get_market_price(symbol)
            exit_px = prices.get("mid", entry)

        pnl_sign = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"

        if pnl > 0 or reason == "take_profit":
            header_line = f"🏆 [S&D WON] {symbol} {pnl_sign}"
        elif pnl == 0 or reason == "breakeven" or trade.get("is_breakeven"):
            header_line = f"🛡️ [S&D BREAKEVEN] {symbol} $0.00"
        else:
            header_line = f"❌ [S&D CLOSED] {symbol} {pnl_sign}"

        card = (
            f"{header_line}\n\n"
            f"• Strategy: Supply & Demand + Imbalance (Config E)\n"
            f"• Side: {side} ({lots} Lots)\n"
            f"• Entry: {entry:.5f}\n"
            f"• Exit: {exit_px:.5f}\n"
            f"• Net PnL: {pnl_sign}\n"
            f"• Reason: {reason}"
        )
        await self.notify_text(card)
