"""
smart_trail_forex_bot.py - Smart Trail Margin Forex Bot (Zero Martingale, Asymmetric 1:2 R:R)

Adapts TradingView's 'Smart Trail Signals NO CONDITIONS' indicator to real Margin Forex
trading on IQ Option's official marginal-forex engine via Model Context Protocol (MCP).

Core Principles:
1. Zero Martingale: Mathematical risk management with fixed equity percentage risk (1%).
2. Dynamic Volatility Stops: Stop Loss placed precisely at the Smart Trail ATR line.
3. Asymmetric Take Profit: Configured at 1:2 or 1:3 Risk-to-Reward ratio.
4. Ratchet Trailing Stop: Stop Loss moves with the Smart Trail as price moves in profit.
5. Live Google Sheets Sync: Real-time logging to 'Forex_Margin_Trades' tab with colored PnL.
"""

import os
import sys
import time
import json
import logging
import argparse
import math
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Ensure root directory is on path
sys.path.append(os.getcwd())

from forex_mcp_client import IQForexMCPClient
from gsheet_logger import gsheet_logger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SmartTrailForex")

# Mapping of standard pair names to MCP asset IDs
PAIR_TO_ASSET_ID = {
    "EURUSD": 1,
    "EUR/USD": 1,
    "EURGBP": 2,
    "EUR/GBP": 2,
    "GBPJPY": 3,
    "GBP/JPY": 3,
    "EURJPY": 4,
    "EUR/JPY": 4,
    "GBPUSD": 5,
    "GBP/USD": 5,
    "USDJPY": 6,
    "USD/JPY": 6,
    "AUDCAD": 7,
    "AUD/CAD": 7,
    "NZDUSD": 8,
    "NZD/USD": 8,
    "USDCHF": 72,
    "USD/CHF": 72,
    "AUDUSD": 99,
    "AUD/USD": 99,
    "USDCAD": 100,
    "USD/CAD": 100
}

def calculate_smart_trail(df: pd.DataFrame, length: int = 14, multiplier: float = 2.0, sensitivity: int = 3) -> pd.DataFrame:
    """
    Direct Python implementation of TradingView's 'Smart Trail Signals NO CONDITIONS'.
    Returns DataFrame with:
    - smart_trend: +1 (Bullish) or -1 (Bearish)
    - smart_trail_value: Trailing support/resistance level
    - smart_bull: True on Bullish trend flip
    - smart_bear: True on Bearish trend flip
    - atr: Current True Range EMA
    """
    high = pd.to_numeric(df['max'] if 'max' in df.columns else df['high']).values
    low = pd.to_numeric(df['min'] if 'min' in df.columns else df['low']).values
    close = pd.to_numeric(df['close']).values
    n = len(df)

    if n < length + 2:
        return df

    # 1. True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    # 2. Wilder's smoothed ATR (RMA)
    atr = pd.Series(tr).ewm(alpha=1.0 / length, adjust=False).mean().values
    volatility_factor = atr * multiplier * (sensitivity / 3.0)

    basic_up = close - volatility_factor
    basic_down = close + volatility_factor

    trail_up = np.zeros(n)
    trail_down = np.zeros(n)
    smart_trend = np.ones(n, dtype=int)

    trail_up[0] = basic_up[0]
    trail_down[0] = basic_down[0]

    for i in range(1, n):
        # Smart Trail UP (Support for Bullish Trend)
        if close[i] > trail_up[i - 1] and close[i - 1] > trail_up[i - 1]:
            trail_up[i] = max(trail_up[i - 1], basic_up[i])
        else:
            trail_up[i] = basic_up[i]

        # Smart Trail DOWN (Resistance for Bearish Trend)
        if close[i] < trail_down[i - 1] and close[i - 1] < trail_down[i - 1]:
            trail_down[i] = min(trail_down[i - 1], basic_down[i])
        else:
            trail_down[i] = basic_down[i]

        # Determine trend flip
        if close[i] > trail_down[i - 1]:
            smart_trend[i] = 1
        elif close[i] < trail_up[i - 1]:
            smart_trend[i] = -1
        else:
            smart_trend[i] = smart_trend[i - 1]

    df['smart_trend'] = smart_trend
    df['smart_trail_value'] = np.where(smart_trend == 1, trail_up, trail_down)
    df['atr'] = atr

    # Trend change detection (signal fires only on bar where trend shifts)
    df['smart_bull'] = (smart_trend == 1) & (np.roll(smart_trend, 1) == -1)
    df['smart_bear'] = (smart_trend == -1) & (np.roll(smart_trend, 1) == 1)
    df.loc[0, ['smart_bull', 'smart_bear']] = False

    return df

class SmartTrailForexBot:
    def __init__(self, account_type: str = "training", risk_pct: float = 1.0,
                 risk_reward: float = 2.0, leverage: int = 50, candle_size: int = 60,
                 assets: Optional[List[str]] = None, max_open_positions: int = 2):
        self.account_type = account_type.lower()
        self.risk_pct = risk_pct
        self.risk_reward = risk_reward
        self.leverage = leverage
        self.candle_size = candle_size
        self.assets = assets or ["EURUSD", "GBPUSD", "USDJPY", "EURGBP", "AUDUSD"]
        self.max_open_positions = max_open_positions
        
        self.client = IQForexMCPClient()
        self.balance_id: Optional[int] = None
        self.currency: str = "USD"
        self.equity: float = 0.0
        self.free_margin: float = 0.0
        
        # In-memory tracking of managed positions: {position_id: {details}}
        self.managed_positions: Dict[int, Dict[str, Any]] = {}
        # Last signal timestamp per asset to avoid re-triggering on same bar
        self.last_signal_time: Dict[str, str] = {}

    def connect(self) -> bool:
        """Initialize MCP session and retrieve balance."""
        logger.info("Initializing IQ Option MCP Forex engine...")
        if not self.client.initialize():
            logger.error("❌ Failed to connect to IQ Option MCP server.")
            return False

        self.update_account_state()
        if not self.balance_id:
            logger.error("❌ Could not determine balance ID for trading.")
            return False

        logger.info(f"✅ Connected! Account: {self.account_type.upper()} | Balance ID: {self.balance_id} | Equity: ${self.equity:.2f} | Free Margin: ${self.free_margin:.2f}")
        return True

    def update_account_state(self):
        """Fetch current balance, equity, and open positions."""
        bal = self.client.get_training_balance() if self.account_type == "training" else self.client.get_real_balance()
        if bal:
            self.balance_id = bal.get("balance_id")
            self.currency = bal.get("currency", "USD")
            self.equity = float(bal.get("equity", 0.0))
            self.free_margin = float(bal.get("free_margin", 0.0))
            
        # Re-sync open positions
        if self.balance_id:
            open_pos_list = self.client.list_positions(balance_id=self.balance_id)
            current_ids = set()
            for p in open_pos_list:
                pos_id = p.get("position_id") or p.get("id")
                current_ids.add(pos_id)
                if pos_id not in self.managed_positions:
                    # Found an open position not previously tracked or from prior run
                    self.managed_positions[pos_id] = {
                        "position_id": pos_id,
                        "asset_id": p.get("asset_id"),
                        "asset_name": p.get("asset_name"),
                        "side": "BUY" if p.get("type") == "long" else "SELL",
                        "lots": p.get("count", 0.0) / 100000.0 if p.get("count", 0) >= 100 else p.get("count", 0.0),
                        "entry_price": float(p.get("open_price", 0.0)),
                        "stop_loss": float(p.get("stop_lose_price", 0.0)),
                        "take_profit": float(p.get("take_profit_price", 0.0)),
                        "entry_time": p.get("open_time")
                    }

            # Check if any managed positions closed
            closed_ids = [pid for pid in self.managed_positions if pid not in current_ids]
            for pid in closed_ids:
                self._handle_position_closure(pid)

    def _handle_position_closure(self, position_id: int):
        """Handle position closure: look up trade history, compute PnL, and log to Google Sheets."""
        pos_data = self.managed_positions.pop(position_id, {})
        logger.info(f"🔔 Position #{position_id} closed! Querying trade history for details...")
        
        # Look up in closed trade history
        history = self.client.get_trade_history(balance_id=self.balance_id, limit=20)
        matched = None
        for h in history:
            if h.get("position_id") == position_id:
                matched = h
                break

        if matched:
            pnl = float(matched.get("pnl", 0.0))
            entry_price = float(matched.get("open_price", pos_data.get("entry_price", 0.0)))
            exit_price = float(matched.get("close_price", 0.0))
            reason = matched.get("close_reason", "server_exit")
            asset_name = matched.get("asset_name", pos_data.get("asset_name", "Forex"))
            side = "BUY" if matched.get("type") == "long" else "SELL"
            lots = pos_data.get("lots", 0.001)
            
            # Compute pip movement
            pip_size = 0.01 if "JPY" in asset_name else 0.0001
            pips = ((exit_price - entry_price) / pip_size) if side == "BUY" else ((entry_price - exit_price) / pip_size)
            
            logger.info(f"🏆 Position #{position_id} Result: {asset_name} {side} | PnL: ${pnl:.2f} ({pips:+.1f} pips) | Exit: {reason} @ {exit_price}")
            
            # Google Sheets logging
            gsheet_logger.log_forex_margin_trade({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "asset": asset_name,
                "side": side,
                "lots": lots,
                "entry_price": entry_price,
                "stop_loss": pos_data.get("stop_loss", 0.0),
                "take_profit": pos_data.get("take_profit", 0.0),
                "exit_price": exit_price,
                "pnl": pnl,
                "pips": pips,
                "risk_reward": f"1:{self.risk_reward}",
                "exit_reason": reason,
                "position_id": position_id,
                "balance_equity": self.equity
            })
        else:
            logger.warning(f"Could not find #{position_id} in recent trade history. Logging estimated exit.")
            gsheet_logger.log_forex_margin_trade({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "asset": pos_data.get("asset_name", "Forex"),
                "side": pos_data.get("side", "BUY"),
                "lots": pos_data.get("lots", 0.001),
                "entry_price": pos_data.get("entry_price", 0.0),
                "stop_loss": pos_data.get("stop_loss", 0.0),
                "take_profit": pos_data.get("take_profit", 0.0),
                "exit_price": pos_data.get("entry_price", 0.0),
                "pnl": 0.0,
                "pips": 0.0,
                "risk_reward": f"1:{self.risk_reward}",
                "exit_reason": "closed",
                "position_id": position_id,
                "balance_equity": self.equity
            })

    def process_asset(self, pair: str):
        """Scan candle data, compute Smart Trail, check signals, and manage open positions."""
        clean_pair = pair.replace("/", "").upper()
        asset_id = PAIR_TO_ASSET_ID.get(clean_pair)
        if not asset_id:
            asset_meta = self.client.get_asset(clean_pair)
            if asset_meta:
                asset_id = asset_meta.get("asset_id")
            else:
                logger.warning(f"Asset ID not found for {pair}")
                return

        # 1. Fetch historical candles
        candles = self.client.get_candles(asset_id=asset_id, size=self.candle_size, count=60)
        if not candles or len(candles) < 20:
            return

        df = pd.DataFrame(candles)
        for col in ['open', 'close', 'min', 'max', 'high', 'low']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])

        # 2. Calculate Smart Trail
        df = calculate_smart_trail(df, length=14, multiplier=2.0, sensitivity=3)
        curr = df.iloc[-1]
        candle_time = curr.get('to') or curr.get('from', '')
        
        curr_price = float(curr['close'])
        smart_trail = float(curr['smart_trail_value'])
        atr = float(curr['atr'])

        # 3. Check if we already have an open position on this asset
        existing_pos = None
        for pid, pdata in self.managed_positions.items():
            if pdata.get("asset_id") == asset_id:
                existing_pos = pdata
                break

        # If we have an open position, handle dynamic trailing stop ratchet
        if existing_pos:
            self._update_trailing_stop(existing_pos, curr_price, smart_trail, asset_id)
            return

        # If max open positions reached, do not open new ones
        if len(self.managed_positions) >= self.max_open_positions:
            return

        # Check for Smart Trail trend flips
        is_bull = bool(curr['smart_bull'])
        is_bear = bool(curr['smart_bear'])

        if not is_bull and not is_bear:
            return

        # Avoid repeated entries on the same bar
        if self.last_signal_time.get(clean_pair) == candle_time:
            return

        # Breakout Exhaustion Filter: Skip if candle body is > 2.5x ATR
        candle_body = abs(curr_price - float(curr['open']))
        prev_atr = float(df['atr'].iloc[-2]) if len(df) > 1 else atr
        if prev_atr > 0 and (candle_body / prev_atr) > 2.5:
            logger.warning(f"⚠️ [{pair}] Skipping overextended breakout candle ({candle_body:.5f} > 2.5x ATR).")
            return

        side = "BUY" if is_bull else "SELL"
        self.last_signal_time[clean_pair] = candle_time
        
        logger.info(f"🔥 [SmartTrail FLIP] {pair} Signal: {side} | Price: {curr_price:.5f} | Trail Line: {smart_trail:.5f} | ATR: {atr:.5f}")
        
        # Execute margin order with dynamic SL and TP
        self._execute_smart_trade(pair, asset_id, side, curr_price, smart_trail, atr)

    def _execute_smart_trade(self, pair: str, asset_id: int, side: str,
                             entry_price: float, smart_trail: float, atr: float):
        """Size position for 1% equity risk and execute market order with SL and TP."""
        inst_info = self.client.get_instruments(asset_id)
        if not inst_info or not inst_info.get("instruments"):
            logger.error(f"Failed to get instrument metadata for {pair}")
            return
            
        inst = inst_info["instruments"][0]
        instrument_id = inst.get("instrument_id", f"mf.{asset_id}")
        stop_levels = inst.get("stop_levels", {})
        min_sl_dist = float(stop_levels.get("stop_loss", 0.0002))
        min_tp_dist = float(stop_levels.get("take_profit", 0.0002))

        # 1. Stop Loss directly at Smart Trail line
        if side == "BUY":
            raw_risk = entry_price - smart_trail
            if raw_risk < min_sl_dist:
                raw_risk = min_sl_dist * 1.5
            sl_price = round(entry_price - raw_risk, 5)
            tp_price = round(entry_price + (raw_risk * self.risk_reward), 5)
        else: # SELL
            raw_risk = smart_trail - entry_price
            if raw_risk < min_sl_dist:
                raw_risk = min_sl_dist * 1.5
            sl_price = round(entry_price + raw_risk, 5)
            tp_price = round(entry_price - (raw_risk * self.risk_reward), 5)

        # 2. Risk USD = 1.0% of account equity
        risk_usd = max(0.50, self.equity * (self.risk_pct / 100.0))
        
        # 3. Compute exact lots for risk_usd
        lots, sizing = self.client.calculate_lot_size(
            asset_id=asset_id,
            entry_price=entry_price,
            sl_price=sl_price,
            risk_usd=risk_usd,
            balance_currency=self.currency,
            leverage=self.leverage,
            free_margin=self.free_margin
        )
        
        if lots <= 0:
            logger.error(f"Calculated lots for {pair} is 0. Aborting order.")
            return

        margin_req = sizing.get("margin", 0.0)
        logger.info(f"📊 Order Setup [{pair} {side}]: Lots={lots}, Risk=${risk_usd:.2f} (1%), Margin=${margin_req:.2f}, SL={sl_price}, TP={tp_price} (1:{self.risk_reward} R:R)")

        # 4. Place Market Order via MCP
        order_res = self.client.place_market_order(
            side=side,
            balance_id=self.balance_id,
            instrument_id=instrument_id,
            asset_id=asset_id,
            lots=lots,
            leverage=self.leverage,
            stop_loss=sl_price,
            take_profit=tp_price,
            is_margin_isolated=True,
            keep_position_open=False
        )

        if "order_id" in order_res:
            order_id = order_res["order_id"]
            logger.info(f"✅ Order Placed! Order ID: {order_id}. Waiting for fill confirmation...")
            time.sleep(2)
            self.update_account_state()
        else:
            logger.error(f"❌ Order placement failed for {pair}: {order_res.get('error', order_res)}")

    def _update_trailing_stop(self, pos: Dict[str, Any], curr_price: float,
                              smart_trail: float, asset_id: int):
        """Ratchet the stop-loss order along the Smart Trail volatility line."""
        pos_id = pos["position_id"]
        side = pos["side"]
        curr_sl = pos.get("stop_loss", 0.0)
        
        inst_info = self.client.get_instruments(asset_id)
        stop_levels = inst_info.get("instruments", [{}])[0].get("stop_levels", {}) if inst_info else {}
        min_dist = float(stop_levels.get("stop_loss", 0.0002))

        if side == "BUY":
            # For BUY: SL only moves UP
            # Ensure new SL is above current SL, and at least min_dist below curr_price
            if smart_trail > (curr_sl + min_dist * 0.5) and smart_trail <= (curr_price - min_dist):
                logger.info(f"📈 [Trailing Stop] Ratcheting BUY SL on #{pos_id}: {curr_sl:.5f} ➔ {smart_trail:.5f}")
                res = self.client.change_position_stop_loss(position_id=pos_id, level=smart_trail)
                if res and not res.get("error"):
                    pos["stop_loss"] = smart_trail
        else: # SELL
            # For SELL: SL only moves DOWN
            # Ensure new SL is below current SL, and at least min_dist above curr_price
            if smart_trail < (curr_sl - min_dist * 0.5) and smart_trail >= (curr_price + min_dist):
                logger.info(f"📉 [Trailing Stop] Ratcheting SELL SL on #{pos_id}: {curr_sl:.5f} ➔ {smart_trail:.5f}")
                res = self.client.change_position_stop_loss(position_id=pos_id, level=smart_trail)
                if res and not res.get("error"):
                    pos["stop_loss"] = smart_trail

    def run_loop(self, poll_interval: int = 15):
        """Continuous execution loop scanning assets for Smart Trail signals."""
        logger.info(f"🚀 Starting Smart Trail Forex Bot (Assets: {self.assets} | Interval: {poll_interval}s)")
        try:
            while True:
                self.update_account_state()
                
                for asset in self.assets:
                    try:
                        self.process_asset(asset)
                    except Exception as e:
                        logger.error(f"Error processing asset {asset}: {e}")
                        
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            logger.info("🛑 Bot stopped by user.")

def main():
    parser = argparse.ArgumentParser(description="Smart Trail Margin Forex Bot")
    parser.add_argument("--account", default="training", choices=["training", "regular"], help="Trading account type")
    parser.add_argument("--risk-pct", type=float, default=1.0, help="Account equity risk percentage per trade (default: 1.0)")
    parser.add_argument("--rr", type=float, default=2.0, help="Take Profit Risk-to-Reward ratio (default: 2.0 = 1:2)")
    parser.add_argument("--leverage", type=int, default=50, help="Leverage to use (default: 50)")
    parser.add_argument("--candle-size", type=int, default=60, help="Candle size in seconds (60=1m, 300=5m)")
    parser.add_argument("--assets", default="EURUSD,GBPUSD,USDJPY,EURGBP,AUDUSD", help="Comma-separated pairs")
    parser.add_argument("--max-positions", type=int, default=2, help="Max concurrent open positions")
    parser.add_argument("--interval", type=int, default=15, help="Poll interval in seconds")

    args = parser.parse_args()
    asset_list = [a.strip().upper() for a in args.assets.split(",") if a.strip()]

    bot = SmartTrailForexBot(
        account_type=args.account,
        risk_pct=args.risk_pct,
        risk_reward=args.rr,
        leverage=args.leverage,
        candle_size=args.candle_size,
        assets=asset_list,
        max_open_positions=args.max_positions
    )

    if bot.connect():
        bot.run_loop(poll_interval=args.interval)

if __name__ == "__main__":
    main()
