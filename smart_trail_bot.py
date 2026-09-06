"""
smart_trail_bot.py - Smart Trail Trend-Reversal Bot (Higher Expiries & Safe Gale Capping)
Translates Pine Script 'Smart Trail Signals NO CONDITIONS' into real-time IQ Option execution.
Features:
- Dynamic ATR Trailing Stops (Length 14, Multiplier 2.0, Sensitivity 3)
- Higher Expiries (Default: 3 Minutes / 180s) to bypass 1-minute pullback traps
- Breakout Exhaustion Filter (skips overextended candles > 2.5x ATR)
- Capped Martingale (Max Gale 1: Worst-case loss -$3.22, never -$8.47)
- 15-Minute Asset Cooldown on Loss
- Dedicated Google Sheets Sync: 'Smart_Trail_Trades'
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
import argparse
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.getcwd())

from iqclient import IQOptionAPI, run_trade
from settings import config
from trade import calculate_dynamic_trade_amount
from trade_database import db
from gsheet_logger import gsheet_logger

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SmartTrailBot")

# Default OTC testing pairs
DEFAULT_ASSETS = ["EURUSD-OTC", "GBPUSD-OTC", "USDJPY-OTC", "EURGBP-OTC", "AUDUSD-OTC"]

def calculate_smart_trail(df: pd.DataFrame, length: int = 14, multiplier: float = 2.0, sensitivity: int = 3):
    """
    Direct Python implementation of TradingView's 'Smart Trail Signals NO CONDITIONS'.
    Returns DataFrame with:
    - smart_trend: +1 (Bullish) or -1 (Bearish)
    - smart_trail_value: Trailing support/resistance level
    - smart_bull: True on Bullish trend flip
    - smart_bear: True on Bearish trend flip
    - atr: Current True Range EMA
    """
    # Support both IQ Option's ('max', 'min') and standard ('high', 'low') columns
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
        # Smart Trail UP (Support)
        if close[i] > trail_up[i - 1] and close[i - 1] > trail_up[i - 1]:
            trail_up[i] = max(trail_up[i - 1], basic_up[i])
        else:
            trail_up[i] = basic_up[i]

        # Smart Trail DOWN (Resistance)
        if close[i] < trail_down[i - 1] and close[i - 1] < trail_down[i - 1]:
            trail_down[i] = min(trail_down[i - 1], basic_down[i])
        else:
            trail_down[i] = basic_down[i]

        # Determine smart trend direction
        if close[i] > trail_down[i - 1]:
            smart_trend[i] = 1
        elif close[i] < trail_up[i - 1]:
            smart_trend[i] = -1
        else:
            smart_trend[i] = smart_trend[i - 1]

    df['smart_trend'] = smart_trend
    df['smart_trail_value'] = np.where(smart_trend == 1, trail_up, trail_down)
    df['atr'] = atr

    # Trend change detection
    df['smart_bull'] = (smart_trend == 1) & (np.roll(smart_trend, 1) == -1)
    df['smart_bear'] = (smart_trend == -1) & (np.roll(smart_trend, 1) == 1)
    df.loc[0, ['smart_bull', 'smart_bear']] = False

    return df

def analyze_smart_trail_strategy(candles_data, length=14, multiplier=2.0, sensitivity=3, max_candle_atr_ratio=2.5):
    """
    Evaluates candle history for a Smart Trail trend reversal signal.
    Applies an exhaustion filter to skip overextended breakout candles.
    """
    if not candles_data or len(candles_data) < length + 5:
        return None, None

    df = pd.DataFrame(candles_data)
    for c in ['open', 'close', 'min', 'max', 'high', 'low']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c])

    df = calculate_smart_trail(df, length=length, multiplier=multiplier, sensitivity=sensitivity)

    curr = df.iloc[-1]
    prev_atr = df['atr'].iloc[-2] if len(df) > 1 else df['atr'].iloc[-1]
    candle_body = abs(curr['close'] - curr['open'])

    signal = None

    if curr['smart_bull']:
        # Check breakout exhaustion: If candle body is too huge, skip to avoid immediate pullback
        if prev_atr > 0 and (candle_body / prev_atr) > max_candle_atr_ratio:
            logger.warning(f"⚠️ [SmartTrail] Skipping Bullish Flip: Overextended breakout candle ({candle_body:.5f} > {max_candle_atr_ratio}x ATR).")
        else:
            signal = "CALL"

    elif curr['smart_bear']:
        if prev_atr > 0 and (candle_body / prev_atr) > max_candle_atr_ratio:
            logger.warning(f"⚠️ [SmartTrail] Skipping Bearish Flip: Overextended breakout candle ({candle_body:.5f} > {max_candle_atr_ratio}x ATR).")
        else:
            signal = "PUT"

    features = {
        'signal_source': 'smart_trail_bot',
        'is_realtime_bot': True,
        'strategy': 'Smart_Trail_NO_CONDITIONS',
        'smart_trend': int(curr['smart_trend']),
        'trail_value': round(float(curr['smart_trail_value']), 6),
        'atr': round(float(curr['atr']), 6),
        'close': float(curr['close'])
    }

    return signal, features

async def run_smart_trail_bot(assets=None, base_amount=1.0, timeframe=60, expiry=3, max_gales=1, scan_interval=5):
    if not assets:
        assets = DEFAULT_ASSETS

    logger.info("=" * 65)
    logger.info("  SMART TRAIL BOT - VOLATILITY TRAILING STOP TRADER (HIGHER EXPIRY)")
    logger.info("=" * 65)
    logger.info(f"Strategy: Smart Trail (ATR Length 14, Multiplier 2.0, Sens 3)")
    logger.info(f"Timeframe: {timeframe}s Candles | Trade Expiry: {expiry} Minutes")
    logger.info(f"Risk Rules: Capped at Gale {max_gales} Max | 15-Min Loss Cooldown")
    logger.info(f"Monitored Assets: {assets}")
    logger.info(f"Target Google Sheet: 'Smart_Trail_Trades'")
    logger.info("=" * 65)

    # 1. Credentials & Connection
    email = os.getenv("IQ_EMAIL") or os.getenv("email")
    password = os.getenv("IQ_PASSWORD") or os.getenv("password")

    if not email or not password:
        logger.error("❌ Missing IQ_EMAIL or IQ_PASSWORD in .env file.")
        return

    api = IQOptionAPI(email=email, password=password)
    logger.info("Connecting to IQ Option...")
    await api._connect()

    # 2. Account Safety Switch to Practice
    balance = api.get_current_account_balance()
    logger.info(f"✅ Connected! Current Balance: ${balance:.2f} ({api.account_mode.upper()})")

    if api.account_mode.lower() != "practice":
        logger.info("🛡️ Switching to PRACTICE account for test safety...")
        api.switch_account("practice")
        await asyncio.sleep(2)
        balance = api.get_current_account_balance()
        logger.info(f"✅ Switched! Practice Balance: ${balance:.2f}")

    # Track asset cooldowns (timestamp until which asset is paused)
    cooldowns = {}
    cycle = 0
    trades_executed = 0

    try:
        while True:
            cycle += 1
            print(f"\n--- [SmartTrail] Scan Cycle #{cycle} [{datetime.now().strftime('%H:%M:%S')}] ---", flush=True)

            for asset in assets:
                # Check cooldown
                now_ts = time.time()
                if now_ts < cooldowns.get(asset, 0):
                    remaining_cooldown = int(cooldowns[asset] - now_ts)
                    # logger.info(f"⏳ {asset} on loss cooldown ({remaining_cooldown}s remaining)")
                    continue

                try:
                    # Fetch candle history (60 candles is ample for ATR 14)
                    candles = api.get_candle_history(asset, count=60, timeframe=timeframe)
                    if not candles:
                        continue

                    # Analyze Smart Trail Strategy
                    signal, features = analyze_smart_trail_strategy(candles)

                    if signal:
                        trades_executed += 1
                        logger.info(f"🎯 SMART TRAIL FLIP DETECTED: {signal} on {asset} ({expiry}m Expiry)")

                        # Calculate dynamic bet amount
                        dynamic_amount = calculate_dynamic_trade_amount(api=api, base_amount=base_amount, gale_level=0)
                        logger.info(f"💰 Trade Stake: ${dynamic_amount:.2f} | Max Gales: {max_gales}")

                        entry_target_time = datetime.now(timezone.utc)

                        # Execute Trade with Gale Capping and Higher Expiry
                        result = await run_trade(
                            api=api,
                            asset=asset,
                            direction=signal,
                            expiry=expiry,
                            amount=dynamic_amount,
                            max_gales=max_gales,  # Cap at Gale 1 Max!
                            auto_martingale=True,
                            features=features,
                            ignore_sl_tp=True,
                            target_time=entry_target_time
                        )

                        logger.info(f"🏁 Trade Outcome for {asset}: {result}")

                        # Handle Cooldown on Loss
                        if isinstance(result, dict):
                            trade_result = result.get('result', 'N/A')
                            trade_profit = result.get('profit', 0.0)

                            if trade_result == "LOSS":
                                logger.warning(f"🛡️ Activating 15-Minute Cooldown on {asset} to prevent streak losses.")
                                cooldowns[asset] = time.time() + (15 * 60)

                            # Log to Google Sheets tab 'Smart_Trail_Trades'
                            log_entry = {
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'asset': asset,
                                'direction': signal,
                                'amount': dynamic_amount,
                                'expiry': expiry,
                                'result': trade_result,
                                'profit': trade_profit,
                                'gale_level': result.get('gale_level', 0),
                                'signal_source': 'smart_trail_bot',
                                'rsi': '',
                                'adx': '',
                                'bb_width': '',
                                'atr': features.get('atr', ''),
                                'ema200_diff': '',
                                'orderbook_ratio': '',
                                'recent_win_rate': '',
                                'loss_prob': '',
                                'nn_prob': '',
                                'entry_latency': '',
                                'close': features.get('close', '')
                            }
                            gsheet_logger.log_trade(log_entry, worksheet_name="Smart_Trail_Trades")

                except Exception as e:
                    logger.error(f"Error scanning {asset}: {e}")

                await asyncio.sleep(1.0)

            await asyncio.sleep(scan_interval)

    except KeyboardInterrupt:
        logger.info("\n🛑 Smart Trail Bot stopped by user (Ctrl+C).")
    finally:
        final_balance = api.get_current_account_balance()
        logger.info("=" * 65)
        logger.info("            SMART TRAIL SESSION SUMMARY                 ")
        logger.info("=" * 65)
        logger.info(f"Final Balance: ${final_balance:.2f} (Net: ${final_balance - balance:+.2f})")
        logger.info(f"Total Trades Executed: {trades_executed}")
        logger.info("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Trail Signals Bot with Higher Expiries")
    parser.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS, help="List of asset pairs to trade")
    parser.add_argument("--amount", type=float, default=1.0, help="Base trade amount in USD")
    parser.add_argument("--timeframe", type=int, default=60, help="Candle timeframe in seconds (default: 60)")
    parser.add_argument("--expiry", type=int, default=3, help="Option expiry duration in minutes (default: 3)")
    parser.add_argument("--max-gales", type=int, default=1, help="Maximum martingale gales (default: 1)")
    parser.add_argument("--interval", type=int, default=5, help="Scan interval in seconds between cycles")

    args = parser.parse_args()

    try:
        asyncio.run(run_smart_trail_bot(
            assets=args.assets,
            base_amount=args.amount,
            timeframe=args.timeframe,
            expiry=args.expiry,
            max_gales=args.max_gales,
            scan_interval=args.interval
        ))
    except KeyboardInterrupt:
        pass
