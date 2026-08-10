"""
realtime_bot2.py - Standalone AM_IQ Simple Strategy Tester (No Telegram Required)
Tests a pure, simple AM_IQ (Fast SMA 1 / Slow SMA 34 + WMA 4 Crossover) strategy.
Logs trades into a separate Google Sheet tab ('Realtime_Bot2_Trades').
Runs without modifying any existing codebase files.
"""

import asyncio
import logging
import os
import sys
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
logger = logging.getLogger("StandaloneBot2_AMIQ")

# Default OTC testing pairs for weekend trading
DEFAULT_ASSETS = ["EURUSD-OTC", "GBPUSD-OTC", "USDJPY-OTC", "EURGBP-OTC", "AUDUSD-OTC"]

def wma(series, period):
    """Calculates Weighted Moving Average."""
    return series.rolling(period).apply(
        lambda x: ((x * np.arange(1, period + 1)).sum()) / np.arange(1, period + 1).sum(),
        raw=True
    )

def analyze_amiq_simple_strategy(candles_data):
    """
    Pure AM_IQ Strategy:
    - Fast SMA (period 1)
    - Slow SMA (period 34)
    - Buffer 1 = Fast - Slow
    - Buffer 2 = WMA(Buffer 1, 4)
    Returns: ('CALL', features), ('PUT', features), or (None, None)
    """
    if not candles_data or len(candles_data) < 40:
        return None, None

    df = pd.DataFrame(candles_data)

    # 1. Calculate Moving Averages & Buffers
    df['sma_fast'] = df['close'].rolling(window=1).mean()
    df['sma_slow'] = df['close'].rolling(window=34).mean()
    df['buffer1'] = df['sma_fast'] - df['sma_slow']
    df['buffer2'] = wma(df['buffer1'], 4)

    # 2. Additional indicators for rich retraining logging
    df['rsi'] = df['close'].diff().apply(lambda x: max(x, 0)).rolling(14).mean() / (
        df['close'].diff().abs().rolling(14).mean() + 1e-9
    ) * 100

    amiq_call = (
        df['buffer1'].iloc[-1] > df['buffer2'].iloc[-1] and
        df['buffer1'].iloc[-2] < df['buffer2'].iloc[-2]
    )

    amiq_put = (
        df['buffer1'].iloc[-1] < df['buffer2'].iloc[-1] and
        df['buffer1'].iloc[-2] > df['buffer2'].iloc[-2]
    )

    signal = None
    if amiq_call:
        signal = "CALL"
    elif amiq_put:
        signal = "PUT"

    features = {
        'signal_source': 'realtime_bot2',
        'is_realtime_bot': True,
        'strategy': 'AM_IQ_Simple',
        'rsi': round(float(df['rsi'].iloc[-1]), 2) if 'rsi' in df and not pd.isna(df['rsi'].iloc[-1]) else 50.0,
        'buffer1': round(float(df['buffer1'].iloc[-1]), 6),
        'buffer2': round(float(df['buffer2'].iloc[-1]), 6),
        'close': float(df['close'].iloc[-1])
    }

    return signal, features

async def run_standalone_bot2(assets=None, base_amount=1.0, timeframe=60, scan_interval=5):
    if not assets:
        assets = DEFAULT_ASSETS

    logger.info("=" * 60)
    logger.info("  REALTIME_BOT2 - SIMPLE AM_IQ STRATEGY TRADER (SEPARATE SHEET)")
    logger.info("=" * 60)
    logger.info(f"Strategy: Pure AM_IQ Crossover (Fast SMA 1 / Slow SMA 34 / WMA 4)")
    logger.info(f"Assets Monitored: {assets}")
    logger.info(f"Target Sheet Tab: Realtime_Bot2_Trades")
    logger.info(f"Bet Sizing Mode: {config.bet_sizing_mode} | Base Amount: ${base_amount:.2f}")
    logger.info("=" * 60)

    # 1. Credentials & Connection
    email = os.getenv("IQ_EMAIL") or os.getenv("email")
    password = os.getenv("IQ_PASSWORD") or os.getenv("password")

    if not email or not password:
        logger.error("❌ Missing IQ_EMAIL or IQ_PASSWORD in .env file.")
        return

    api = IQOptionAPI(email=email, password=password)
    logger.info("Connecting to IQ Option...")
    await api._connect()

    # 2. Account Check & Safety Switch to Practice
    balance = api.get_current_account_balance()
    logger.info(f"✅ Connected! Current Account Balance: ${balance:.2f} ({api.account_mode.upper()})")

    if api.account_mode.lower() != "practice":
        logger.info("🛡️ Switching to PRACTICE account for test safety...")
        api.switch_account("practice")
        await asyncio.sleep(2)
        balance = api.get_current_account_balance()
        logger.info(f"✅ Account Switched! New Practice Balance: ${balance:.2f}")

    # 3. Continuous Scanning Loop
    cycle = 0
    trades_executed = 0

    try:
        while True:
            cycle += 1
            print(f"\n--- [Bot2 AM_IQ] Scan Cycle #{cycle} [{datetime.now().strftime('%H:%M:%S')}] ---", flush=True)

            for asset in assets:
                try:
                    # Fetch candle history (50 candles sufficient for 34 SMA)
                    candles = api.get_candle_history(asset, count=50, timeframe=timeframe)
                    if not candles:
                        logger.warning(f"Failed to fetch candles for {asset}")
                        continue

                    # Analyze Simple AM_IQ Strategy
                    signal, features = analyze_amiq_simple_strategy(candles)

                    if signal:
                        trades_executed += 1
                        logger.info(f"🚀 AM_IQ SIGNAL DETECTED: {signal} on {asset} ({timeframe}s)")

                        # Calculate dynamic bet amount
                        dynamic_amount = calculate_dynamic_trade_amount(api=api, base_amount=base_amount, gale_level=0)
                        logger.info(f"💰 Trade Stake ({config.bet_sizing_mode}): ${dynamic_amount:.2f}")

                        entry_target_time = datetime.now(timezone.utc)
                        expiry = 1

                        # Execute Trade (ignore_sl_tp=True for testing)
                        result = await run_trade(
                            api=api,
                            asset=asset,
                            direction=signal,
                            expiry=expiry,
                            amount=dynamic_amount,
                            auto_martingale=True,
                            features=features,
                            ignore_sl_tp=True,
                            target_time=entry_target_time
                        )

                        logger.info(f"🎯 Trade Completed for {asset}. Result: {result}")

                        # Ensure log row is sent directly to 'Realtime_Bot2_Trades' sheet tab
                        if isinstance(result, dict) and 'result' in result:
                            log_entry = {
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'asset': asset,
                                'direction': signal,
                                'amount': dynamic_amount,
                                'expiry': expiry,
                                'result': result.get('result', 'N/A'),
                                'profit': result.get('profit', 0.0),
                                'gale_level': result.get('gale_level', 0),
                                'signal_source': 'realtime_bot2',
                                'rsi': features.get('rsi', ''),
                                'adx': '',
                                'bb_width': '',
                                'atr': '',
                                'ema200_diff': '',
                                'orderbook_ratio': '',
                                'recent_win_rate': '',
                                'loss_prob': '',
                                'nn_prob': '',
                                'entry_latency': '',
                                'close': features.get('close', '')
                            }
                            gsheet_logger.log_trade(log_entry, worksheet_name="Realtime_Bot2_Trades")

                except Exception as e:
                    logger.error(f"Error scanning {asset}: {e}")

                # Short delay between pair scans
                await asyncio.sleep(1.0)

            # Sleep between scanning cycles
            await asyncio.sleep(scan_interval)

    except KeyboardInterrupt:
        logger.info("\n🛑 Realtime_Bot2 stopped by user (Ctrl+C).")
    finally:
        final_balance = api.get_current_account_balance()
        logger.info("=" * 60)
        logger.info("                BOT2 SESSION SUMMARY                    ")
        logger.info("=" * 60)
        logger.info(f"Final Balance: ${final_balance:.2f} (Net: ${final_balance - balance:+.2f})")
        logger.info(f"Total Trades Executed: {trades_executed}")
        logger.info("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realtime Bot 2 - Simple AM_IQ Strategy Tester")
    parser.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS, help="List of asset pairs to trade")
    parser.add_argument("--amount", type=float, default=1.0, help="Base trade amount in USD")
    parser.add_argument("--timeframe", type=int, default=60, help="Candle timeframe in seconds")
    parser.add_argument("--interval", type=int, default=5, help="Scan interval in seconds between cycles")

    args = parser.parse_args()

    try:
        asyncio.run(run_standalone_bot2(
            assets=args.assets,
            base_amount=args.amount,
            timeframe=args.timeframe,
            scan_interval=args.interval
        ))
    except KeyboardInterrupt:
        pass
