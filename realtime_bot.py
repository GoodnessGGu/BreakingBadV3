"""
realtime_bot.py - Standalone Local Testing Bot (No Telegram Required)
Runs continuous technical & AI strategy scans on IQ Option pairs (OTC or Real).
Fully utilizes counterfactual off-policy AI learning, dynamic bet sizing, and latency safeguards.
Ignores session risk filter (hours 0 & 23) by default for testing at any time of day.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
import argparse
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.getcwd())

from iqclient import IQOptionAPI, run_trade
from strategies import analyze_strategy
from settings import config
from trade import calculate_dynamic_trade_amount
from trade_database import db

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StandaloneBot")

# Default OTC testing pairs for weekend trading
DEFAULT_ASSETS = ["EURUSD-OTC", "GBPUSD-OTC", "USDJPY-OTC", "EURGBP-OTC", "AUDUSD-OTC"]

async def run_standalone_bot(assets=None, base_amount=1.0, timeframe=60, scan_interval=5, ignore_session_filter=True, ignore_sl_tp=True):
    if not assets:
        assets = DEFAULT_ASSETS

    logger.info("=" * 60)
    logger.info("  BREAKING BAD V3 - STANDALONE LOCAL TEST TRADER (NO TELEGRAM)")
    logger.info("=" * 60)
    logger.info(f"Assets Monitored: {assets}")
    logger.info(f"Bet Sizing Mode: {config.bet_sizing_mode} | Base Amount: ${base_amount:.2f}")
    logger.info(f"Session Risk Filter: {'IGNORED (Test Mode Active)' if ignore_session_filter else 'ENABLED'}")
    logger.info(f"Daily Stop Loss / Take Profit: {'IGNORED (Test Mode Active)' if ignore_sl_tp else 'ENABLED'}")
    logger.info(f"Max Entry Delay: {config.max_entry_delay_seconds}s | Counterfactual Learning: {config.enable_counterfactual_learning}")
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
            print(f"\n--- Scan Cycle #{cycle} [{datetime.now().strftime('%H:%M:%S')}] ---", flush=True)

            for asset in assets:
                try:
                    # Fetch candle history (280 candles for indicators)
                    candles = api.get_candle_history(asset, count=280, timeframe=timeframe)
                    if not candles:
                        logger.warning(f"Failed to fetch candles for {asset}")
                        continue

                    # Fetch orderbook confirmation if available
                    active_id = api.market_manager.get_marginal_asset_id(asset)
                    orderbook = api.message_handler.orderbook.get(active_id) if active_id else None

                    # Fetch recent win rate for adaptive loss gates
                    win_rate = db.get_recent_win_rate(limit=10)

                    # Expiry determination: 1m for 60s tf, 5m for 300s tf
                    expiry = 5 if timeframe >= 300 else 1

                    # Analyze Strategy (ignore_session_filter skips hour 0 & 23 session blocking)
                    signal, features = analyze_strategy(
                        candles,
                        use_ai=True,
                        expiry=expiry,
                        return_features=True,
                        orderbook=orderbook,
                        recent_win_rate=win_rate,
                        api=api,
                        ignore_session_filter=ignore_session_filter  # <-- IGNORE SESSION FILTER FOR TESTING
                    )

                    if signal:
                        trades_executed += 1
                        logger.info(f"🚀 EXECUTABLE SIGNAL DETECTED: {signal} on {asset} ({expiry}m)")

                        if features is None:
                            features = {}
                        features['signal_source'] = 'realtime_bot'
                        features['is_realtime_bot'] = True

                        # Calculate dynamic bet amount
                        dynamic_amount = calculate_dynamic_trade_amount(api=api, base_amount=base_amount, gale_level=0)
                        logger.info(f"💰 Trade Stake ({config.bet_sizing_mode}): ${dynamic_amount:.2f}")

                        entry_target_time = datetime.now(timezone.utc)

                        # Execute Trade (ignore_sl_tp=True bypasses Daily Stop Loss / Take Profit limits for continuous testing)
                        result = await run_trade(
                            api=api,
                            asset=asset,
                            direction=signal,
                            expiry=expiry,
                            amount=dynamic_amount,
                            auto_martingale=True,
                            features=features,
                            ignore_sl_tp=ignore_sl_tp,  # <-- IGNORE DAILY STOP LOSS / TAKE PROFIT
                            target_time=entry_target_time  # <-- LATENCY CHECK
                        )

                        logger.info(f"🎯 Trade Completed for {asset}. Result: {result}")

                except Exception as e:
                    logger.error(f"Error scanning {asset}: {e}")

                # Short delay between pair scans
                await asyncio.sleep(1.0)

            # Sleep between scanning cycles
            await asyncio.sleep(scan_interval)

    except KeyboardInterrupt:
        logger.info("\n🛑 Standalone test trader stopped by user (Ctrl+C).")
    finally:
        final_balance = api.get_current_account_balance()
        logger.info("=" * 60)
        logger.info("                     SESSION SUMMARY                     ")
        logger.info("=" * 60)
        logger.info(f"Final Balance: ${final_balance:.2f} (Net: ${final_balance - balance:+.2f})")
        logger.info(f"Total Trades Executed: {trades_executed}")

        stats = db.get_filter_efficiency_stats(days=1)
        logger.info(f"🧪 Counterfactual Rejected Signals: {stats['total_rejected']}")
        logger.info(f"🛡️ Saved Losses (Blocked Bad Signals): {stats['saved_losses']}")
        logger.info(f"⚠️ Missed Wins: {stats['missed_wins']}")
        logger.info(f"🎯 Filter Accuracy: {stats['filter_accuracy']:.1f}%")
        logger.info("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Local Testing Trader for BreakingBadV3")
    parser.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS, help="List of asset pairs to trade (e.g. EURUSD-OTC GBPUSD-OTC)")
    parser.add_argument("--amount", type=float, default=1.0, help="Base trade amount in USD")
    parser.add_argument("--timeframe", type=int, default=60, help="Candle timeframe in seconds (default: 60)")
    parser.add_argument("--interval", type=int, default=5, help="Scan interval in seconds between cycles")
    parser.add_argument("--enforce-session-filter", action="store_true", help="Enable session risk filtering (hours 0 & 23)")
    parser.add_argument("--enforce-sl-tp", action="store_true", help="Enable Daily Stop Loss / Take Profit limits")

    args = parser.parse_args()

    try:
        asyncio.run(run_standalone_bot(
            assets=args.assets,
            base_amount=args.amount,
            timeframe=args.timeframe,
            scan_interval=args.interval,
            ignore_session_filter=not getattr(args, 'enforce_session_filter', False),
            ignore_sl_tp=not getattr(args, 'enforce_sl_tp', False)
        ))
    except KeyboardInterrupt:
        pass
