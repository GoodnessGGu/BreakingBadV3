import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
import pandas as pd

# Add current directory to path
sys.path.append(os.getcwd())

from iqclient import IQOptionAPI, run_trade
from strategies import analyze_strategy
from settings import config
from trade import calculate_dynamic_trade_amount
from trade_database import db
from dotenv import load_dotenv

# Load env
load_dotenv()

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LiveAITest")

# Configuration
TEST_ASSETS = ["EURUSD-OTC", "GBPUSD-OTC", "USDJPY-OTC", "EURGBP-OTC", "AUDUSD-OTC"]
TEST_AMOUNT = 1.0
EXPIRY = 1
MAX_TEST_CYCLES = 3  # Run 3 scanning cycles for test

async def test_asset(api, asset):
    logger.info(f"--- Testing {asset} ---")
    
    # 1. Fetch Data
    try:
        candles = api.get_candle_history(asset, count=210, timeframe=60)
        if not candles:
            logger.error(f"Failed to fetch candles for {asset}")
            return False
    except Exception as e:
        logger.error(f"Error fetching data for {asset}: {e}")
        return False

    # 2. Run AI Analysis (with Counterfactual Virtual Trade logging enabled)
    logger.info(f"Running AI Strategy Analysis for {asset}...")
    win_rate = db.get_recent_win_rate(limit=10)
    
    # Passing api=api triggers counterfactual logging if filters block a signal!
    signal, features = analyze_strategy(
        candles, 
        use_ai=True, 
        expiry=EXPIRY, 
        return_features=True, 
        recent_win_rate=win_rate,
        api=api
    )

    if not signal:
        logger.info(f"ℹ️ No active trade signal (or signal filtered) for {asset}.")
        return False

    logger.info(f"🚀 EXECUTABLE SIGNAL APPROVED: {signal} for {asset}!")
    
    # 3. Dynamic Bet Sizing & Latency Check
    dynamic_amount = calculate_dynamic_trade_amount(api=api, base_amount=TEST_AMOUNT, gale_level=0)
    logger.info(f"💰 Dynamic Sizing ({config.bet_sizing_mode}): Bet Amount = ${dynamic_amount:.2f}")

    entry_target_time = datetime.now(timezone.utc)
    
    # 4. Execute Practice Trade
    logger.info(f"Executing PRACTICE trade for {asset} to verify live execution...")
    result = await run_trade(
        api=api,
        asset=asset,
        direction=signal,
        expiry=EXPIRY,
        amount=dynamic_amount,
        auto_martingale=False, # Single test trade
        features=features,
        target_time=entry_target_time
    )
    
    logger.info(f"Trade Execution Finished. Result: {result}")
    return True

async def main():
    logger.info("="*60)
    logger.info("   BREAKING BAD V3 - LIVE AI & COUNTERFACTUAL LEARNING TEST   ")
    logger.info("="*60)

    email = os.getenv("IQ_EMAIL") or os.getenv("email")
    password = os.getenv("IQ_PASSWORD") or os.getenv("password")
    
    if not email or not password:
        logger.error("Missing credentials in .env (IQ_EMAIL / IQ_PASSWORD)")
        return

    api = IQOptionAPI(email=email, password=password)
    
    try:
        logger.info("Connecting to IQ Option API...")
        await api._connect()
        
        # Verify Practice Account
        balance = api.get_current_account_balance()
        logger.info(f"Connected! Initial Balance: ${balance:.2f} ({api.account_mode.upper()})")
        
        if api.account_mode.lower() != "practice":
            logger.info("Switching to PRACTICE account for safety...")
            api.switch_account("practice")
            await asyncio.sleep(2)
            balance = api.get_current_account_balance()
            logger.info(f"Switched! New Balance: ${balance:.2f}")

        logger.info(f"Scanning assets: {TEST_ASSETS}")
        logger.info(f"Bet Sizing Mode: {config.bet_sizing_mode} | Max Entry Delay: {config.max_entry_delay_seconds}s")
        logger.info(f"Counterfactual Learning Enabled: {config.enable_counterfactual_learning}")
        
        trades_attempted = 0
        cycle_count = 0
        
        while cycle_count < MAX_TEST_CYCLES:
            cycle_count += 1
            logger.info(f"\n{'='*20} SCAN CYCLE {cycle_count}/{MAX_TEST_CYCLES} (Total Executed: {trades_attempted}) {'='*20}")
            
            for asset in TEST_ASSETS:
                success = await test_asset(api, asset)
                if success:
                    trades_attempted += 1
                await asyncio.sleep(1.5)
            
            if cycle_count < MAX_TEST_CYCLES:
                logger.info(f"Cycle {cycle_count} complete. Waiting 10s before next scan cycle...")
                await asyncio.sleep(10)

        # Final Summary
        logger.info("\n" + "="*60)
        logger.info("                  LIVE TEST COMPLETED RESULTS                  ")
        logger.info("="*60)
        final_balance = api.get_current_account_balance()
        logger.info(f"💰 Account Balance: ${final_balance:.2f} (Net Change: ${final_balance - balance:+.2f})")
        logger.info(f"🎯 Total Trades Executed: {trades_attempted}")
        
        # Fetch Counterfactual Filter Efficiency Stats
        stats = db.get_filter_efficiency_stats(days=1)
        logger.info(f"🧪 Counterfactual Rejected Signals (Last 24h): {stats['total_rejected']}")
        logger.info(f"🛡️ Saved Losses (Correctly Blocked): {stats['saved_losses']}")
        logger.info(f"⚠️ Missed Wins: {stats['missed_wins']}")
        logger.info(f"🎯 Filter Accuracy: {stats['filter_accuracy']:.1f}%")
        logger.info("="*60)

    except KeyboardInterrupt:
        logger.info("Test stopped by user.")
    except Exception as e:
        logger.error(f"Fatal Error during test: {e}", exc_info=True)
    finally:
        logger.info("Test session closed.")

if __name__ == "__main__":
    asyncio.run(main())
