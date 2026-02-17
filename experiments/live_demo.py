import sys
import os
import time
import logging
import asyncio
import pandas as pd
from datetime import datetime

# Add parent dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from iqclient import IQOptionAPI as IQClient, run_trade
from strategies import analyze_strategy
from settings import config

# Setup specialized logging for the demo
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("LiveDemo")

ASSETS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    "EURUSD-OTC", "GBPUSD-OTC", "USDJPY-OTC", "AUDUSD-OTC", "USDCAD-OTC"
] # 10 Assets (Real + OTC)
TRADE_AMOUNT = 1
POLL_INTERVAL = 10 # Seconds

async def dashboard(api, asset):
    """Prints a real-time status update for the asset."""
    try:
        # 1. Fetch live candles
        candles = api.get_candle_history(asset, count=210, timeframe=60)
        if not candles:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {asset:12} | DATA FETCH ERROR")
            return

        df = pd.DataFrame(candles)
        price = df['close'].iloc[-1]
        
        # 2. Fetch latest orderbook for orderflow analysis
        active_id = api.market_manager.get_marginal_asset_id(asset)
        orderbook = api.message_handler.orderbook.get(active_id)

        # 3. Run Strategy Analysis (with verbose logging)
        signal, features = analyze_strategy(candles, use_ai=True, expiry=1, return_features=True, orderbook=orderbook)
        
        rsi = features.get('rsi', 0) if features else 0
        nn_prob = features.get('nn_prob', 0) if features else 0 # We might need to check how to extract this if it's not in features
        
        # Format status line
        status = "SCANNING"
        if signal:
            status = f"SIGNAL: {signal}"
        
        line = (
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"{asset:12} | "
            f"Price: {price:.5f} | "
            f"RSI: {rsi:4.1f} | "
            f"NN Prob: {nn_prob:4.3f} | "
            f"Status: {status}"
        )
        print(line)

        # 3. If signal exists, run_trade will handle the execution (Martingale etc.)
        if signal:
            print(f"   >>> ATTEMPTING TRADE: {asset} {signal}...")
            await run_trade(
                api=api,
                asset=asset,
                direction=signal,
                expiry=1,
                amount=TRADE_AMOUNT,
                auto_martingale=False, # Single shot for demo
                features=features
            )

    except Exception as e:
        logger.error(f"Dashboard Error ({asset}): {e}")

async def main():
    print("=" * 60)
    print("       BREAKING BAD V3 - LIVE TRADING DASHBOARD       ")
    print("=" * 60)
    print("Mode: PRACTICE ACCOUNT (Safety First)")
    print(f"Assets: {', '.join(ASSETS)}")
    print("Looking for: AM_IQ + Sniper + AI Verification")
    print("=" * 60)

    # 1. Connect
    api = IQClient()
    try:
        print("Connecting...", end="", flush=True)
        await api._connect()
        print(" OK - CONNECTED")
    except Exception as e:
        print(f" FAILED: {e}")
        return

    # 2. Force Practice mode if not already
    current_balance = api.get_current_account_balance()
    print(f"Balance: {api.get_currency_symbol()}{current_balance:.2f} ({api.account_mode.upper()})")
    
    if api.account_mode.lower() != "practice":
        print("Switching to PRACTICE account for test...")
        api.switch_account("practice")
        await asyncio.sleep(2)
        print(f"Balance: {api.get_currency_symbol()}{api.get_current_account_balance():.2f}")

    # 3. Monitor Loop
    try:
        # Subscribe to orderflow for all assets
        print("Enabling orderflow streams...")
        for asset in ASSETS:
            api.subscribe_orderflow(asset)
            await asyncio.sleep(0.1) # Avoid flooding
            
        while True:
            tasks = [dashboard(api, asset) for asset in ASSETS]
            await asyncio.gather(*tasks)
            await asyncio.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\nDemo Stopped.")

if __name__ == "__main__":
    asyncio.run(main())
