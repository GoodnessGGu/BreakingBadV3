import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock
import sys
import os

# Add parent dir to sys.path to import iqclient
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from iqclient import run_trade, ACTIVE_TRADES
import settings

# Configure logging to see output
logging.basicConfig(level=logging.INFO)

async def test_suppression():
    print("--- Starting Suppression Test ---")
    
    # Mock API
    api = MagicMock()
    api.execute_digital_option_trade = AsyncMock(return_value=(True, 123))
    # Mock outcome to delay a bit so we can try to overlap
    async def delayed_outcome(*args, **kwargs):
        await asyncio.sleep(2) # Simulate trade duration
        return True, 1.0 # Win
    
    api.get_trade_outcome = AsyncMock(side_effect=delayed_outcome)
    api.get_current_account_balance = MagicMock(return_value=1000)

    # Ensure suppression is ON
    settings.SUPPRESS_OVERLAPPING_SIGNALS = True
    
    # Clear active trades just in case
    ACTIVE_TRADES.clear()

    # Define two conflicting trades
    asset = "EURUSD"
    direction = "CALL"
    
    print(f"Launching Trade 1 for {asset} {direction}...")
    task1 = asyncio.create_task(run_trade(api, asset, direction, 1, 1))
    
    # Wait a tiny bit to ensure task1 starts and grabs the lock
    await asyncio.sleep(0.1)
    
    print(f"Launching Trade 2 for {asset} {direction} (Should be suppressed)...")
    result2 = await run_trade(api, asset, direction, 1, 1)
    
    print(f"Trade 2 Result: {result2['result']}")
    
    # Wait for task1
    result1 = await task1
    print(f"Trade 1 Result: {result1['result']}")

    if result2['result'] == "SUPPRESSED" and result1['result'] == "WIN":
        print("✅ TEST PASSED: Signal was suppressed.")
    else:
        print("❌ TEST FAILED.")

if __name__ == "__main__":
    asyncio.run(test_suppression())
