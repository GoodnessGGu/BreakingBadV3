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

async def test_pnl_calculation():
    print("--- Starting PnL Calculation Test ---")
    
    # Mock API
    api = MagicMock()
    api.execute_digital_option_trade = AsyncMock(return_value=(True, 123))
    api.get_current_account_balance = MagicMock(return_value=1000)

    # Scenario: Loss on first attempt ($1), Win on second ($2)
    # Expected PnL: -$1 + (Profit from $2 trade)
    # Let's say profit is 80% -> $1.6
    # Total PnL = -1 + 1.6 = 0.6
    
    async def mock_outcome(order_id, expiry):
        # First call (Gale 0): Loss
        if api.get_trade_outcome.call_count == 1:
            return True, -1.0
        # Second call (Gale 1): Win
        else:
            return True, 1.6

    api.get_trade_outcome = AsyncMock(side_effect=mock_outcome)

    # Ensure suppression is OFF for this test or clear active trades
    settings.SUPPRESS_OVERLAPPING_SIGNALS = False
    ACTIVE_TRADES.clear()

    asset = "EURUSD"
    direction = "CALL"
    amount = 1
    
    print(f"Launching Trade for {asset} {direction} (Expect Loss then Win)...")
    result = await run_trade(api, asset, direction, 1, amount, max_gales=1)
    
    print(f"Trade Result: {result['result']}")
    print(f"Total Profit: {result['profit']}")
    
    expected_pnl = 0.6
    # Floating point comparison
    if abs(result['profit'] - expected_pnl) < 0.001:
        print("✅ TEST PASSED: PnL is correct.")
    else:
        print(f"❌ TEST FAILED. Expected {expected_pnl}, got {result['profit']}")

if __name__ == "__main__":
    asyncio.run(test_pnl_calculation())
