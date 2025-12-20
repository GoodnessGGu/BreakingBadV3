
import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from settings import config
from iqclient import run_trade, ACTIVE_TRADES

class TestDynamicSettings(unittest.IsolatedAsyncioTestCase):
    async def test_config_updates(self):
        """Verify config object updates correctly"""
        print("\nTesting Config Object Updates...")
        original_amount = config.trade_amount
        config.trade_amount = 100
        self.assertEqual(config.trade_amount, 100)
        config.trade_amount = original_amount # Reset
        print("[OK] Config updates verified")

    async def test_pause_functionality(self):
        """Verify run_trade respects config.paused"""
        print("\nTesting Pause Functionality...")
        api_mock = MagicMock()
        
        # Test Not Paused
        config.paused = False
        # Mock execute_digital_option_trade to return success so it doesn't try to actually trade
        api_mock.execute_digital_option_trade = AsyncMock(return_value=(True, 12345))
        api_mock.get_trade_outcome = AsyncMock(return_value=(True, 10.0))
        api_mock.get_current_account_balance = MagicMock(return_value=1000)

        result = await run_trade(api_mock, "EURUSD", "call", 1, 1)
        self.assertNotEqual(result['result'], "SKIPPED")
        print("[OK] Bot traded when NOT paused")

        # Test Paused
        config.paused = True
        result = await run_trade(api_mock, "EURUSD", "call", 1, 1)
        self.assertEqual(result['result'], "SKIPPED")
        print("[OK] Bot skipped trade when PAUSED")
        
        config.paused = False # Reset

    async def test_martingale_config(self):
        """Verify run_trade respects config.max_martingale_gales"""
        print("\nTesting Martingale Config...")
        api_mock = MagicMock()
        
        # Mock failure/loss to trigger martingale
        # We need it to return success=True (trade placed) but result=LOSS multiple times
        api_mock.execute_digital_option_trade = AsyncMock(return_value=(True, 12345))
        # get_trade_outcome returns (success_bool, pnl)
        # We want pnl < 0 for loss
        api_mock.get_trade_outcome = AsyncMock(return_value=(True, -1.0))
        api_mock.get_current_account_balance = MagicMock(return_value=1000)

        config.max_martingale_gales = 1
        
        result = await run_trade(api_mock, "EURUSD", "call", 1, 1)
        
        # With max_gales=1, we expect:
        # Gale 0: Loss
        # Gale 1: Loss
        # Stop
        # Result gales should be 1
        self.assertEqual(result['gales'], 1)
        print("[OK] Max martingale gales respected (1 gale)")

        # Change config
        config.max_martingale_gales = 0
        result = await run_trade(api_mock, "EURUSD", "call", 1, 1)
        self.assertEqual(result['gales'], 0)
        print("[OK] Max martingale gales respected (0 gales)")

    async def test_suppression(self):
        """Verify signal suppression"""
        print("\nTesting Signal Suppression...")
        api_mock = MagicMock()
        api_mock.execute_digital_option_trade = AsyncMock(return_value=(True, 12345))
        api_mock.get_trade_outcome = AsyncMock(return_value=(True, 10.0))
        api_mock.get_current_account_balance = MagicMock(return_value=1000)

        config.suppress_overlapping_signals = True
        ACTIVE_TRADES.clear()

        # Start a fake trade that "hangs" to simulate active trade
        # We can't easily simulate concurrency in this unittest without running parallel tasks
        # But we can manually modify ACTIVE_TRADES to simulate state
        
        ACTIVE_TRADES.add(("EURUSD", "call"))
        
        result = await run_trade(api_mock, "EURUSD", "call", 1, 1)
        self.assertEqual(result['result'], "SUPPRESSED")
        print("[OK] Signal suppressed correctly")

        ACTIVE_TRADES.clear()
        result = await run_trade(api_mock, "EURUSD", "call", 1, 1)
        self.assertNotEqual(result['result'], "SUPPRESSED")
        print("[OK] Signal allowed when clear")

if __name__ == '__main__':
    unittest.main()
