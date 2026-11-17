"""tests/test_persistent_breakeven.py - Unit tests for persistent Breakeven protection on multi-entry trades.
"""

import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
from strategies.ict_engine import ICTStrategyEngine
from strategies.trendline_engine import TrendlineStrategyEngine


class TestPersistentBreakeven(unittest.TestCase):

    def setUp(self):
        self.mock_mcp = MagicMock()

    def test_ict_persistent_breakeven_retry_on_distance_and_broker_rejection(self):
        """Verify ICT engine maintains pending_be across ticks until broker accepts."""
        engine = ICTStrategyEngine(
            mcp_client=self.mock_mcp,
            symbol="XAUUSD",
            account_type="training",
            lots=2.0
        )
        engine.balance_id = 12345
        engine.notify = AsyncMock()

        # Mock active dual ticket trade
        engine.active_trades["XAUUSD"] = {
            "order_id": 1001,
            "symbol": "XAUUSD",
            "side": "BUY",
            "entry_price": 2700.00,
            "initial_sl": 2690.00,
            "current_sl": 2690.00,
            "tp": 2720.00,
            "tp1": 2710.00,
            "tp2": 2720.00,
            "lots": 2.0,
            "moved_to_be": False,
            "tickets": [
                {
                    "tag": "TP1 Partial (50%)",
                    "order_id": 1001,
                    "position_id": 2001,
                    "lots": 1.0,
                    "side": "BUY",
                    "entry_price": 2700.00,
                    "sl": 2690.00,
                    "tp": 2710.00,
                    "is_tp1": True,
                    "is_open": True,
                    "moved_to_be": False
                },
                {
                    "tag": "Macro Runner (50%)",
                    "order_id": 1002,
                    "position_id": 2002,
                    "lots": 1.0,
                    "side": "BUY",
                    "entry_price": 2700.00,
                    "sl": 2690.00,
                    "tp": 2720.00,
                    "is_tp1": False,
                    "is_open": True,
                    "moved_to_be": False
                }
            ]
        }

        # Tick 1: TP1 (2001) has closed, Runner (2002) is still open
        # But market price is too close to entry (2700.30 vs entry 2700.00 -> dist 0.30 < 0.50 min)
        self.mock_mcp.list_positions.return_value = [{"position_id": 2002, "asset_id": 74}]
        
        asyncio.run(engine.manage_active_trade("XAUUSD", {"mid": 2700.30}))

        trade = engine.active_trades["XAUUSD"]
        tp1_tk = trade["tickets"][0]
        runner_tk = trade["tickets"][1]

        # TP1 should be marked closed
        self.assertFalse(tp1_tk["is_open"])
        # Runner must be armed with pending_be = True, but NOT moved_to_be yet due to distance
        self.assertTrue(runner_tk["pending_be"])
        self.assertFalse(runner_tk["moved_to_be"])
        self.mock_mcp.change_position_stop_loss.assert_not_called()

        # Tick 2: Market expands to 2708.00, but broker returns a validation_error
        self.mock_mcp.change_position_stop_loss.return_value = {"error": {"message": "validation_error"}}
        asyncio.run(engine.manage_active_trade("XAUUSD", {"mid": 2708.00}))

        # Must attempt broker call, but because broker rejected, pending_be remains True!
        self.mock_mcp.change_position_stop_loss.assert_called_once()
        self.assertTrue(runner_tk["pending_be"])
        self.assertFalse(runner_tk["moved_to_be"])

        # Tick 3: Broker call succeeds!
        self.mock_mcp.change_position_stop_loss.return_value = {"success": True}
        asyncio.run(engine.manage_active_trade("XAUUSD", {"mid": 2708.00}))

        # Runner is now locked in breakeven, pending_be cleared
        self.assertTrue(runner_tk["moved_to_be"])
        self.assertFalse(runner_tk["pending_be"])
        self.assertEqual(runner_tk["sl"], 2700.25)
        self.assertTrue(trade["moved_to_be"])


if __name__ == "__main__":
    unittest.main()
