"""
tests/test_audit_fixes.py - Verification tests for 2026-11-25/26 trade audit fixes.
"""

import os
import sys
import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clients.forex_mcp_client import IQForexMCPClient, MCPRateLimitError, resolve_dual_ticket_positions
from strategies.ict_engine import ICTStrategyEngine
from strategies.crt_engine import CRTStrategyEngine
from strategies.trendline_engine import TrendlineStrategyEngine
from strategies.snd_engine import SND_INSTRUMENT_PROFILES


class TestAuditFixes(unittest.IsolatedAsyncioTestCase):

    def test_01_resolve_dual_ticket_positions_by_take_profit_price(self):
        """Broker returns newest position (Leg 2) first and uses 'take_profit_price'."""
        open_positions = [
            {
                "position_id": 100177905160,
                "asset_id": 74,
                "open_time": "2026-11-25T23:15:34.539Z",
                "take_profit_price": 3531.45,  # Runner TP2
                "stop_lose_price": 3593.64,
            },
            {
                "position_id": 100177905153,
                "asset_id": 74,
                "open_time": "2026-11-25T23:15:32.100Z",
                "take_profit_price": 3552.18,  # TP1
                "stop_lose_price": 3593.64,
            },
        ]
        leg1, leg2 = resolve_dual_ticket_positions(
            open_positions, asset_id=74, tp1=3552.18, tp2=3531.45
        )
        self.assertEqual(leg1, "100177905153")
        self.assertEqual(leg2, "100177905160")

        # When only Leg 2 remains open after Leg 1 hits TP1:
        leg1_rem, leg2_rem = resolve_dual_ticket_positions(
            [open_positions[0]], asset_id=74, tp1=3552.18, tp2=3531.45
        )
        self.assertIsNone(leg1_rem)
        self.assertEqual(leg2_rem, "100177905160")

    def test_02_resolve_dual_ticket_chronological_fallback(self):
        """Even without take_profit_price, oldest open_time maps to Leg 1 and newest to Leg 2."""
        open_positions = [
            {"position_id": 2002, "asset_id": 816, "open_time": "2026-11-25T18:22:21.0Z"},
            {"position_id": 2001, "asset_id": 816, "open_time": "2026-11-25T18:22:20.0Z"},
        ]
        leg1, leg2 = resolve_dual_ticket_positions(
            open_positions, asset_id=816, tp1=59200.0, tp2=58600.0
        )
        self.assertEqual(leg1, "2001")
        self.assertEqual(leg2, "2002")

    def test_03_snd_default_lots_calibrated(self):
        """SND Forex default lots must be 0.1 (10k units), not 1.0 (100k units / $1k margin)."""
        for sym in ("NZDUSD", "USDJPY", "AUDUSD", "USDCAD"):
            self.assertEqual(SND_INSTRUMENT_PROFILES[sym]["default_lots"], 0.1)

    def test_04_naked_fallback_attaches_sl_tp_post_fill(self):
        """If place_market_order strips SL/TP on stop_levels error, it immediately re-attaches them post-fill."""
        client = IQForexMCPClient(token="dummy")
        call_log = []

        def fake_call(tool_name, args, endpoint=None):
            call_log.append((tool_name, dict(args), endpoint))
            if tool_name == "place_market_order":
                if "stop_loss" in args:
                    return {"error": {"message": "stop_levels error"}}
                return {"order_id": 9991, "position_id": 5551}
            if tool_name == "list_positions":
                return {"positions": [{"position_id": 5551, "asset_id": 74, "open_time": "2026-11-25T12:00:00Z"}]}
            if tool_name in ("change_position_stop_loss", "change_position_take_profit"):
                return {"status": "ok"}
            return {}

        client.call_tool = fake_call
        with patch("clients.forex_mcp_client.time.sleep"):
            res = client.place_market_order(
                side="buy", balance_id=123, instrument_id="mcfd.74", asset_id=74,
                lots=1.0, leverage=100, stop_loss=3503.78, take_profit=3540.0
            )

        self.assertTrue(res.get("post_fill_sl_attached"))
        sl_calls = [c for c in call_log if c[0] == "change_position_stop_loss"]
        tp_calls = [c for c in call_log if c[0] == "change_position_take_profit"]
        self.assertEqual(len(sl_calls), 1)
        self.assertEqual(sl_calls[0][1]["level"], 3503.78)
        self.assertEqual(len(tp_calls), 1)
        self.assertEqual(tp_calls[0][1]["level"], 3540.0)

    async def test_05_ict_arms_be_only_on_tp1_profit_and_fixes_naked_sl(self):
        """ICTEngine matches by take_profit_price, skips BE when Leg 1 hits SL, and arms BE when Leg 1 wins."""
        mock_mcp = MagicMock()
        engine = ICTStrategyEngine(mcp_client=mock_mcp, symbol="XAUUSD")
        engine.balance_id = 123
        engine.notify = AsyncMock()
        engine.notify_text = AsyncMock()
        engine.notify_photo = AsyncMock()

        # Case A: Leg 1 closes via stop_loss -> Runner must NOT arm Breakeven
        engine.active_trades["XAUUSD"] = {
            "symbol": "XAUUSD",
            "side": "SELL",
            "entry_price": 3540.0,
            "initial_sl": 3558.0,
            "current_sl": 3558.0,
            "tp": 3504.0,
            "tp1": 3522.0,
            "tp2": 3504.0,
            "tickets": [
                {"tag": "TP1 Partial (50%)", "order_id": 1, "position_id": 101, "lots": 1.0, "side": "SELL",
                 "entry_price": 3540.0, "sl": 3558.0, "initial_sl": 3558.0, "tp": 3522.0, "is_tp1": True, "is_open": True},
                {"tag": "Macro Runner (50%)", "order_id": 2, "position_id": 102, "lots": 1.0, "side": "SELL",
                 "entry_price": 3540.0, "sl": 3558.0, "initial_sl": 3558.0, "tp": 3504.0, "is_tp1": False, "is_open": True},
            ],
        }
        # Only Leg 2 still open (with NO broker SL to also test naked failsafe)
        mock_mcp.list_positions.return_value = [
            {"position_id": 102, "asset_id": 74, "take_profit_price": 3504.0, "stop_lose_price": 0.0}
        ]
        mock_mcp.get_trade_history.return_value = [
            {"position_id": 101, "pnl": -18.0, "close_price": 3558.0, "close_reason": "stop_loss"}
        ]
        mock_mcp.change_position_stop_loss.return_value = {"status": "ok"}

        with patch("strategies.ict_engine.gsheet_logger"):
            await engine.manage_active_trade("XAUUSD", {"mid": 3550.0, "bid": 3549.5, "ask": 3550.5})

            runner = engine.active_trades["XAUUSD"]["tickets"][1]
            self.assertFalse(runner.get("pending_be", False))
            # Naked SL failsafe should have attached initial SL 3558.0
            mock_mcp.change_position_stop_loss.assert_called_with(position_id=102, level=3558.0, endpoint="cfd")

            # Case B: Leg 1 closes via take_profit (+21.0) -> Runner MUST arm Breakeven (and then trail SL)
            mock_mcp.change_position_stop_loss.reset_mock()
            engine.active_trades["XAUUSD"] = {
                "symbol": "XAUUSD",
                "side": "SELL",
                "entry_price": 3573.0,
                "initial_sl": 3593.0,
                "current_sl": 3593.0,
                "tp": 3531.0,
                "tp1": 3552.0,
                "tp2": 3531.0,
                "tickets": [
                    {"tag": "TP1 Partial (50%)", "order_id": 11, "position_id": 201, "lots": 1.0, "side": "SELL",
                     "entry_price": 3573.0, "sl": 3593.0, "initial_sl": 3593.0, "tp": 3552.0, "is_tp1": True, "is_open": True},
                    {"tag": "Macro Runner (50%)", "order_id": 12, "position_id": 202, "lots": 1.0, "side": "SELL",
                     "entry_price": 3573.0, "sl": 3593.0, "initial_sl": 3593.0, "tp": 3531.0, "is_tp1": False, "is_open": True},
                ],
            }
            mock_mcp.list_positions.return_value = [
                {"position_id": 202, "asset_id": 74, "take_profit_price": 3531.0, "stop_lose_price": 3593.0}
            ]
            mock_mcp.get_trade_history.return_value = [
                {"position_id": 201, "pnl": 21.0, "close_price": 3552.0, "close_reason": "take_profit"}
            ]
            await engine.manage_active_trade("XAUUSD", {"mid": 3550.0, "bid": 3549.5, "ask": 3550.5})
            runner_b = engine.active_trades["XAUUSD"]["tickets"][1]
            self.assertTrue(runner_b.get("moved_to_be"))
            mock_mcp.change_position_stop_loss.assert_any_call(position_id=202, level=3572.75, endpoint="cfd")

    async def test_06_crt_gsheet_no_rr_target_attribute_error(self):
        """CRTEngine._log_ticket_closure must not raise AttributeError for missing rr_target."""
        mock_mcp = MagicMock()
        mock_mcp.get_trade_history.return_value = [
            {"position_id": 301, "pnl": 7.89, "close_price": 57793.0, "close_reason": "take_profit"}
        ]
        mock_mcp.list_balances.return_value = [{"balance_id": 123, "equity": 4200.0}]

        crt = CRTStrategyEngine(mcp_client=mock_mcp, symbols=["BTCUSD"])
        crt.balance_id = 123
        crt.active_trades["BTCUSD"] = {"side": "SELL"}

        with patch("strategies.crt_engine.gsheet_logger") as mock_gs:
            pnl, reason = await crt._log_ticket_closure(
                symbol="BTCUSD", tag="Leg 2 (Runner)", pos_id=301,
                lots=0.01, entry_px=58582.0, sl_px=59000.0, tp_px=57793.0
            )
            await asyncio.sleep(0.05)
            self.assertEqual(pnl, 7.89)
            self.assertEqual(reason, "take_profit")
            mock_gs.log_forex_margin_trade.assert_called_once()
            payload = mock_gs.log_forex_margin_trade.call_args[0][0]
            self.assertIn("1:2.2 Plan", payload["risk_reward"])

    async def test_07_trendline_handles_rate_limit_gracefully(self):
        """TrendlineEngine run_loop handles MCPRateLimitError without crashing."""
        mock_mcp = MagicMock()
        tl = TrendlineStrategyEngine(mcp_client=mock_mcp, symbol="XAUUSD")
        tl.is_enabled = True

        async def mock_eval(sym):
            tl.is_running = False
            raise MCPRateLimitError("Rate limited on get_candles (cfd)", retry_after_ms=3000)

        tl._evaluate_symbol_cycle = mock_eval
        with patch("strategies.trendline_engine.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await tl.run_loop()
            mock_sleep.assert_called_with(3.0)


if __name__ == "__main__":
    unittest.main()
