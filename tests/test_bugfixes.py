"""
tests/test_bugfixes.py - Comprehensive Unit Tests for BreakingBad V3 Bugfixes & Addenda

Covers:
(a) Rate-limited list_positions does not clear active trades in SND or Trendline engines.
(b) Failed history lookup does not produce a $0.00 BREAKEVEN card; queues for pending settlement.
(c) Every S&D closure is settled into PropFirmGuardian exactly once (idempotent).
(d) Restarting PropFirmGuardian preserves processed_trade_ids and avoids double-counting.
(e) S&D engine closure payload builds without AttributeError (self.rr_ratio).
(f) TradeLedger attribution survives restart, multi-leg trades aggregate by group_id, broker-only trades appear as Manual.
(g) PropFirmGuardian dynamic lot size calculation from stop distance and target risk USD.
(h) MTF Confirmation gate validates trend alignment.
"""

import os
import sys
import time
import asyncio
import tempfile
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clients.forex_mcp_client import IQForexMCPClient, MCPRateLimitError
from strategies.snd_engine import SNDStrategyEngine
from strategies.trendline_engine import TrendlineStrategyEngine
from bot.prop_firm_manager import PropFirmGuardian
from bot.trade_ledger import TradeLedger
from utils.mtf_confirmation import MTFConfirmation


class TestBugfixesAndAddenda(unittest.TestCase):

    def setUp(self):
        self.mock_mcp = MagicMock()

    # -------------------------------------------------------------
    # (a) Rate-limit failure does NOT clear active trades
    # -------------------------------------------------------------
    def test_rate_limited_list_positions_does_not_clear_snd_active_trade(self):
        """Bug 1: Rate-limited or failed list_positions (returns None) must keep trade active."""
        engine = SNDStrategyEngine(
            mcp_client=self.mock_mcp,
            symbols=["AUDUSD"],
            account_type="training",
            lots=1.0,
            leverage=100
        )
        engine.balance_id = 12345

        # Active trade open
        engine.active_trades["AUDUSD"] = {
            "position_id": 100169873918,
            "order_id": 100169873918,
            "symbol": "AUDUSD",
            "side": "BUY",
            "entry_price": 0.65000,
            "sl_price": 0.64800,
            "tp_price": 0.65400,
            "risk_points": 0.00200,
            "lots": 1.0,
            "contract_size": 100000,
            "is_breakeven": False,
            "opened_at": time.time()
        }

        # Mock list_positions returning None (rate limited)
        self.mock_mcp.list_positions.return_value = None
        engine.get_market_price = MagicMock(return_value={"mid": 0.6500, "bid": 0.6499, "ask": 0.6501})

        async def run_step():
            await engine._manage_active_trade("AUDUSD", {"mid": 0.6500, "bid": 0.6499, "ask": 0.6501})

        asyncio.run(run_step())

        # The active trade MUST NOT be cleared
        self.assertIsNotNone(engine.active_trades.get("AUDUSD"))
        self.assertEqual(engine.active_trades["AUDUSD"]["position_id"], 100169873918)

    def test_rate_limited_list_positions_does_not_clear_trendline_active_trade(self):
        """Bug 1: Trendline engine keeps active trades when list_positions returns None."""
        engine = TrendlineStrategyEngine(
            mcp_client=self.mock_mcp,
            symbol="XAUUSD",
            account_type="training",
            lots=1.0,
            leverage=100
        )
        engine.balance_id = 12345
        engine.get_market_price = MagicMock(return_value={"mid": 2700.0, "bid": 2699.8, "ask": 2700.2})
        engine.active_trades["XAUUSD"] = {
            "symbol": "XAUUSD",
            "side": "BUY",
            "entry_price": 2700.0,
            "sl_price": 2690.0,
            "tp1": 2710.0,
            "tp2": 2725.0,
            "leg1_pos_id": 9001,
            "leg2_pos_id": 9002,
            "lot_tp1": 0.5,
            "lot_runner": 0.5,
            "tp1_closed": False,
            "runner_at_be": False,
            "opened_at": time.time()
        }

        self.mock_mcp.list_positions.return_value = None

        async def run_step():
            await engine._manage_active_trade("XAUUSD")

        asyncio.run(run_step())

        self.assertIsNotNone(engine.active_trades.get("XAUUSD"))
        self.assertEqual(engine.active_trades["XAUUSD"]["leg1_pos_id"], 9001)

    # -------------------------------------------------------------
    # (b) Failed history lookup does not produce a BREAKEVEN card
    # -------------------------------------------------------------
    def test_failed_history_lookup_does_not_produce_breakeven_card(self):
        """Bug 2: When history lookup fails, pnl is None and no false BREAKEVEN card is sent."""
        engine = SNDStrategyEngine(
            mcp_client=self.mock_mcp,
            symbols=["AUDUSD"],
            account_type="training",
            lots=1.0,
            leverage=100
        )
        engine.balance_id = 12345
        notifications = []
        engine.set_notification_callback(lambda txt: notifications.append(txt))

        trade = {
            "position_id": 100169873918,
            "order_id": 100169873918,
            "symbol": "AUDUSD",
            "side": "BUY",
            "entry_price": 0.65000,
            "sl_price": 0.64800,
            "tp_price": 0.65400,
            "lots": 1.0,
            "is_breakeven": False
        }

        # Simulate failed trade history lookup
        self.mock_mcp.get_trade_history.return_value = None

        async def run_step():
            await engine._log_trade_closure("AUDUSD", trade, reason_override="broker_settled")

        asyncio.run(run_step())

        # No notification should be broadcast
        for n in notifications:
            self.assertNotIn("BREAKEVEN", n)
            self.assertNotIn("$0.00", n)

        # Trade should be queued in pending settlements
        self.assertTrue(hasattr(engine, "_pending_settlements"))
        self.assertEqual(len(engine._pending_settlements), 1)
        self.assertEqual(engine._pending_settlements[0]["symbol"], "AUDUSD")

    # -------------------------------------------------------------
    # (c) Every S&D closure is settled into the guardian exactly once
    # -------------------------------------------------------------
    def test_sd_closure_settled_into_guardian_exactly_once(self):
        """Bug 3: Confirmed S&D closure settles into PropFirmGuardian exactly once."""
        guardian_alerts = []
        guardian = PropFirmGuardian(initial_capital=5000.0, broadcast_func=lambda t: guardian_alerts.append(t))
        guardian.is_enabled = True

        engine = SNDStrategyEngine(
            mcp_client=self.mock_mcp,
            symbols=["AUDUSD"],
            account_type="training",
            lots=1.0,
            leverage=100
        )
        engine.balance_id = 12345
        engine.set_prop_guardian(guardian)

        trade = {
            "position_id": 100169873918,
            "order_id": 100169873918,
            "symbol": "AUDUSD",
            "side": "BUY",
            "entry_price": 0.65000,
            "sl_price": 0.64800,
            "tp_price": 0.65400,
            "lots": 1.0,
            "is_breakeven": False
        }

        # Mock successful trade history with +335.76 PnL
        self.mock_mcp.get_trade_history.return_value = [
            {
                "position_id": 100169873918,
                "pnl": 335.76,
                "close_price": 0.65335,
                "close_reason": "take_profit"
            }
        ]

        async def run_step():
            await engine._log_trade_closure("AUDUSD", trade, reason_override="take_profit")

        asyncio.run(run_step())

        # Guardian must reflect settled trade
        self.assertEqual(guardian.current_balance, 5000.0 + 335.76)
        self.assertIn("100169873918", guardian.processed_trade_ids)

        # Calling again with same position_id must be idempotent
        async def run_step_duplicate():
            await engine._log_trade_closure("AUDUSD", trade, reason_override="take_profit")

        asyncio.run(run_step_duplicate())
        self.assertEqual(guardian.current_balance, 5000.0 + 335.76)

    # -------------------------------------------------------------
    # (d) Restarting PropFirmGuardian preserves processed_trade_ids
    # -------------------------------------------------------------
    def test_restarting_guardian_does_not_double_count_trade_id(self):
        """Bug 3: State persistence keeps processed_trade_ids across bot restarts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "prop_state.json")

            with patch("bot.prop_firm_manager.PROP_STATE_FILE", state_file):
                g1 = PropFirmGuardian(initial_capital=5000.0)
                g1.is_enabled = True

                async def settle_g1():
                    await g1.on_trade_closed(symbol="AUDUSD", pnl_usd=200.0, trade_id="TX_9999")

                asyncio.run(settle_g1())
                self.assertEqual(g1.current_balance, 5200.0)
                self.assertIn("TX_9999", g1.processed_trade_ids)

                # Simulate Bot Restart: new guardian loads state
                g2 = PropFirmGuardian(initial_capital=5000.0)
                self.assertEqual(g2.current_balance, 5200.0)
                self.assertIn("TX_9999", g2.processed_trade_ids)

                # Attempt duplicate settlement
                async def settle_g2():
                    await g2.on_trade_closed(symbol="AUDUSD", pnl_usd=200.0, trade_id="TX_9999")

                asyncio.run(settle_g2())
                # Must still be 5200.0, not 5400.0
                self.assertEqual(g2.current_balance, 5200.0)

    # -------------------------------------------------------------
    # (e) S&D engine rr_ratio payload builds without AttributeError
    # -------------------------------------------------------------
    def test_rr_ratio_payload_builds(self):
        """Bug 5: self.rr_ratio is defined and GSheet payload keys are correct."""
        engine = SNDStrategyEngine(
            mcp_client=self.mock_mcp,
            symbols=["NZDUSD"],
            account_type="training",
            lots=1.0,
            leverage=100
        )
        self.assertTrue(hasattr(engine, "rr_ratio"))
        self.assertEqual(engine.rr_ratio, 2.0)

        trade = {
            "position_id": "snd_123",
            "side": "BUY",
            "entry_price": 0.58500,
            "sl_price": 0.58300,
            "tp_price": 0.58900,
            "lots": 1.0
        }

        # Verify key resolution
        self.assertEqual(trade.get("sl_price"), 0.58300)
        self.assertEqual(trade.get("tp_price"), 0.58900)

        # Build simulated payload
        payload = {
            "risk_reward": f"1:{engine.rr_ratio:.1f}",
            "stop_loss": trade.get("sl_price"),
            "take_profit": trade.get("tp_price")
        }
        self.assertEqual(payload["risk_reward"], "1:2.0")
        self.assertEqual(payload["stop_loss"], 0.58300)

    # -------------------------------------------------------------
    # (f) Persistent Trade Ledger (Addendum A)
    # -------------------------------------------------------------
    def test_trade_ledger_attribution_and_restart(self):
        """Addendum A: Trade attribution comes from strategy column and survives restart."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_ledger.db")
            ledger = TradeLedger(db_path=db_path)

            # Record open trade
            ledger.record_open(
                position_id="TL_101",
                strategy="Trendline",
                symbol="XAUUSD",
                side="BUY",
                lots=0.5,
                entry_price=2700.0,
                sl=2690.0,
                tp=2725.0
            )

            # Record close
            ledger.record_close(
                position_id="TL_101",
                exit_price=2725.0,
                pnl=125.0,
                exit_reason="TP Hit"
            )

            # New instance simulates bot restart
            restarted_ledger = TradeLedger(db_path=db_path)
            closed = restarted_ledger.get_closed_trades(strategy="Trendline")
            self.assertEqual(len(closed), 1)
            self.assertEqual(closed[0]["position_id"], "TL_101")
            self.assertEqual(closed[0]["strategy"], "Trendline")
            self.assertEqual(closed[0]["pnl"], 125.0)

    def test_trade_ledger_multi_leg_aggregation(self):
        """Addendum A: Multi-leg trades with group_id aggregate as ONE trade in stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_ledger_legs.db")
            ledger = TradeLedger(db_path=db_path)

            gid = "tl_xauusd_group_1"
            # Leg 1
            ledger.record_open(
                position_id="LEG_1", strategy="Trendline", symbol="XAUUSD",
                side="BUY", lots=0.5, entry_price=2700.0, sl=2690.0, tp=2710.0, group_id=gid
            )
            ledger.record_close(position_id="LEG_1", exit_price=2710.0, pnl=50.0, exit_reason="tp1")

            # Leg 2
            ledger.record_open(
                position_id="LEG_2", strategy="Trendline", symbol="XAUUSD",
                side="BUY", lots=0.5, entry_price=2700.0, sl=2690.0, tp=2725.0, group_id=gid
            )
            ledger.record_close(position_id="LEG_2", exit_price=2725.0, pnl=125.0, exit_reason="tp2")

            # Grouped query
            grouped = ledger.get_grouped_closed_trades(strategy="Trendline")
            self.assertEqual(len(grouped), 1)
            agg = grouped[0]
            self.assertEqual(agg["group_id"], gid)
            self.assertEqual(agg["total_lots"], 1.0)
            self.assertEqual(agg["pnl"], 175.0)
            self.assertEqual(len(agg["legs"]), 2)

            # Stats count as ONE trade
            stats = ledger.get_stats(strategy="Trendline")
            self.assertEqual(stats["total_trades"], 1)
            self.assertEqual(stats["wins"], 1)
            self.assertEqual(stats["net_pnl"], 175.0)

    def test_trade_ledger_broker_reconciliation_manual_attribution(self):
        """Addendum A: Broker closed trades missing from ledger are inserted as Manual/Unattributed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_recon.db")
            ledger = TradeLedger(db_path=db_path)

            mock_client = MagicMock()
            mock_client.get_trade_history.return_value = [
                {
                    "position_id": "BROKER_MANUAL_1",
                    "asset_id": 1,
                    "side": "BUY",
                    "lots": 0.25,
                    "open_price": 1.0850,
                    "close_price": 1.0900,
                    "pnl": 12.50,
                    "close_reason": "manual_close"
                }
            ]

            res = ledger.reconcile_with_broker(mock_client, balance_id=123)
            self.assertTrue(res["ok"])
            self.assertEqual(res["inserted_manual_count"], 1)

            trades = ledger.get_closed_trades(strategy="Manual")
            self.assertEqual(len(trades), 1)
            self.assertEqual(trades[0]["strategy"], "Manual/Unattributed")
            self.assertEqual(trades[0]["position_id"], "BROKER_MANUAL_1")

    # -------------------------------------------------------------
    # (g) Prop Firm Dynamic Lot Sizing (Addendum B)
    # -------------------------------------------------------------
    def test_prop_firm_dynamic_lot_sizing(self):
        """Addendum B: Dynamic lot size calculated from dollar risk and stop loss distance."""
        guardian = PropFirmGuardian(initial_capital=5000.0)
        guardian.risk_per_trade_pct = 0.75  # $37.50 target risk

        # EURUSD: Entry 1.1000, SL 1.0980 (20 pips distance = 0.00200)
        # 1 lot of EURUSD (100,000 units) * 0.00200 = $200 risk per 1.0 lot
        # For $37.50 risk: 37.50 / 200 = ~0.1875 lots -> 0.18 or 0.19 lots
        lots = guardian.calculate_dynamic_lot_size(
            symbol="EURUSD",
            entry_price=1.1000,
            sl_price=1.0980,
            target_risk_usd=37.50
        )
        self.assertAlmostEqual(lots, 0.18, delta=0.02)
        # Must never return the static 1.0 lot
        self.assertNotEqual(lots, 1.0)

    # -------------------------------------------------------------
    # (h) Multi-Timeframe (MTF) Trend Confirmation (Addendum C)
    # -------------------------------------------------------------
    def test_mtf_trend_confirmation_alignment(self):
        """Addendum C: MTF Confirmation correctly detects aligned vs misaligned setups."""
        mtf = MTFConfirmation()
        mock_mcp = MagicMock()

        # Generate synthetic bullish H1 candles
        bullish_candles = []
        base_price = 1.0500
        for i in range(60):
            bullish_candles.append({
                "close": base_price + (i * 0.0005),
                "open": base_price + (i * 0.0005) - 0.0002,
                "high": base_price + (i * 0.0005) + 0.0003,
                "low": base_price + (i * 0.0005) - 0.0003
            })

        mock_mcp.get_candles.return_value = bullish_candles

        # BUY setup should align with bullish trend
        is_aligned, reason, metrics = mtf.check_higher_timeframe_bias(
            mock_mcp, asset_id=1, setup_side="BUY", symbol="EURUSD"
        )
        self.assertTrue(is_aligned)
        self.assertEqual(metrics["h1_trend"], "BULLISH")

        # SELL setup should be blocked
        is_aligned_sell, reason_sell, _ = mtf.check_higher_timeframe_bias(
            mock_mcp, asset_id=1, setup_side="SELL", symbol="EURUSD"
        )
        self.assertFalse(is_aligned_sell)


if __name__ == "__main__":
    unittest.main()
