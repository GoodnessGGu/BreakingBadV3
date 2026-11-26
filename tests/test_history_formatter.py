"""
tests/test_history_formatter.py - Unit tests for compact mobile HTML history formatter
and fast non-blocking response optimizations.
"""

import os
import re
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bot.history_formatter import (
    DEFAULT_SYMBOL_MAP,
    build_symbol_map_from_assets,
    format_history_html,
    format_blitz_history_html,
    resolve_symbol_name,
)
from clients.blitz_mcp_client import IQBlitzMCPClient


SAMPLE_TRADES = [
    {"symbol": "74", "side": "BUY", "total_lots": 1.00, "pnl": -1.63, "exit_reason": "stop_loss", "strategy": "Manual/Unattributed", "close_time_utc": "2026-11-26 08:24:22"},
    {"symbol": "74", "side": "BUY", "total_lots": 1.00, "pnl": 3.20, "exit_reason": "take_profit", "strategy": "Manual/Unattributed", "close_time_utc": "2026-11-26 08:23:14"},
    {"symbol": "XAGUSD", "side": "SELL", "total_lots": 10.00, "pnl": 8.06, "exit_reason": "take_profit", "strategy": "ICT", "close_time_utc": "2026-11-26 08:07:05"},
    {"symbol": "1471", "side": "BUY", "total_lots": 0.50, "pnl": 24.67, "exit_reason": "take_profit", "strategy": "Manual/Unattributed", "close_time_utc": "2026-11-26 06:56:43"},
    {"symbol": "NAS100", "side": "BUY", "total_lots": 1.00, "pnl": 60.53, "exit_reason": "take_profit", "strategy": "Trendline", "close_time_utc": "2026-11-26 06:56:40"},
    {"symbol": "XAUUSD", "side": "SELL", "total_lots": 1.00, "pnl": 24.85, "exit_reason": "take_profit", "strategy": "ICT", "close_time_utc": "2026-11-26 05:49:32"},
    {"symbol": "XAUUSD", "side": "BUY", "total_lots": 1.00, "pnl": 10.25, "exit_reason": "take_profit", "strategy": "Trendline", "close_time_utc": "2026-11-26 02:09:24"},
    {"symbol": "XAUUSD", "side": "BUY", "total_lots": 1.00, "pnl": -9.35, "exit_reason": "stop_loss", "strategy": "Trendline", "close_time_utc": "2026-11-26 02:09:21"},
]


class TestHistoryFormatter(unittest.TestCase):

    def test_sample_data_row_widths_and_structure(self):
        """Every rendered <pre> row must be <= 38 chars and follow compact mobile HTML layout."""
        stats = {
            "total_trades": 8,
            "wins": 6,
            "losses": 2,
            "breakevens": 0,
            "win_rate": 75.0,
            "net_pnl": 120.58,
            "profit_factor": 11.98,
            "gross_profit": 131.56,
            "gross_loss": 10.98,
            "max_drawdown": 9.35,
        }
        blitz_summary = {"count": 20, "wins": 11, "losses": 9, "pnl": -3.63}
        broker_map = build_symbol_map_from_assets([
            {"asset_id": 74, "name": "Gold"},
            {"asset_id": 1471, "name": "US 100"},
        ])

        html_out = format_history_html(
            account_type="training",
            category="all",
            period="today",
            page=1,
            total_pages=1,
            stats=stats,
            trades=SAMPLE_TRADES,
            symbol_map=broker_map,
            blitz_summary=blitz_summary,
        )

        # Header checks (HTML bold, compact stats, Blitz directly beneath header)
        lines = html_out.splitlines()
        self.assertTrue(lines[0].startswith("📜 <b>TRAINING · ALL · TODAY</b> (1/1)"))
        self.assertIn("⚡ Blitz: -$3.63 (11W 9L)", lines[4])

        # Group headers keep full strategy labels with subtotal & count
        self.assertIn("<b>Manual/Unattributed</b> · +$26.24 · 3 trades", html_out)
        self.assertIn("<b>ICT</b> · +$32.91 · 2 trades", html_out)
        self.assertIn("<b>Trendline</b> · +$61.43 · 3 trades", html_out)

        # Stacked 02:09:24 + 02:09:21 Trendline XAUUSD BUY entries collapsed with ×2 marker
        self.assertIn("×2", html_out)

        # Assert no row inside <pre> blocks exceeds 38 characters
        pre_blocks = re.findall(r"<pre>(.*?)</pre>", html_out, flags=re.DOTALL)
        self.assertEqual(len(pre_blocks), 3)
        for block in pre_blocks:
            for row in block.splitlines():
                self.assertLessEqual(len(row), 38, f"Row exceeded 38 chars ({len(row)}): {row!r}")
                # No raw dates or verbose words or '$' or 'L' inside row columns
                self.assertNotIn("2026-", row)
                self.assertNotIn("take_profit", row)
                self.assertNotIn("stop_loss", row)
                self.assertNotIn("$", row)

    def test_symbol_resolution_and_html_escaping(self):
        """Numeric IDs resolve via cached map or fall back to ID<num>; dynamic text is HTML-escaped."""
        self.assertEqual(resolve_symbol_name("74", DEFAULT_SYMBOL_MAP), "XAUUSD")
        self.assertEqual(resolve_symbol_name("74", {}), "ID74")
        self.assertEqual(resolve_symbol_name("9999", DEFAULT_SYMBOL_MAP), "ID9999")

        xss_trade = [{
            "symbol": "LONG<SYM>&NAME",
            "side": "BUY",
            "total_lots": 1.0,
            "pnl": 5.0,
            "exit_reason": "manual",
            "strategy": "Strat<1>&2",
            "close_time_utc": "2026-11-26 10:15:00",
        }]
        out = format_history_html(
            account_type="training",
            category="all",
            period="today",
            page=1,
            total_pages=1,
            stats={"total_trades": 1, "wins": 1, "losses": 0, "breakevens": 0, "win_rate": 100.0,
                   "net_pnl": 5.0, "profit_factor": 999.0, "gross_profit": 5.0, "gross_loss": 0.0, "max_drawdown": 0.0},
            trades=xss_trade,
        )
        self.assertIn("Strat&lt;1&gt;&amp;2", out)
        self.assertNotIn("<1>", out)

    def test_blitz_client_caching_reduces_http_calls(self):
        """IQBlitzMCPClient caches balances & trade history across rapid consecutive reads."""
        client = IQBlitzMCPClient(token="dummy")
        client.call_tool = MagicMock(side_effect=lambda name, args: (
            {"balances": [{"type": "training", "amount": 5000.0}]}
            if name == "list_balances"
            else {"history": [{"position_id": 1, "profit": 1.8, "result": "win"}]}
        ))

        b1 = client.get_training_balance()
        b2 = client.get_training_balance()
        self.assertEqual(b1["amount"], 5000.0)
        self.assertEqual(b2["amount"], 5000.0)

        h1 = client.get_trade_history(limit=20)
        h2 = client.get_trade_history(limit=20)
        self.assertEqual(len(h1), 1)
        self.assertEqual(len(h2), 1)

        # Only 1 list_balances + 1 get_trade_history call despite 4 reads
        self.assertEqual(client.call_tool.call_count, 2)


if __name__ == "__main__":
    unittest.main()
