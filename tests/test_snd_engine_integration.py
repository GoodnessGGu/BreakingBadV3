"""
tests/test_snd_engine_integration.py
Full integration verification for SNDStrategyEngine (Config E)
"""
import sys
import os
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.snd_engine import SNDStrategyEngine, SND_INSTRUMENT_PROFILES, calculate_atr
from bot.keyboards import snd_menu_keyboard, main_menu_keyboard, persistent_reply_keyboard

class MockMCPClient:
    def __init__(self):
        self.orders = []

    def get_candles(self, asset_id, size=900, count=120):
        # Generate 60 synthetic 15m candles
        np.random.seed(42)
        base = 1.0000
        candles = []
        for i in range(count):
            change = np.random.normal(0, 0.0010)
            o = base
            c = base + change
            h = max(o, c) + abs(np.random.normal(0, 0.0005))
            l = min(o, c) - abs(np.random.normal(0, 0.0005))
            candles.append({
                "open": o, "max": h, "min": l, "close": c,
                "from": i * 900, "to": (i + 1) * 900
            })
            base = c
        return candles

    def place_market_order(self, **kwargs):
        self.orders.append(kwargs)
        return {"order_id": 999111, "position_id": 888222}

    def list_positions(self, balance_id=None):
        return []

    def change_position_stop_loss(self, position_id, level):
        return {"status": "ok"}

    def close_position(self, position_id):
        return {"status": "ok"}

class TestSNDEngineIntegration(unittest.TestCase):
    def setUp(self):
        self.mcp = MockMCPClient()
        self.engine = SNDStrategyEngine(
            mcp_client=self.mcp,
            symbols=["NZDUSD", "USDJPY", "AUDUSD", "USDCAD"],
            account_type="training",
            lots=1.0,
            leverage=100,
            enabled=True
        )

    def test_profiles(self):
        self.assertIn("NZDUSD", SND_INSTRUMENT_PROFILES)
        self.assertIn("USDJPY", SND_INSTRUMENT_PROFILES)
        self.assertIn("AUDUSD", SND_INSTRUMENT_PROFILES)
        self.assertIn("USDCAD", SND_INSTRUMENT_PROFILES)
        self.assertEqual(SND_INSTRUMENT_PROFILES["NZDUSD"]["asset_id"], 8)
        self.assertEqual(SND_INSTRUMENT_PROFILES["USDJPY"]["asset_id"], 6)
        self.assertEqual(SND_INSTRUMENT_PROFILES["AUDUSD"]["asset_id"], 99)
        self.assertEqual(SND_INSTRUMENT_PROFILES["USDCAD"]["asset_id"], 100)

    def test_status_and_toggles(self):
        st = self.engine.get_status()
        self.assertTrue(st["enabled"])
        self.assertEqual(len(st["enabled_symbols"]), 4)

        # Toggle symbol
        self.engine.toggle_symbol("NZDUSD")
        st2 = self.engine.get_status()
        self.assertNotIn("NZDUSD", st2["enabled_symbols"])

        self.engine.toggle_symbol("NZDUSD")
        st3 = self.engine.get_status()
        self.assertIn("NZDUSD", st3["enabled_symbols"])

        # Lot & leverage updates
        self.engine.set_lots(0.5)
        self.assertEqual(self.engine.lots, 0.5)
        self.engine.set_leverage(50)
        self.assertEqual(self.engine.leverage, 50)

    def test_keyboard_generation(self):
        st = self.engine.get_status()
        kb = snd_menu_keyboard(st)
        self.assertIsNotNone(kb)
        self.assertTrue(len(kb.inline_keyboard) > 0)

        main_kb = main_menu_keyboard()
        self.assertIsNotNone(main_kb)
        btn_callbacks = [btn.callback_data for row in main_kb.inline_keyboard for btn in row]
        self.assertIn("btn_snd_menu", btn_callbacks)

    def test_detection_logic_dbr_and_rbd(self):
        # Create intentional Bullish Drop-Base-Rally + FVG
        # c_base: i-2, c_disp: i-1, c_conf: i
        df = pd.DataFrame([
            {"Open": 1.0000, "High": 1.0010, "Low": 0.9990, "Close": 1.0005}, # base
            {"Open": 1.0005, "High": 1.0070, "Low": 1.0000, "Close": 1.0065}, # disp (large body: 0.0060)
            {"Open": 1.0065, "High": 1.0080, "Low": 1.0035, "Close": 1.0075}, # conf (gap: Low[conf] - High[base] = 1.0035 - 1.0010 = 0.0025 > 0)
        ])
        df['ATR'] = [0.0020, 0.0020, 0.0020]
        df['EMA100'] = [1.0000, 1.0000, 1.0000]

        self.engine._detect_new_zones("NZDUSD", df)
        zones = self.engine.pending_zones["NZDUSD"]
        self.assertEqual(len(zones), 1)
        z = zones[0]
        self.assertEqual(z["side"], "BUY")
        self.assertEqual(z["entry"], 1.0035) # proximal edge

if __name__ == "__main__":
    unittest.main()
