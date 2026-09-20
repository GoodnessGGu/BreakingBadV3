import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import pytest
import re

SAMPLE_MESSAGES = [
    # 1. Real Signal - Buy
    ("""📤**GOLD BUY NOW**🐂

🚧**EntryZone** : 4370-4365

⛔**Cut Loss** : 4358

➡️**Take Profit **🤑 : 4375
➡️**Take Profit**🤑 : 4380

🤩**Tips for Traders**💯:
Use Proper Risk Management! 🔴""", "BUY", 4365.0, 4370.0, 4358.0, 4375.0, 4380.0),

    # 2. Real Signal - Buy 2
    ("""📤**GOLD BUY NOW**🐂

🚧**EntryZone** : 4384-4380

⛔**Cut Loss** : 4372

➡️**Take Profit **🤑 : 4390
➡️**Take Profit**🤑 : 4395

🤩**Tips for Traders**💯:
Use Proper Risk Management! 🔴""", "BUY", 4380.0, 4384.0, 4372.0, 4390.0, 4395.0),

    # 3. Real Signal - Sell
    ("""📤**GOLD SELL NOW**🐻

🚧**EntryZone** : 4410-4415

⛔**Cut Loss** : 4425

➡️**Take Profit **🤑 : 4400
➡️**Take Profit**🤑 : 4390""", "SELL", 4410.0, 4415.0, 4425.0, 4400.0, 4390.0),
]

FALSE_POSITIVES = [
    # Daily Performance Recap
    """DAILY PERFORMANCE 

🌳🌳🌳🌳🌳🌳🌳🌳🌳🌳🌳

🗓FRIDAY (18 SEPTEMBER 2026)

⭕️ XAUUSD Sell : 230 pips 🐳👑

⭕️ XAUUSD Sell : 90 pips 🐻

XAUUSD Buy : 150 pips 🚀🔥

⭕️ XAUUSD Buy : 150 pips 🚀

⭕️ XAUUSD Sell : 400 pips 🐳👑

🌳🌳🌳🌳🌳🌳🌳🌳🌳🌳🌳

100% win rate today
5 setups - 5 wins 0 loss
Total 1,020 pips won 💰""",

    # Market Analysis
    """XAUUSD (H1) Market Analysis 📊

Bearish Scenario 📉

Price has faced multiple rejections around 4394–4399 resistance. 📉

As long as price remains below this area, SELLERS may continue to have the advantage. 🐻

Bullish Scenario ↗️ 

If price manages to break and close above 4399 with strong momentum, the current bearish setup could weaken. 📈

Further upside may then target 4407–4416. 🐂""",

    # Teaser / Heads-up
    """LONDON SESSION IS COMING! 🇬🇧

The BULLISH momentum is still looking strong. ↗️

Let’s keep your eyes SHARP! 👀

We’re waiting for the PULLBACK BUY! 🛒""",

    # Incomplete Alert
    "Gold Buy Now",

    # Marketing
    """I'm giving FREE CAPITAL 📢

Attention traders!

Today might be your final chance to secure your spot and claim Golden Launch Giveaway 🎁"""
]

INSTRUCTION_MESSAGES = [
    ("""250 pips! ☑️

❤️Instructions:

🛡Protect Capital
Close all positions❌
💰Maximise Profit
Hold the positions with breakeven""", "BREAKEVEN"),

    ("Let’s officially CLOSE out this trading week here! ❌", "CLOSE_ALL"),
    ("Close Gold positions now! ❌", "CLOSE_ALL")
]

from copiers.gold_pips_copier import GoldSignalParser

def test_real_signals():
    for text, exp_side, exp_min, exp_max, exp_sl, exp_tp1, exp_tp2 in SAMPLE_MESSAGES:
        sig = GoldSignalParser.parse_signal(text)
        assert sig is not None, f"Failed to parse valid signal: {text}"
        assert sig["side"] == exp_side
        assert sig["entry_min"] == exp_min
        assert sig["entry_max"] == exp_max
        assert sig["sl"] == exp_sl
        assert sig["tp1"] == exp_tp1
        if exp_tp2:
            assert sig["tp2"] == exp_tp2

def test_false_positives_rejected():
    for text in FALSE_POSITIVES:
        sig = GoldSignalParser.parse_signal(text)
        assert sig is None, f"False positive incorrectly parsed as signal: {text}"

def test_instructions():
    for text, exp_type in INSTRUCTION_MESSAGES:
        inst = GoldSignalParser.parse_instruction(text)
        assert inst is not None, f"Failed to parse instruction: {text}"
        assert inst["type"] == exp_type

if __name__ == "__main__":
    print("Running parser unit tests...")
    test_real_signals()
    test_false_positives_rejected()
    test_instructions()
    print("✅ All parser unit tests passed!")
