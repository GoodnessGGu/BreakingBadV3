# test_channel_parser.py
"""
Test script for channel signal parser.
Run this to verify the signal parsing works correctly.
"""

from channel_signal_parser import parse_channel_signal, is_signal_message

# Test signal message (from user's example)
test_message = """🔔 NEW SIGNAL!

🎫 Trade: 🇦🇺 AUD/JPY 🇯🇵 (OTC)
⏳ Timer: 5 minutes
➡️ Entry: 12:36 PM
📈 Direction: BUY 🟩

↪️ Martingale Levels:
 Level 1 → 12:41 PM
 Level 2 → 12:46 PM
 Level 3 → 12:51 PM"""

print("=" * 60)
print("Testing Channel Signal Parser")
print("=" * 60)

# Test 1: Check if message is detected as signal
print("\n1. Testing signal detection...")
is_signal = is_signal_message(test_message)
print(f"   Is signal message: {is_signal}")
assert is_signal, "Failed to detect signal message"
print("   ✅ PASSED")

# Test 2: Parse the signal
print("\n2. Testing signal parsing...")
signal = parse_channel_signal(test_message)
print(f"   Parsed signal: {signal}")

if signal:
    print(f"\n   📊 Pair: {signal['pair']}")
    print(f"   📈 Direction: {signal['direction']}")
    print(f"   ⏰ Entry Time: {signal['time'].strftime('%I:%M %p')}")
    print(f"   ⏳ Expiry: {signal['expiry']} minutes")
    
    # Verify expected values
    assert signal['pair'] == 'AUDJPY', f"Expected AUDJPY, got {signal['pair']}"
    assert signal['direction'] == 'CALL', f"Expected CALL, got {signal['direction']}"
    assert signal['expiry'] == 5, f"Expected 5, got {signal['expiry']}"
    
    print("\n   ✅ PASSED - All values correct!")
else:
    print("   ❌ FAILED - Could not parse signal")
    exit(1)

# Test 3: Test with SELL direction
print("\n3. Testing SELL direction...")
test_sell = """🔔 NEW SIGNAL!

🎫 Trade: EUR/USD (OTC)
⏳ Timer: 1 minutes
➡️ Entry: 02:30 PM
📈 Direction: SELL 🟥"""

signal_sell = parse_channel_signal(test_sell)
if signal_sell:
    print(f"   Direction: {signal_sell['direction']}")
    assert signal_sell['direction'] == 'PUT', f"Expected PUT, got {signal_sell['direction']}"
    assert signal_sell['pair'] == 'EURUSD', f"Expected EURUSD, got {signal_sell['pair']}"
    print("   ✅ PASSED")
else:
    print("   ❌ FAILED")
    exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✅")
print("=" * 60)
