import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add parent dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mocking api global in telegram_bot
with patch('telegram_bot.api') as mock_api:
    from telegram_bot import ensure_connection

# Configure logging
logging.basicConfig(level=logging.INFO)

async def test_retry_logic():
    print("--- Starting Connection Retry Test ---")
    
    # Scenario 1: Fail twice, then succeed
    print("\nScenario 1: Fail 2x, then Success")
    mock_api._connected = False
    
    async def side_effect_connect_success():
        # On 3rd call, set connected to True
        if mock_api._connect.call_count >= 3:
             mock_api._connected = True
        else:
             raise Exception("Simulated Network Error")

    mock_api._connect = AsyncMock(side_effect=side_effect_connect_success)
    
    try:
        await ensure_connection()
        print("✅ Scenario 1 PASSED: Connected after retries.")
    except Exception as e:
        print(f"❌ Scenario 1 FAILED: {e}")

    # Scenario 2: Fail 3x (Max retries)
    print("\nScenario 2: Fail 3x (Max retries)")
    mock_api._connected = False
    mock_api._connect = AsyncMock(side_effect=Exception("Persistent Error"))
    
    try:
        await ensure_connection()
        print("❌ Scenario 2 FAILED: Should have raised ConnectionError.")
    except ConnectionError:
        print("✅ Scenario 2 PASSED: Raised ConnectionError as expected.")
    except Exception as e:
        print(f"❌ Scenario 2 FAILED: Raised wrong exception: {type(e)}")

if __name__ == "__main__":
    asyncio.run(test_retry_logic())
