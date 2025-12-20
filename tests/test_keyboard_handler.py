
import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telegram_bot import handle_message, status, balance, pause_bot, resume_bot, settings_info, help_command

class TestKeyboardHandler(unittest.IsolatedAsyncioTestCase):
    async def test_router(self):
        """Verify that text messages route to the correct handlers"""
        print("\nTesting Keyboard Routing...")
        
        # Mapping of Text -> Expected Function
        test_cases = [
            ("📊 Status", status),
            ("💰 Balance", balance),
            ("⏸ Pause", pause_bot),
            ("▶ Resume", resume_bot),
            ("⚙️ Settings", settings_info),
            ("ℹ️ Help", help_command)
        ]

        for text, expected_func in test_cases:
            # Mock the update object
            update = MagicMock()
            update.message.text = text
            context = MagicMock()

            # Mock the expected function globally if possible, or just check execution flow.
            # Since we imported func directly, we can't easily mock it *inside* handle_message 
            # effectively without patching the module.
            # Instead, we will patch 'telegram_bot.status', 'telegram_bot.balance' etc.
            
            # Re-import inside test to apply patches
            with unittest.mock.patch(f'telegram_bot.{expected_func.__name__}', new_callable=AsyncMock) as mock_func:
                await handle_message(update, context)
                if mock_func.called:
                    print(f"[OK] {expected_func.__name__} called")
                else:
                    print(f"[FAIL] '{text}' did not call {expected_func.__name__}")
                    self.fail(f"Handler failed for {text}")

if __name__ == '__main__':
    unittest.main()
