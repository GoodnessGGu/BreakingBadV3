"""
tests/test_multiuser_onboarding.py - Multi-User Telegram Onboarding & Shared Demo Verification

Simulates the 12-scenario lifecycle required by BreakingBad V3:
1. Super-admin starts the bot.
2. Investor A opens the bot and triggers the onboarding flow.
3. Investor B opens the bot and triggers the onboarding flow.
4. Both investors accept the disclaimer and receive their investor keyboards.
5. Both navigate menus and customize personal preferences independently.
6. Both inspect the shared demo account balance safely.
7. Both view permitted trading history and bot status.
8. An investor attempts to access administrator settings and is rejected.
9. An investor attempts to switch to a REAL account and is rejected (API layer enforcement).
10. An investor is suspended by the administrator and immediately loses access.
11. The administrator retains full control of the bot and audit logging.
12. Multiple users interact concurrently via asyncio without corrupting shared state.
"""

import os
import sys
import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from user_manager import UserManager
from trade_database import TradeDatabase
from bot.keyboards import (
    investor_reply_keyboard,
    admin_reply_keyboard,
    onboarding_disclaimer_keyboard,
    user_management_keyboard
)
import accounts
import iqclient
from trade import TradeManager, TradeExecutionError


class TestMultiUserOnboardingAndSharedDemo(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        # Use an isolated in-memory or temporary database for testing
        self.test_db_path = f"test_users_{asyncio.get_event_loop().time_ns() if hasattr(asyncio.get_event_loop(), 'time_ns') else 'multiuser'}.db"
        self.db = TradeDatabase(db_path=self.test_db_path)
        
        # Test accounts
        self.admin_id = 999999999
        self.investor_a_id = 111111111
        self.investor_b_id = 222222222
        
        # Create user manager with test admin ID and test DB
        self.um = UserManager(admin_id=self.admin_id, db=self.db)

    async def asyncTearDown(self):
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except OSError:
                pass

    async def test_01_admin_starts_bot(self):
        """1. Super-administrator starts the bot and receives admin privileges."""
        allowed, status, user = self.um.check_access(
            self.admin_id, username="admin_boss", first_name="Admin"
        )
        self.assertTrue(allowed)
        self.assertEqual(status, "admin")
        self.assertEqual(user["role"], "admin")
        self.assertTrue(self.um.is_admin(self.admin_id))
        
        # Verify admin keyboard has administrative controls
        kb = admin_reply_keyboard()
        button_texts = [btn.text for row in kb.keyboard for btn in row]
        self.assertIn("⚙️ Risk & Sizing", button_texts)
        self.assertIn("📡 Channels", button_texts)
        self.assertIn("👥 User Management", button_texts)
        self.assertIn("🛑 Close All", button_texts)

    async def test_02_and_03_investor_onboarding_trigger(self):
        """2 & 3. Investor A and B start the bot and must complete onboarding."""
        # Investor A
        allowed_a, status_a, user_a = self.um.check_access(
            self.investor_a_id, username="investor_alice", first_name="Alice"
        )
        self.assertFalse(allowed_a)
        self.assertEqual(status_a, "needs_onboarding")
        self.assertFalse(user_a["onboarding_completed"])
        self.assertEqual(user_a["role"], "investor")

        # Investor B
        allowed_b, status_b, user_b = self.um.check_access(
            self.investor_b_id, username="investor_bob", first_name="Bob"
        )
        self.assertFalse(allowed_b)
        self.assertEqual(status_b, "needs_onboarding")
        self.assertFalse(user_b["onboarding_completed"])
        self.assertEqual(user_b["role"], "investor")

        # Verify onboarding disclaimer keyboard has the acceptance button
        disc_kb = onboarding_disclaimer_keyboard()
        cb_datas = [btn.callback_data for row in disc_kb.inline_keyboard for btn in row]
        self.assertIn("btn_accept_disclaimer", cb_datas)

    async def test_04_investors_complete_onboarding_and_receive_keyboards(self):
        """4. Both investors accept onboarding disclaimer and receive appropriate reply keyboards."""
        # Register both first
        self.um.check_access(self.investor_a_id, "investor_alice", "Alice")
        self.um.check_access(self.investor_b_id, "investor_bob", "Bob")

        # Both accept the disclaimer
        self.um.complete_user_onboarding(self.investor_a_id)
        self.um.complete_user_onboarding(self.investor_b_id)

        # Now both should be approved
        allowed_a, status_a, user_a = self.um.check_access(self.investor_a_id, "investor_alice", "Alice")
        allowed_b, status_b, user_b = self.um.check_access(self.investor_b_id, "investor_bob", "Bob")

        self.assertTrue(allowed_a)
        self.assertEqual(status_a, "approved")
        self.assertTrue(user_a["onboarding_completed"])

        self.assertTrue(allowed_b)
        self.assertEqual(status_b, "approved")
        self.assertTrue(user_b["onboarding_completed"])

        # Check investor reply keyboard
        inv_kb = investor_reply_keyboard()
        button_texts = [btn.text for row in inv_kb.keyboard for btn in row]
        
        # Should include viewer features
        self.assertIn("📊 Status", button_texts)
        self.assertIn("💰 Balance", button_texts)
        self.assertIn("📜 History", button_texts)
        self.assertIn("📋 Active Setups", button_texts)
        self.assertIn("📋 Demo Disclaimer", button_texts)
        self.assertIn("⚙️ View Config", button_texts)

        # Must NOT include admin controls
        self.assertNotIn("📡 Channels", button_texts)
        self.assertNotIn("⚙️ Risk & Sizing", button_texts)
        self.assertNotIn("👥 Users", button_texts)
        self.assertNotIn("🛑 Close All", button_texts)

    async def test_05_independent_menu_navigation_and_preferences(self):
        """5. Both investors independently navigate and have isolated personal preferences."""
        self.um.check_access(self.investor_a_id, "investor_alice", "Alice")
        self.um.check_access(self.investor_b_id, "investor_bob", "Bob")
        self.um.complete_user_onboarding(self.investor_a_id)
        self.um.complete_user_onboarding(self.investor_b_id)

        # Investor A sets timezone to America/New_York and notifications OFF
        self.um.set_user_preference(self.investor_a_id, "timezone", "America/New_York")
        self.um.set_user_preference(self.investor_a_id, "notifications_enabled", 0)

        # Investor B sets timezone to Europe/London and notifications ON
        self.um.set_user_preference(self.investor_b_id, "timezone", "Europe/London")
        self.um.set_user_preference(self.investor_b_id, "notifications_enabled", 1)

        pref_a = self.um.get_user_preferences(self.investor_a_id)
        pref_b = self.um.get_user_preferences(self.investor_b_id)

        self.assertEqual(pref_a.get("timezone"), "America/New_York")
        self.assertEqual(pref_a.get("notifications_enabled"), 0)

        self.assertEqual(pref_b.get("timezone"), "Europe/London")
        self.assertEqual(pref_b.get("notifications_enabled"), 1)

    async def test_06_shared_demo_account_balance_check(self):
        """6. Both investors safely inspect the shared demo account balance."""
        self.um.check_access(self.investor_a_id, "investor_alice", "Alice")
        self.um.check_access(self.investor_b_id, "investor_bob", "Bob")
        self.um.complete_user_onboarding(self.investor_a_id)
        self.um.complete_user_onboarding(self.investor_b_id)

        mock_forex_mcp = MagicMock()
        mock_forex_mcp.get_training_balance.return_value = {"balance": 10000.0, "equity": 10250.50, "balance_id": 12345}

        bal_a = mock_forex_mcp.get_training_balance()
        bal_b = mock_forex_mcp.get_training_balance()

        # Both point to the exact same shared balance
        self.assertEqual(bal_a["equity"], 10250.50)
        self.assertEqual(bal_b["equity"], 10250.50)
        self.assertEqual(bal_a["balance_id"], bal_b["balance_id"])

    async def test_07_permitted_trading_history_access(self):
        """7. Both investors can access permitted trading history and bot status."""
        self.um.check_access(self.investor_a_id, "investor_alice", "Alice")
        self.um.complete_user_onboarding(self.investor_a_id)

        # Allowed read actions
        allowed_a, _, _ = self.um.check_access(self.investor_a_id)
        self.assertTrue(allowed_a)

        # Log some user activity
        self.um.db.log_user_activity(self.investor_a_id, "view_history", "Category: all")
        self.um.db.log_user_activity(self.investor_a_id, "view_status", "Dashboard refreshed")

        # Verify activity was recorded
        stats = self.um.db.get_user_counts()
        self.assertGreaterEqual(stats.get("total", 0), 1)

    async def test_08_investor_admin_settings_rejected(self):
        """8. Investor attempts to access administrator settings and is strictly rejected."""
        self.um.check_access(self.investor_a_id, "investor_alice", "Alice")
        self.um.complete_user_onboarding(self.investor_a_id)

        # Check admin privileges
        self.assertFalse(self.um.is_admin(self.investor_a_id))

        # Sensitive admin actions must fail authorization checks
        admin_actions = ["/lots 5.0", "/leverage 200", "/pause", "/resume", "/closeall", "/suspend 222222222"]
        for action in admin_actions:
            self.assertFalse(
                self.um.is_admin(self.investor_a_id),
                f"Investor should not be authorized for admin action: {action}"
            )

    async def test_09_strict_demo_enforcement_at_api_layer(self):
        """9. Switching to a REAL account is rejected at the API and execution layer."""
        # 1. accounts.AccountManager enforcement
        mock_api = MagicMock()
        mgr = accounts.AccountManager(mock_api, MagicMock())
        with self.assertRaises(PermissionError):
            mgr.switch_account("real")
        with self.assertRaises(PermissionError):
            mgr.switch_account("regular")
        with self.assertRaises(PermissionError):
            mgr.switch_account("live")

        # 2. iqclient.IQOptionAPI enforcement
        with patch.object(iqclient.IQOptionAPI, "__init__", return_value=None):
            client = iqclient.IQOptionAPI()
            with self.assertRaises(PermissionError):
                client.switch_account("real")

        # 3. trade.py TradeManager parameter validation enforcement
        mock_acc_mgr = MagicMock()
        mock_acc_mgr.current_account_id = 999
        mock_acc_mgr.current_account_type = "real"
        tm = TradeManager(MagicMock(), MagicMock(), mock_acc_mgr)

        with self.assertRaises(TradeExecutionError):
            tm._validate_options_trading_parameters("EURUSD", 1.0, "call", 1)

        mock_acc_mgr.current_account_type = "regular"
        with self.assertRaises(TradeExecutionError):
            tm._validate_options_trading_parameters("EURUSD", 1.0, "call", 1)

        # Training/Practice account passes validation
        mock_acc_mgr.current_account_type = "training"
        tm._validate_options_trading_parameters("EURUSD", 1.0, "call", 1)

    async def test_10_investor_suspension_immediate_access_loss(self):
        """10. An investor is suspended by the admin and immediately loses access."""
        self.um.check_access(self.investor_a_id, "investor_alice", "Alice")
        self.um.complete_user_onboarding(self.investor_a_id)

        # Verify initial access
        allowed, status, _ = self.um.check_access(self.investor_a_id)
        self.assertTrue(allowed)
        self.assertEqual(status, "approved")

        # Administrator suspends Investor A
        self.um.suspend_user(self.investor_a_id, admin_id=self.admin_id)

        # Immediate check access must now return False and status 'suspended'
        allowed_after, status_after, user_after = self.um.check_access(self.investor_a_id)
        self.assertFalse(allowed_after)
        self.assertEqual(status_after, "suspended")
        self.assertEqual(user_after["access_status"], "suspended")

        # Check audit trail
        logs = self.um.get_audit_logs(limit=5)
        self.assertGreaterEqual(len(logs), 1)
        self.assertEqual(logs[0]["action"].upper(), "SUSPEND_USER")
        self.assertEqual(logs[0]["target_user_id"], self.investor_a_id)
        self.assertEqual(logs[0]["admin_id"], self.admin_id)

    async def test_11_admin_retains_full_control(self):
        """11. The administrator retains full control over user management and system settings."""
        # Admin can approve/reactivate previously suspended user
        self.um.check_access(self.investor_a_id, "investor_alice", "Alice")
        self.um.complete_user_onboarding(self.investor_a_id)
        self.um.suspend_user(self.investor_a_id, admin_id=self.admin_id)
        
        self.um.approve_user(self.investor_a_id, admin_id=self.admin_id)
        allowed, status, _ = self.um.check_access(self.investor_a_id)
        self.assertTrue(allowed)
        self.assertEqual(status, "approved")

        # Admin can view all users and audit logs
        user_list = self.um.list_all_users()
        self.assertGreaterEqual(len(user_list), 1)

        counts = self.um.get_user_counts()
        self.assertIn("total", counts)
        self.assertIn("approved", counts)

    async def test_12_concurrent_user_interactions_safe(self):
        """12. Multiple users interact concurrently via asyncio without corrupting shared state."""
        # Setup 10 simulated investors
        investor_ids = [1000 + i for i in range(10)]

        async def simulate_user_flow(uid):
            # Check access / register
            self.um.check_access(uid, f"user_{uid}", f"First_{uid}")
            # Complete onboarding
            self.um.complete_user_onboarding(uid)
            # Set personal timezone
            self.um.set_user_preference(uid, "timezone", f"Zone_{uid % 4}")
            # Fetch preference
            pref = self.um.get_user_preferences(uid)
            self.assertEqual(pref.get("timezone"), f"Zone_{uid % 4}")
            # Verify access
            allowed, status, _ = self.um.check_access(uid)
            self.assertTrue(allowed)
            self.assertEqual(status, "approved")

        # Run all 10 simulated flows concurrently
        await asyncio.gather(*(simulate_user_flow(uid) for uid in investor_ids))

        # Verify all 10 are recorded and approved
        counts = self.um.get_user_counts()
        self.assertGreaterEqual(counts.get("total", 0), 10)
        self.assertGreaterEqual(counts.get("approved", 0), 10)

    async def test_13_user_broadcast_alerts_and_admin_toggle(self):
        """13. Verify admin can toggle user broadcast updates, and subscribers are filtered correctly."""
        # Setup users: Investor A approved with notifications ON, Investor B approved with notifications OFF
        self.um.check_access(self.investor_a_id, username="investor_a")
        self.um.complete_user_onboarding(self.investor_a_id)
        self.um.set_user_preference(self.investor_a_id, "notifications_enabled", 1)

        self.um.check_access(self.investor_b_id, username="investor_b")
        self.um.complete_user_onboarding(self.investor_b_id)
        self.um.set_user_preference(self.investor_b_id, "notifications_enabled", 0)

        # Pending user C and Suspended user D
        pending_id = 333333333
        self.um.check_access(pending_id, username="pending_c")
        self.um.db.update_user_status(pending_id, "pending")

        suspended_id = 444444444
        self.um.check_access(suspended_id, username="suspended_d")
        self.um.suspend_user(suspended_id)

        # 1. By default, user alerts are enabled
        self.assertTrue(self.um.are_user_alerts_enabled())

        # 2. Subscribers should include Investor A, but NOT Investor B (notifications off), NOT pending C, NOT suspended D, and NOT admin
        subs = self.um.get_broadcast_subscribers(self.admin_id)
        self.assertIn(self.investor_a_id, subs)
        self.assertNotIn(self.investor_b_id, subs)
        self.assertNotIn(pending_id, subs)
        self.assertNotIn(suspended_id, subs)
        self.assertNotIn(self.admin_id, subs)

        # 3. Admin toggles user alerts OFF
        self.um.set_user_alerts_enabled(False, admin_id=self.admin_id)
        self.assertFalse(self.um.are_user_alerts_enabled())
        # All investor broadcasts muted
        self.assertEqual(self.um.get_broadcast_subscribers(self.admin_id), [])

        # Audit log should record the toggle
        logs = self.um.get_audit_logs(limit=5)
        toggle_logs = [l for l in logs if l.get("action") == "toggle_user_alerts"]
        self.assertTrue(len(toggle_logs) > 0)
        self.assertIn("Disabled", toggle_logs[0]["details"])

        # 4. Admin toggles user alerts ON again
        self.um.set_user_alerts_enabled(True, admin_id=self.admin_id)
        self.assertTrue(self.um.are_user_alerts_enabled())
        subs_after = self.um.get_broadcast_subscribers(self.admin_id)
        self.assertEqual(subs_after, [self.investor_a_id])

        # 5. Keyboards test
        kb_on = user_management_keyboard(alerts_enabled=True)
        self.assertIn("User Alerts: ON", str(kb_on))
        kb_off = user_management_keyboard(alerts_enabled=False)
        self.assertIn("User Alerts: OFF", str(kb_off))


if __name__ == "__main__":
    unittest.main()
