"""
bot/user_manager.py - Multi-User Access Control, Role Management, and Shared Demo Gatekeeper

Provides:
  - Role-Based Access Control (Admin vs. Investor/Standard User)
  - Interactive Onboarding & Demo Disclaimer Tracking
  - Shared Demo Execution Lock (Task coordination for concurrent users)
  - Rate Limiting / Debounce for Button & Command Interactions
  - Strict Demo-Only Execution Enforcement (Blocks any REAL mode access)
  - Administrative User Management (List, Inspect, Suspend, Reactivate)
"""

import os
import time
import logging
import asyncio
from typing import Dict, Any, Optional, Tuple, List
from trade_database import db as default_db

logger = logging.getLogger("UserManager")


class UserAccessDenied(Exception):
    """Raised when an unauthorized user attempts a restricted action."""
    pass


class DemoAccountEnforcementError(Exception):
    """Raised when any operation attempts to breach the shared DEMO-only boundary."""
    pass


class UserManager:
    """
    Central controller for multi-user identity, permissions, and shared account coordination.
    """

    def __init__(self, admin_id: Optional[int] = None, require_approval: bool = False, db: Optional[Any] = None):
        raw_admin = admin_id or os.getenv("ADMIN_ID", "0")
        try:
            self.admin_id = int(raw_admin)
        except (ValueError, TypeError):
            self.admin_id = 0

        self.require_approval = require_approval or os.getenv("REQUIRE_USER_APPROVAL", "false").lower() == "true"
        self.db = db or default_db
        
        # Async lock to coordinate trade operations on the shared single IQ Option connection
        self._trade_lock = asyncio.Lock()
        
        # Memory debounce / rate-limit tracking: {user_id: last_timestamp}
        self._last_interaction: Dict[int, float] = {}
        self._debounce_window = 0.4  # seconds

        # Ensure super-admin is always registered as admin in DB
        if self.admin_id > 0:
            self._ensure_admin_registered()

    def _ensure_admin_registered(self):
        """Ensures the environment ADMIN_ID has super-admin status in database."""
        try:
            user = self.db.get_user(self.admin_id)
            if not user:
                self.db.register_or_get_user(
                    telegram_id=self.admin_id,
                    username="Admin",
                    first_name="SuperAdmin",
                    default_role="admin",
                    default_status="approved"
                )
                self.db.complete_onboarding(self.admin_id)
            else:
                if user.get("role") != "admin" or user.get("status") != "approved":
                    self.db.update_user_role(self.admin_id, "admin")
                    self.db.update_user_status(self.admin_id, "approved")
                    self.db.complete_onboarding(self.admin_id)
        except Exception as e:
            logger.warning(f"Could not auto-register admin in DB: {e}")

    def is_admin(self, user_id: int) -> bool:
        """Returns True if the given user ID is the configured super-administrator."""
        return int(user_id) == self.admin_id

    def check_rate_limit(self, user_id: int) -> bool:
        """
        Rate limits rapid duplicate button clicks or command flooding.
        Returns True if action is allowed, False if throttled.
        """
        now = time.time()
        last = self._last_interaction.get(int(user_id), 0.0)
        if now - last < self._debounce_window:
            return False
        self._last_interaction[int(user_id)] = now
        return True

    def check_access(self, user_id: int, username: str = "", first_name: str = "", last_name: str = "") -> Tuple[bool, str, Dict[str, Any]]:
        """
        Checks a Telegram user's authorization status.
        
        Returns:
            Tuple of (is_authorized, status_code, user_record)
            status_code can be:
                - 'admin': Super-admin user
                - 'approved': Active authorized investor/user who completed onboarding
                - 'needs_onboarding': User must acknowledge demo disclaimer
                - 'pending': User registration is awaiting admin approval
                - 'suspended': User access has been revoked by admin
        """
        uid = int(user_id)
        if self.is_admin(uid):
            user = self.db.get_user(uid) or {
                "telegram_id": uid,
                "role": "admin",
                "status": "approved",
                "onboarded": 1
            }
            user["onboarding_completed"] = bool(user.get("onboarded", 1))
            user["access_status"] = user.get("status", "approved")
            user["created_at"] = user.get("registered_at", "")
            user["last_active"] = user.get("last_active_at", "")
            return True, "admin", user

        default_status = "pending" if self.require_approval else "approved"
        user, is_new = self.db.register_or_get_user(
            telegram_id=uid,
            username=username,
            first_name=first_name,
            last_name=last_name,
            default_role="investor",
            default_status=default_status
        )
        user["onboarding_completed"] = bool(user.get("onboarded", 0))
        user["access_status"] = user.get("status", "approved")
        user["created_at"] = user.get("registered_at", "")
        user["last_active"] = user.get("last_active_at", "")

        status = user.get("status", "approved")
        if status == "suspended":
            return False, "suspended", user

        if status == "pending":
            return False, "pending", user

        if not user.get("onboarded", 0):
            return False, "needs_onboarding", user

        return True, "approved", user

    def complete_user_onboarding(self, user_id: int) -> bool:
        """Marks that the user accepted the demo risk disclaimer."""
        uid = int(user_id)
        success = self.db.complete_onboarding(uid)
        if success:
            self.db.log_user_activity(uid, "onboarding_completed", "Accepted Shared Demo Disclaimer")
        return success

    def enforce_demo_only(self, requested_mode: str, user_id: Optional[int] = None):
        """
        Enforces strict DEMO-only mode at the execution and controller layer.
        Rejects ANY attempt to select, switch to, or execute on a REAL account.
        """
        clean = (requested_mode or "").strip().lower()
        if clean in ("real", "regular", "live"):
            msg = "⛔ SECURITY ENFORCEMENT: REAL account mode is strictly forbidden. BreakingBad V3 is operating in Shared Demonstration Mode."
            logger.warning(f"🚨 [Security] Attempt to access REAL account mode blocked! (User ID: {user_id or 'System'})")
            if user_id:
                self.db.log_admin_action(
                    admin_id=int(user_id),
                    action="BLOCKED_REAL_ACCOUNT_ATTEMPT",
                    details="Attempted switch to REAL mode"
                )
            raise DemoAccountEnforcementError(msg)

    @property
    def trade_lock(self) -> asyncio.Lock:
        """Returns the shared async lock for coordinating account operations."""
        return self._trade_lock

    # =========================================================================
    # Admin User Management Commands & Helper Methods
    # =========================================================================

    def suspend_user(self, target_user_id: int, admin_id: Optional[int] = None, reason: str = "") -> bool:
        """Suspends an investor account."""
        adm = int(admin_id or self.admin_id)
        if not self.is_admin(adm):
            raise UserAccessDenied("Only the super-administrator can suspend users.")
        
        target_uid = int(target_user_id)
        if target_uid == self.admin_id:
            raise ValueError("The administrator cannot suspend themselves.")

        success = self.db.update_user_status(target_uid, "suspended")
        if success:
            self.db.log_admin_action(
                admin_id=adm,
                action="SUSPEND_USER",
                target_user_id=target_uid,
                details=reason or "Suspended by admin"
            )
        return success

    def approve_user(self, target_user_id: int, admin_id: Optional[int] = None) -> bool:
        """Approves a pending registration or reactivates a user."""
        adm = int(admin_id or self.admin_id)
        if not self.is_admin(adm):
            raise UserAccessDenied("Only the super-administrator can approve users.")

        target_uid = int(target_user_id)
        success = self.db.update_user_status(target_uid, "approved")
        if success:
            self.db.log_admin_action(
                admin_id=adm,
                action="APPROVE_USER",
                target_user_id=target_uid,
                details="Approved by admin"
            )
        return success

    def reactivate_user(self, target_user_id: int, admin_id: Optional[int] = None) -> bool:
        """Reactivates a suspended investor account."""
        return self.approve_user(target_user_id, admin_id)

    def get_user(self, telegram_id: int) -> Optional[Dict[str, Any]]:
        u = self.db.get_user(int(telegram_id))
        if u:
            u["onboarding_completed"] = bool(u.get("onboarded", 0))
            u["access_status"] = u.get("status", "approved")
            u["created_at"] = u.get("registered_at", "")
            u["last_active"] = u.get("last_active_at", "")
        return u

    def list_all_users(self) -> List[Dict[str, Any]]:
        users = self.db.list_users()
        for u in users:
            u["onboarding_completed"] = bool(u.get("onboarded", 0))
            u["access_status"] = u.get("status", "approved")
            u["created_at"] = u.get("registered_at", "")
            u["last_active"] = u.get("last_active_at", "")
        return users

    def get_user_counts(self) -> Dict[str, int]:
        return self.db.get_user_counts()

    def get_user_preferences(self, telegram_id: int) -> Dict[str, Any]:
        return self.db.get_user_preferences(int(telegram_id))

    def set_user_preference(self, telegram_id: int, key: str, value: Any) -> bool:
        if key == "timezone":
            return self.db.set_user_preferences(int(telegram_id), timezone=str(value))
        elif key in ("notifications_enabled", "notifications"):
            return self.db.set_user_preferences(int(telegram_id), notifications_enabled=bool(value))
        return self.db.set_user_preferences(int(telegram_id), {key: value})

    def get_audit_logs(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self.db.get_recent_audit_logs(limit)

    def get_users_overview_text(self) -> str:
        """Builds a formatted summary of user statistics and recent registrations."""
        counts = self.db.get_user_counts()
        users = self.db.list_users()[:15]  # Most recent 15

        msg_lines = [
            "👥 *BreakingBad V3 User Management*",
            f"━━━━━━━━━━━━━━━━━━━━━━━━",
            f"• *Total Registered*: `{counts.get('total', 0)}`",
            f"• *Active Investors*: `{counts.get('approved', 0)}`",
            f"• *Suspended*: `{counts.get('suspended', 0)}`",
            f"• *Pending Approval*: `{counts.get('pending', 0)}`",
            f"━━━━━━━━━━━━━━━━━━━━━━━━",
            "*Recent Users:*"
        ]

        if not users:
            msg_lines.append("_No other users registered yet._")
        else:
            for u in users:
                status_icon = "🟢" if u["status"] == "approved" else ("🔴" if u["status"] == "suspended" else "🟡")
                role_icon = "👑" if u["role"] == "admin" else "👤"
                uname = f"@{u['username']}" if u.get("username") else (u.get("first_name") or "User")
                onboard_icon = "✅" if u.get("onboarded") else "⏳"
                msg_lines.append(
                    f"{status_icon} {role_icon} `{u['telegram_id']}` | {uname} | Onboard: {onboard_icon} | Status: `{u['status'].upper()}`"
                )

        msg_lines.append("\n*Admin Controls:*")
        msg_lines.append("• `/suspend <id>` - Revoke user access")
        msg_lines.append("• `/activate <id>` - Reactivate user access")
        msg_lines.append("• `/user <id>` - View registration details")
        return "\n".join(msg_lines)

    def format_user_detail(self, target_user_id: int) -> str:
        """Formats comprehensive information about a single user."""
        target_uid = int(target_user_id)
        user = self.db.get_user(target_uid)
        if not user:
            return f"❌ User `#{target_uid}` not found in the database."

        prefs = self.db.get_user_preferences(target_uid)
        status_icon = "🟢" if user["status"] == "approved" else ("🔴" if user["status"] == "suspended" else "🟡")
        
        return (
            f"👤 *User Profile: #{target_uid}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• *Name*: {user.get('first_name', '')} {user.get('last_name', '') or ''}\n"
            f"• *Username*: @{user.get('username') or 'None'}\n"
            f"• *Role*: `{user.get('role', 'investor').upper()}`\n"
            f"• *Status*: {status_icon} `{user.get('status', 'approved').upper()}`\n"
            f"• *Onboarded*: {'✅ Yes' if user.get('onboarded') else '⏳ Pending Disclaimer'}\n"
            f"• *Registered*: `{user.get('registered_at', 'N/A')[:19]}`\n"
            f"• *Last Active*: `{user.get('last_active_at', 'N/A')[:19]}`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚙️ *Preferences*:\n"
            f"• *Timezone*: `{prefs.get('timezone', 'UTC')}`\n"
            f"• *Notifications*: `{'Enabled' if prefs.get('notifications_enabled') else 'Disabled'}`"
        )


# Global user manager instance
user_manager = UserManager()
