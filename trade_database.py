# trade_database.py
import sqlite3
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple, Any
import os
import threading

logger = logging.getLogger(__name__)

DATABASE_PATH = os.getenv("DATABASE_PATH", "trades.db")


class TradeDatabase:
    """Manages trade history storage, counterfactual analysis, and multi-user platform state using SQLite."""
    
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize database and create tables if they don't exist."""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                cursor = conn.cursor()
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        asset TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        amount REAL NOT NULL,
                        expiry INTEGER NOT NULL,
                        entry_time TEXT,
                        exit_time TEXT,
                        result TEXT,
                        profit REAL DEFAULT 0,
                        gale_level INTEGER DEFAULT 0,
                        signal_source TEXT DEFAULT 'manual',
                        error_message TEXT
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS virtual_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        asset TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        expiry INTEGER NOT NULL,
                        rejection_reason TEXT NOT NULL,
                        entry_price REAL,
                        exit_price REAL,
                        virtual_result TEXT DEFAULT 'PENDING_VIRTUAL',
                        features_json TEXT
                    )
                """)

                # Multi-User Management Tables
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        telegram_id INTEGER PRIMARY KEY,
                        username TEXT,
                        first_name TEXT,
                        last_name TEXT,
                        role TEXT DEFAULT 'investor',
                        status TEXT DEFAULT 'approved',
                        onboarded INTEGER DEFAULT 0,
                        registered_at TEXT NOT NULL,
                        last_active_at TEXT NOT NULL
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        telegram_id INTEGER PRIMARY KEY,
                        timezone TEXT DEFAULT 'UTC',
                        notifications_enabled INTEGER DEFAULT 1,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (telegram_id) REFERENCES users (telegram_id)
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS admin_audit_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        admin_id INTEGER NOT NULL,
                        action TEXT NOT NULL,
                        target_user_id INTEGER,
                        details TEXT,
                        timestamp TEXT NOT NULL
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_activity_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        telegram_id INTEGER NOT NULL,
                        action TEXT NOT NULL,
                        details TEXT,
                        timestamp TEXT NOT NULL
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_settings (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)

                conn.commit()
                conn.close()
            logger.info(f"✅ Trade database & multi-user tables initialized: {self.db_path}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
    
    def save_trade(self, trade_data: Dict) -> bool:
        """
        Save a trade to the database.
        
        Args:
            trade_data: Dictionary with trade information
                - asset, direction, amount, expiry, result, profit, gale_level, etc.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO trades (
                    timestamp, asset, direction, amount, expiry,
                    entry_time, exit_time, result, profit, gale_level, signal_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data.get('timestamp', datetime.now().isoformat()),
                trade_data.get('asset', 'UNKNOWN'),
                trade_data.get('direction', 'UNKNOWN'),
                trade_data.get('amount', 0),
                trade_data.get('expiry', 0),
                trade_data.get('entry_time'),
                trade_data.get('exit_time'),
                trade_data.get('result', 'PENDING'),
                trade_data.get('profit', 0),
                trade_data.get('gale_level', 0),
                trade_data.get('signal_source', 'manual')
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"✅ Trade saved to database: {trade_data.get('asset')} {trade_data.get('result')}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save trade: {e}")
            return False
    
    def get_trades(self, days: int = 7, asset: Optional[str] = None) -> List[Dict]:
        """
        Retrieve trades from the database.
        
        Args:
            days: Number of days to look back
            asset: Filter by specific asset (optional)
        
        Returns:
            List of trade dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            if asset:
                cursor.execute("""
                    SELECT * FROM trades 
                    WHERE timestamp >= ? AND asset = ?
                    ORDER BY timestamp DESC
                """, (cutoff_date, asset))
            else:
                cursor.execute("""
                    SELECT * FROM trades 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                """, (cutoff_date,))
            
            rows = cursor.fetchall()
            trades = [dict(row) for row in rows]
            
            conn.close()
            return trades
        except Exception as e:
            logger.error(f"❌ Failed to retrieve trades: {e}")
            return []
    
    def get_statistics(self, days: int = 7) -> Dict:
        """
        Calculate trading statistics.
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Dictionary with statistics
        """
        try:
            trades = self.get_trades(days=days)
            
            if not trades:
                return {
                    'total_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_rate': 0,
                    'total_profit': 0,
                    'avg_profit': 0
                }
            
            wins = sum(1 for t in trades if t['result'] == 'WIN')
            losses = sum(1 for t in trades if t['result'] == 'LOSS')
            total_profit = sum(t['profit'] for t in trades if t['profit'])
            
            # Gale Stats
            straight_wins = sum(1 for t in trades if t['result'] == 'WIN' and t['gale_level'] == 0)
            gale_wins = wins - straight_wins
            
            return {
                'total_trades': len(trades),
                'wins': wins,
                'losses': losses,
                'win_rate': (wins / len(trades) * 100) if trades else 0,
                'total_profit': total_profit,
                'avg_profit': total_profit / len(trades) if trades else 0,
                'straight_wins': straight_wins,
                'gale_wins': gale_wins
            }
        except Exception as e:
            logger.error(f"❌ Failed to calculate statistics: {e}")
            return {}
    
    def get_daily_summary(self, date: Optional[datetime] = None) -> Dict:
        """Get summary for a specific day."""
        if date is None:
            date = datetime.now()
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
            end_of_day = date.replace(hour=23, minute=59, second=59, microsecond=999999).isoformat()
            
            cursor.execute("""
                SELECT * FROM trades 
                WHERE timestamp >= ? AND timestamp <= ?
            """, (start_of_day, end_of_day))
            
            rows = cursor.fetchall()
            trades = [dict(row) for row in rows]
            conn.close()
            
            if not trades:
                return {'date': date.strftime('%Y-%m-%d'), 'total_trades': 0}
            
            wins = sum(1 for t in trades if t['result'] == 'WIN')
            losses = sum(1 for t in trades if t['result'] == 'LOSS')
            total_profit = sum(t['profit'] for t in trades if t['profit'])
            
            return {
                'date': date.strftime('%Y-%m-%d'),
                'total_trades': len(trades),
                'wins': wins,
                'losses': losses,
                'win_rate': (wins / len(trades) * 100) if trades else 0,
                'total_profit': total_profit
            }
        except Exception as e:
            logger.error(f"❌ Failed to get daily summary: {e}")
            return {}

    def get_recent_win_rate(self, limit: int = 10) -> float:
        """Get the win rate of the last N trades (0.0 to 1.0)."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT result FROM trades 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return 1.0 # Default to perfect if no trades yet
            
            wins = sum(1 for row in rows if row['result'] == 'WIN')
            return wins / len(rows)
        except Exception as e:
            logger.error(f"❌ Error fetching recent win rate: {e}")
            return 1.0

    def get_best_pairs(self, days: int = 30, limit: int = 5) -> List[Dict]:
        """Get best performing currency pairs."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute("""
                SELECT 
                    asset,
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(profit) as total_profit
                FROM trades
                WHERE timestamp >= ?
                GROUP BY asset
                ORDER BY total_profit DESC
                LIMIT ?
            """, (cutoff_date, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            results = []
            for row in rows:
                asset, total, wins, profit = row
                results.append({
                    'asset': asset,
                    'total_trades': total,
                    'wins': wins,
                    'win_rate': (wins / total * 100) if total > 0 else 0,
                    'total_profit': profit or 0
                })
            
            return results
        except Exception as e:
            logger.error(f"❌ Failed to get best pairs: {e}")
            return []

    def save_virtual_trade(self, virtual_data: Dict) -> Optional[int]:
        """Saves a skipped/rejected trade signal for counterfactual learning."""
        try:
            import json
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            features_str = json.dumps(virtual_data.get('features', {})) if isinstance(virtual_data.get('features'), dict) else str(virtual_data.get('features', '{}'))
            
            cursor.execute("""
                INSERT INTO virtual_trades (
                    timestamp, asset, direction, expiry, rejection_reason,
                    entry_price, exit_price, virtual_result, features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                virtual_data.get('timestamp', datetime.now().isoformat()),
                virtual_data.get('asset', 'UNKNOWN'),
                virtual_data.get('direction', 'UNKNOWN'),
                virtual_data.get('expiry', 1),
                virtual_data.get('rejection_reason', 'UNKNOWN'),
                virtual_data.get('entry_price', 0.0),
                virtual_data.get('exit_price', 0.0),
                virtual_data.get('virtual_result', 'PENDING_VIRTUAL'),
                features_str
            ))
            
            trade_id = cursor.lastrowid
            conn.commit()
            conn.close()
            logger.info(f"🧪 [Counterfactual] Virtual trade logged (ID: {trade_id}) | Reason: {virtual_data.get('rejection_reason')}")
            return trade_id
        except Exception as e:
            logger.error(f"❌ Failed to save virtual trade: {e}")
            return None

    def update_virtual_trade_outcome(self, trade_id: int, exit_price: float, virtual_result: str) -> bool:
        """Updates the outcome of a skipped/rejected trade after expiry."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE virtual_trades
                SET exit_price = ?, virtual_result = ?
                WHERE id = ?
            """, (exit_price, virtual_result, trade_id))
            
            conn.commit()
            conn.close()
            logger.info(f"🎯 [Counterfactual] Virtual trade #{trade_id} outcome updated -> {virtual_result}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to update virtual trade outcome: {e}")
            return False

    def get_filter_efficiency_stats(self, days: int = 7) -> Dict:
        """Calculates filter precision and blocked trade accuracy."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            cursor.execute("""
                SELECT * FROM virtual_trades
                WHERE timestamp >= ? AND virtual_result != 'PENDING_VIRTUAL'
            """, (cutoff_date,))
            
            rows = [dict(r) for r in cursor.fetchall()]
            conn.close()
            
            total_rejected = len(rows)
            saved_losses = sum(1 for r in rows if r['virtual_result'] == 'VIRTUAL_LOSS')
            missed_wins = sum(1 for r in rows if r['virtual_result'] == 'VIRTUAL_WIN')
            filter_accuracy = (saved_losses / total_rejected * 100) if total_rejected > 0 else 0.0
            
            reasons_summary = {}
            for r in rows:
                reason = r.get('rejection_reason', 'UNKNOWN')
                if reason not in reasons_summary:
                    reasons_summary[reason] = {'total': 0, 'saved_losses': 0, 'missed_wins': 0}
                reasons_summary[reason]['total'] += 1
                if r['virtual_result'] == 'VIRTUAL_LOSS':
                    reasons_summary[reason]['saved_losses'] += 1
                elif r['virtual_result'] == 'VIRTUAL_WIN':
                    reasons_summary[reason]['missed_wins'] += 1

            return {
                'total_rejected': total_rejected,
                'saved_losses': saved_losses,
                'missed_wins': missed_wins,
                'filter_accuracy': filter_accuracy,
                'reasons_breakdown': reasons_summary
            }
        except Exception as e:
            logger.error(f"❌ Failed to compute filter stats: {e}")
            return {'total_rejected': 0, 'saved_losses': 0, 'missed_wins': 0, 'filter_accuracy': 0.0, 'reasons_breakdown': {}}

    # =========================================================================
    # Multi-User Management & Access Control Layer
    # =========================================================================

    def register_or_get_user(
        self,
        telegram_id: int,
        username: str = "",
        first_name: str = "",
        last_name: str = "",
        default_role: str = "investor",
        default_status: str = "approved"
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Retrieves user if exists, or registers a new user with default preferences.
        Returns (user_dict, is_new).
        """
        now = datetime.now().isoformat()
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute("SELECT * FROM users WHERE telegram_id = ?", (telegram_id,))
                row = cursor.fetchone()

                if row:
                    user = dict(row)
                    # Update username / first_name if changed, and bump last_active_at
                    cursor.execute("""
                        UPDATE users
                        SET username = ?, first_name = ?, last_name = ?, last_active_at = ?
                        WHERE telegram_id = ?
                    """, (username or user.get("username"), first_name or user.get("first_name"), last_name or user.get("last_name"), now, telegram_id))
                    conn.commit()
                    user["last_active_at"] = now
                    conn.close()
                    return user, False

                # Register brand new user
                cursor.execute("""
                    INSERT INTO users (
                        telegram_id, username, first_name, last_name, role, status, onboarded, registered_at, last_active_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
                """, (telegram_id, username, first_name, last_name, default_role, default_status, now, now))

                cursor.execute("""
                    INSERT OR IGNORE INTO user_preferences (
                        telegram_id, timezone, notifications_enabled, created_at, updated_at
                    ) VALUES (?, 'UTC', 1, ?, ?)
                """, (telegram_id, now, now))

                conn.commit()

                cursor.execute("SELECT * FROM users WHERE telegram_id = ?", (telegram_id,))
                new_user = dict(cursor.fetchone())
                conn.close()

                logger.info(f"👤 [UserManager] New Telegram user registered: #{telegram_id} (@{username or 'no_user'}) [{default_role}]")
                return new_user, True
        except Exception as e:
            logger.error(f"❌ Failed to register/get user #{telegram_id}: {e}")
            return {
                "telegram_id": telegram_id,
                "username": username,
                "first_name": first_name,
                "last_name": last_name,
                "role": default_role,
                "status": default_status,
                "onboarded": 0,
                "registered_at": now,
                "last_active_at": now
            }, False

    def get_user(self, telegram_id: int) -> Optional[Dict[str, Any]]:
        """Fetch user record by Telegram ID."""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE telegram_id = ?", (telegram_id,))
                row = cursor.fetchone()
                conn.close()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"❌ Failed to get user #{telegram_id}: {e}")
            return None

    def update_user_status(self, telegram_id: int, status: str) -> bool:
        """Update access status ('approved', 'pending', 'suspended')."""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE users
                    SET status = ?
                    WHERE telegram_id = ?
                """, (status.lower(), telegram_id))
                updated = cursor.rowcount > 0
                conn.commit()
                conn.close()
                logger.info(f"🔒 [UserManager] User #{telegram_id} status changed to: {status}")
                return updated
        except Exception as e:
            logger.error(f"❌ Failed to update status for user #{telegram_id}: {e}")
            return False

    def update_user_role(self, telegram_id: int, role: str) -> bool:
        """Update role ('admin', 'investor')."""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE users
                    SET role = ?
                    WHERE telegram_id = ?
                """, (role.lower(), telegram_id))
                updated = cursor.rowcount > 0
                conn.commit()
                conn.close()
                return updated
        except Exception as e:
            logger.error(f"❌ Failed to update role for user #{telegram_id}: {e}")
            return False

    def complete_onboarding(self, telegram_id: int) -> bool:
        """Mark user as having completed disclaimer onboarding."""
        now = datetime.now().isoformat()
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE users
                    SET onboarded = 1, last_active_at = ?
                    WHERE telegram_id = ?
                """, (now, telegram_id))
                updated = cursor.rowcount > 0
                conn.commit()
                conn.close()
                logger.info(f"✅ [UserManager] User #{telegram_id} completed demo disclaimer onboarding.")
                return updated
        except Exception as e:
            logger.error(f"❌ Failed to mark onboarding complete for user #{telegram_id}: {e}")
            return False

    def update_last_active(self, telegram_id: int) -> bool:
        """Touch last_active_at timestamp."""
        now = datetime.now().isoformat()
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                cursor = conn.cursor()
                cursor.execute("UPDATE users SET last_active_at = ? WHERE telegram_id = ?", (now, telegram_id))
                conn.commit()
                conn.close()
                return True
        except Exception as e:
            logger.error(f"❌ Failed to touch last_active for user #{telegram_id}: {e}")
            return False

    def list_users(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List registered users, optionally filtered by status."""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                if status:
                    cursor.execute("SELECT * FROM users WHERE status = ? ORDER BY registered_at DESC", (status.lower(),))
                else:
                    cursor.execute("SELECT * FROM users ORDER BY registered_at DESC")
                rows = cursor.fetchall()
                conn.close()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"❌ Failed to list users: {e}")
            return []

    def get_user_counts(self) -> Dict[str, int]:
        """Return total, active/approved, pending, and suspended user counts."""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                cursor = conn.cursor()
                cursor.execute("SELECT status, COUNT(*) FROM users GROUP BY status")
                rows = cursor.fetchall()
                conn.close()
                stats = {"total": 0, "approved": 0, "pending": 0, "suspended": 0}
                for status_name, count in rows:
                    stats[status_name] = count
                    stats["total"] += count
                return stats
        except Exception as e:
            logger.error(f"❌ Failed to get user counts: {e}")
            return {"total": 0, "approved": 0, "pending": 0, "suspended": 0}

    def get_user_preferences(self, telegram_id: int) -> Dict[str, Any]:
        """Fetch user-specific preferences (timezone, notifications)."""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM user_preferences WHERE telegram_id = ?", (telegram_id,))
                row = cursor.fetchone()
                conn.close()
                if row:
                    return dict(row)
                return {"telegram_id": telegram_id, "timezone": "UTC", "notifications_enabled": 1}
        except Exception as e:
            logger.error(f"❌ Failed to fetch preferences for user #{telegram_id}: {e}")
            return {"telegram_id": telegram_id, "timezone": "UTC", "notifications_enabled": 1}

    def set_user_preferences(
        self,
        telegram_id: int,
        timezone: Optional[Any] = None,
        notifications_enabled: Optional[bool] = None,
        **kwargs
    ) -> bool:
        """Update user-specific display timezone or notification preferences."""
        if isinstance(timezone, dict):
            pref_dict = timezone
            timezone = pref_dict.get("timezone")
            if "notifications_enabled" in pref_dict:
                notifications_enabled = pref_dict["notifications_enabled"]
        if "timezone" in kwargs:
            timezone = kwargs["timezone"]
        if "notifications_enabled" in kwargs:
            notifications_enabled = kwargs["notifications_enabled"]

        now = datetime.now().isoformat()
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM user_preferences WHERE telegram_id = ?", (telegram_id,))
                exists = cursor.fetchone() is not None

                if exists:
                    updates = []
                    params = []
                    if timezone is not None:
                        updates.append("timezone = ?")
                        params.append(timezone)
                    if notifications_enabled is not None:
                        updates.append("notifications_enabled = ?")
                        params.append(1 if notifications_enabled else 0)
                    updates.append("updated_at = ?")
                    params.append(now)
                    params.append(telegram_id)

                    query = f"UPDATE user_preferences SET {', '.join(updates)} WHERE telegram_id = ?"
                    cursor.execute(query, tuple(params))
                else:
                    cursor.execute("""
                        INSERT INTO user_preferences (
                            telegram_id, timezone, notifications_enabled, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?)
                    """, (
                        telegram_id,
                        timezone or "UTC",
                        1 if notifications_enabled or notifications_enabled is None else 0,
                        now,
                        now
                    ))
                conn.commit()
                conn.close()
                return True
        except Exception as e:
            logger.error(f"❌ Failed to set preferences for user #{telegram_id}: {e}")
            return False

    def log_admin_action(
        self,
        admin_id: int,
        action: str,
        target_user_id: Optional[int] = None,
        details: Optional[str] = None
    ) -> bool:
        """Audit record of administrator actions."""
        now = datetime.now().isoformat()
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO admin_audit_logs (
                        admin_id, action, target_user_id, details, timestamp
                    ) VALUES (?, ?, ?, ?, ?)
                """, (admin_id, action, target_user_id, details or "", now))
                conn.commit()
                conn.close()
                logger.info(f"🛡️ [AdminAudit] Admin #{admin_id} action '{action}' on target #{target_user_id}: {details}")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to log admin action: {e}")
            return False

    def log_user_activity(
        self,
        telegram_id: int,
        action: str,
        details: Optional[str] = None
    ) -> bool:
        """Log user activities."""
        now = datetime.now().isoformat()
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO user_activity_logs (
                        telegram_id, action, details, timestamp
                    ) VALUES (?, ?, ?, ?)
                """, (telegram_id, action, details or "", now))
                conn.commit()
                conn.close()
                return True
        except Exception as e:
            logger.error(f"❌ Failed to log user activity for #{telegram_id}: {e}")
            return False

    def get_recent_audit_logs(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Retrieve recent administrative audit logs."""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM admin_audit_logs
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
                rows = cursor.fetchall()
                conn.close()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"❌ Failed to retrieve audit logs: {e}")
            return []

    def clear_trade_history(self) -> bool:
        """Purge all recorded trades and virtual trades while preserving users and preferences."""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM trades")
                cursor.execute("DELETE FROM virtual_trades")
                conn.commit()
                conn.close()
                logger.info("🧹 [TradeDatabase] Cleared trades and virtual_trades tables successfully.")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to clear trade database history: {e}")
            return False

    def clear_trades_before_today(self) -> bool:
        """Purge trades before today while preserving today's trades and user tables."""
        try:
            today_start = datetime.now(timezone.utc).strftime("%Y-%m-%d 00:00:00")
            with self._lock:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM trades WHERE created_at < ?", (today_start,))
                cursor.execute("DELETE FROM virtual_trades WHERE created_at < ?", (today_start,))
                conn.commit()
                conn.close()
                logger.info(f"🧹 [TradeDatabase] Cleared trades prior to {today_start} UTC")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to clear pre-today trades: {e}")
            return False

    def get_system_setting(self, key: str, default: Any = None) -> Any:
        """Fetch a persistent system configuration setting."""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM system_settings WHERE key = ?", (key,))
                row = cursor.fetchone()
                conn.close()
                if row:
                    return row[0]
                return default
        except Exception as e:
            logger.error(f"❌ Failed to fetch system setting {key}: {e}")
            return default

    def set_system_setting(self, key: str, value: Any) -> bool:
        """Save or update a persistent system configuration setting."""
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            with self._lock:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO system_settings (key, value, updated_at) VALUES (?, ?, ?)",
                    (key, str(value), now_iso)
                )
                conn.commit()
                conn.close()
                return True
        except Exception as e:
            logger.error(f"❌ Failed to set system setting {key}: {e}")
            return False


# Global database instance
db = TradeDatabase()
