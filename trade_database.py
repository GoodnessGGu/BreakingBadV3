# trade_database.py
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os

logger = logging.getLogger(__name__)

DATABASE_PATH = os.getenv("DATABASE_PATH", "trades.db")


class TradeDatabase:
    """Manages trade history storage and retrieval using SQLite."""
    
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database and create tables if they don't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
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

            conn.commit()
            conn.close()
            logger.info(f"✅ Trade database & counterfactual virtual trades table initialized: {self.db_path}")
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


# Global database instance
db = TradeDatabase()
