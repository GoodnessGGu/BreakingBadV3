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
            
            conn.commit()
            conn.close()
            logger.info(f"✅ Trade database initialized: {self.db_path}")
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
            
            return {
                'total_trades': len(trades),
                'wins': wins,
                'losses': losses,
                'win_rate': (wins / len(trades) * 100) if trades else 0,
                'total_profit': total_profit,
                'avg_profit': total_profit / len(trades) if trades else 0
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


# Global database instance
db = TradeDatabase()
