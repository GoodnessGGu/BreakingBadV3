"""
bot/trade_ledger.py - Persistent SQLite Trade Ledger for BreakingBad V3

Provides SQLite persistence, trade reconciliation with broker MCP endpoints,
multi-leg trade aggregation (by group_id), strategy and period filtering,
performance statistics (win rate, profit factor, drawdown), and CSV export.
"""

import os
import csv
import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("TradeLedger")

DEFAULT_LEDGER_PATH = os.getenv("LEDGER_PATH", os.path.join(os.getcwd(), "data", "trade_ledger.db"))


def _parse_to_utc_string(val: Any) -> str:
    """Safely convert timestamp, ISO string, or formatted string to '%Y-%m-%d %H:%M:%S' UTC."""
    if val is None:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(val, (int, float)):
        if val > 1e11:  # Milliseconds
            val = val / 1000.0
        try:
            return datetime.fromtimestamp(val, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    s = str(val).strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass

    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return s


class TradeLedger:
    """
    Persistent SQLite Trade Ledger & Reconciler.
    Supports multi-leg execution linking, strategy/period filtering,
    reconciliation with broker endpoints, performance statistics, and CSV export.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_LEDGER_PATH
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Create tables and indexes if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    position_id TEXT PRIMARY KEY,
                    strategy TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    lots REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    sl REAL,
                    tp REAL,
                    open_time_utc TEXT NOT NULL,
                    close_time_utc TEXT,
                    exit_price REAL,
                    pnl REAL,
                    exit_reason TEXT,
                    balance_type TEXT DEFAULT 'training',
                    source_message_id TEXT,
                    group_id TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_close_time ON trades(close_time_utc)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_group_id ON trades(group_id)")
            conn.commit()

    def record_open(
        self,
        position_id: Any,
        strategy: str,
        symbol: str,
        side: str,
        lots: float,
        entry_price: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        open_time_utc: Optional[str] = None,
        balance_type: str = "training",
        source_message_id: Optional[Any] = None,
        group_id: Optional[str] = None
    ) -> bool:
        """Insert or update an open trade in the ledger."""
        pid_str = str(position_id).strip()
        open_time = _parse_to_utc_string(open_time_utc)
        side_norm = side.upper()
        if side_norm in ("LONG", "BUY"):
            side_norm = "BUY"
        elif side_norm in ("SHORT", "SELL"):
            side_norm = "SELL"

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO trades (
                        position_id, strategy, symbol, side, lots, entry_price,
                        sl, tp, open_time_utc, balance_type, source_message_id, group_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(position_id) DO UPDATE SET
                        strategy=excluded.strategy,
                        symbol=excluded.symbol,
                        side=excluded.side,
                        lots=excluded.lots,
                        entry_price=excluded.entry_price,
                        sl=COALESCE(excluded.sl, trades.sl),
                        tp=COALESCE(excluded.tp, trades.tp),
                        balance_type=excluded.balance_type,
                        source_message_id=COALESCE(excluded.source_message_id, trades.source_message_id),
                        group_id=COALESCE(excluded.group_id, trades.group_id)
                """, (
                    pid_str, strategy, symbol, side_norm, float(lots), float(entry_price),
                    float(sl) if sl is not None else None,
                    float(tp) if tp is not None else None,
                    open_time, balance_type.lower(),
                    str(source_message_id) if source_message_id is not None else None,
                    str(group_id) if group_id is not None else None
                ))
                conn.commit()
                logger.info(f"📖 [Ledger] Recorded OPEN: #{pid_str} ({strategy} | {symbol} {side_norm} {lots}L @ {entry_price})")
                return True
        except Exception as e:
            logger.error(f"[Ledger] Error recording open for #{pid_str}: {e}")
            return False

    def record_close(
        self,
        position_id: Any,
        exit_price: Optional[float] = None,
        pnl: Optional[float] = None,
        exit_reason: Optional[str] = None,
        close_time_utc: Optional[str] = None,
        strategy_fallback: Optional[str] = None,
        symbol_fallback: Optional[str] = None,
        side_fallback: Optional[str] = None,
        lots_fallback: Optional[float] = None,
        entry_fallback: Optional[float] = None
    ) -> bool:
        """Record trade closure with confirmed exit price and PnL."""
        pid_str = str(position_id).strip()
        close_time = _parse_to_utc_string(close_time_utc)

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT position_id FROM trades WHERE position_id = ?", (pid_str,))
                row = cursor.fetchone()

                if row:
                    cursor.execute("""
                        UPDATE trades SET
                            exit_price = COALESCE(?, exit_price),
                            pnl = ?,
                            exit_reason = COALESCE(?, exit_reason),
                            close_time_utc = COALESCE(?, close_time_utc)
                        WHERE position_id = ?
                    """, (
                        float(exit_price) if exit_price is not None else None,
                        float(pnl) if pnl is not None else None,
                        exit_reason, close_time, pid_str
                    ))
                else:
                    # Missing from ledger (e.g. broker manual close)
                    cursor.execute("""
                        INSERT INTO trades (
                            position_id, strategy, symbol, side, lots, entry_price,
                            open_time_utc, close_time_utc, exit_price, pnl, exit_reason
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pid_str,
                        strategy_fallback or "Manual/Unattributed",
                        symbol_fallback or "UNKNOWN",
                        side_fallback or "BUY",
                        float(lots_fallback or 1.0),
                        float(entry_fallback or exit_price or 0.0),
                        close_time, close_time,
                        float(exit_price) if exit_price is not None else None,
                        float(pnl) if pnl is not None else None,
                        exit_reason or "broker_sync"
                    ))

                conn.commit()
                logger.info(f"📖 [Ledger] Recorded CLOSE: #{pid_str} (PnL: {pnl} | Reason: {exit_reason})")
                return True
        except Exception as e:
            logger.error(f"[Ledger] Error recording close for #{pid_str}: {e}")
            return False

    def _build_strategy_clause(self, strategy: Optional[str]) -> Tuple[str, List[Any]]:
        if not strategy or strategy.lower() == "all":
            return "", []
        strat_low = strategy.lower().strip()
        if strat_low in ("cfd", "copiers", "copier"):
            return " AND LOWER(strategy) IN ('callisto', 'gold pips', 'gsociety', 'kingmahn', 'copier', 'cfd')", []
        elif strat_low in ("manual", "unattributed", "manual/unattributed"):
            return " AND LOWER(strategy) IN ('manual', 'manual/unattributed', 'unattributed')", []
        elif strat_low in ("snd", "s&d"):
            return " AND LOWER(strategy) IN ('snd', 's&d')", []
        elif strat_low in ("trendline", "trendlines", "tl"):
            return " AND LOWER(strategy) IN ('trendline', 'trendlines', 'tl')", []
        return " AND LOWER(strategy) = LOWER(?)", [strategy]

    def get_open_trades(self, strategy: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve all currently open trades (close_time_utc IS NULL)."""
        query = "SELECT * FROM trades WHERE close_time_utc IS NULL"
        strat_clause, params = self._build_strategy_clause(strategy)
        query += strat_clause + " ORDER BY open_time_utc DESC"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(r) for r in cursor.fetchall()]

    def get_closed_trades(
        self,
        strategy: Optional[str] = None,
        period: str = "all",
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Retrieve closed trades with filtering."""
        query = "SELECT * FROM trades WHERE close_time_utc IS NOT NULL"
        strat_clause, params = self._build_strategy_clause(strategy)
        query += strat_clause

        now_utc = datetime.now(timezone.utc)
        if period == "today":
            start_date = now_utc.strftime("%Y-%m-%d 00:00:00")
            query += " AND close_time_utc >= ?"
            params.append(start_date)
        elif period == "7d":
            start_date = (now_utc - timedelta(days=7)).strftime("%Y-%m-%d 00:00:00")
            query += " AND close_time_utc >= ?"
            params.append(start_date)
        elif period == "30d":
            start_date = (now_utc - timedelta(days=30)).strftime("%Y-%m-%d 00:00:00")
            query += " AND close_time_utc >= ?"
            params.append(start_date)

        query += " ORDER BY close_time_utc DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(r) for r in cursor.fetchall()]

    def get_grouped_closed_trades(
        self,
        strategy: Optional[str] = None,
        period: str = "all",
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve closed trades aggregating multi-leg trades by group_id.
        Each multi-leg trade counts as ONE aggregate record with summed PnL.
        """
        raw_trades = self.get_closed_trades(strategy=strategy, period=period, limit=2000, offset=0)

        grouped: Dict[str, Dict[str, Any]] = {}
        ordered_keys: List[str] = []

        for t in raw_trades:
            gid = t.get("group_id")
            key = f"grp_{gid}" if gid else f"pos_{t['position_id']}"

            if key not in grouped:
                ordered_keys.append(key)
                grouped[key] = {
                    "is_grouped": bool(gid),
                    "group_id": gid,
                    "position_ids": [t["position_id"]],
                    "strategy": t["strategy"],
                    "symbol": t["symbol"],
                    "side": t["side"],
                    "total_lots": float(t["lots"]),
                    "entry_price": float(t["entry_price"]),
                    "exit_price": float(t["exit_price"]) if t.get("exit_price") is not None else None,
                    "pnl": float(t["pnl"]) if t.get("pnl") is not None else 0.0,
                    "has_unconfirmed_pnl": t.get("pnl") is None,
                    "close_time_utc": t["close_time_utc"],
                    "open_time_utc": t["open_time_utc"],
                    "exit_reason": t["exit_reason"],
                    "legs": [t]
                }
            else:
                g = grouped[key]
                g["position_ids"].append(t["position_id"])
                g["total_lots"] += float(t["lots"])
                if t.get("pnl") is not None:
                    g["pnl"] += float(t["pnl"])
                else:
                    g["has_unconfirmed_pnl"] = True
                if t.get("close_time_utc") and str(t["close_time_utc"]) > str(g["close_time_utc"]):
                    g["close_time_utc"] = t["close_time_utc"]
                g["legs"].append(t)

        all_grouped = [grouped[k] for k in ordered_keys]
        return all_grouped[offset : offset + limit]

    def count_grouped_closed_trades(self, strategy: Optional[str] = None, period: str = "all") -> int:
        """Count distinct grouped trades matching the filters."""
        raw_trades = self.get_closed_trades(strategy=strategy, period=period, limit=10000, offset=0)
        seen = set()
        count = 0
        for t in raw_trades:
            gid = t.get("group_id")
            key = f"grp_{gid}" if gid else f"pos_{t['position_id']}"
            if key not in seen:
                seen.add(key)
                count += 1
        return count

    def get_stats(self, strategy: Optional[str] = None, period: str = "all") -> Dict[str, Any]:
        """
        Calculate performance stats aggregating multi-leg trades by group_id.
        Multi-leg trades count as ONE trade in count, win rate, and profit factor.
        """
        grouped_trades = self.get_grouped_closed_trades(strategy=strategy, period=period, limit=10000, offset=0)

        total_trades = len(grouped_trades)
        if total_trades == 0:
            return {
                "total_trades": 0, "wins": 0, "losses": 0, "breakevens": 0,
                "win_rate": 0.0, "net_pnl": 0.0, "profit_factor": 0.0,
                "gross_profit": 0.0, "gross_loss": 0.0, "avg_win": 0.0,
                "avg_loss": 0.0, "max_drawdown": 0.0
            }

        wins = 0
        losses = 0
        breakevens = 0
        gross_profit = 0.0
        gross_loss = 0.0
        net_pnl = 0.0
        peak = 0.0
        cumulative = 0.0
        max_drawdown = 0.0

        # Sort chronologically for drawdown calculation
        sorted_trades = sorted(grouped_trades, key=lambda x: str(x.get("close_time_utc") or ""))

        for t in sorted_trades:
            pnl = t.get("pnl", 0.0)
            net_pnl += pnl
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_drawdown:
                max_drawdown = dd

            if pnl > 0.001:
                wins += 1
                gross_profit += pnl
            elif pnl < -0.001:
                losses += 1
                gross_loss += abs(pnl)
            else:
                breakevens += 1

        win_rate = (wins / total_trades) * 100.0 if total_trades > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
        avg_win = (gross_profit / wins) if wins > 0 else 0.0
        avg_loss = (gross_loss / losses) if losses > 0 else 0.0

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "breakevens": breakevens,
            "win_rate": round(win_rate, 1),
            "net_pnl": round(net_pnl, 2),
            "profit_factor": round(profit_factor, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "max_drawdown": round(max_drawdown, 2)
        }

    def reconcile_with_broker(self, mcp_client: Any, balance_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Reconcile ledger against broker trade history across all endpoints with pagination.
        Fills rows with NULL pnl, and inserts missing trades as 'Manual/Unattributed'.
        """
        if not mcp_client:
            return {"ok": False, "message": "No MCP client provided"}

        bid = balance_id
        if not bid:
            try:
                bal = mcp_client.get_training_balance() or mcp_client.get_real_balance()
                if bal:
                    bid = bal.get("balance_id")
            except Exception:
                pass

        if not bid:
            return {"ok": False, "message": "Could not determine balance_id"}

        # Paginate broker history
        all_broker_history: List[Dict[str, Any]] = []
        skip = 0
        limit = 50
        max_pages = 5
        partial_sync = False

        for _ in range(max_pages):
            try:
                hist = mcp_client.get_trade_history(balance_id=bid, skip=skip, limit=limit)
                if hist is None:
                    partial_sync = True
                    break
                if not hist:
                    break
                all_broker_history.extend(hist)
                if len(hist) < limit:
                    break
                skip += limit
            except Exception as e:
                logger.warning(f"[Ledger Recon] History page fetch error: {e}")
                partial_sync = True
                break

        updated_count = 0
        inserted_count = 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            for b_trade in all_broker_history:
                pid = str(b_trade.get("position_id") or b_trade.get("id") or b_trade.get("order_id"))
                if not pid or pid == "None":
                    continue

                pnl = None
                if b_trade.get("pnl") is not None:
                    try:
                        pnl = float(b_trade.get("pnl"))
                    except (ValueError, TypeError):
                        pass

                exit_px = float(b_trade.get("close_price") or b_trade.get("exit_price", 0.0))
                entry_px = float(b_trade.get("open_price") or b_trade.get("entry_price", 0.0))
                side = str(b_trade.get("side") or b_trade.get("direction", "BUY")).upper()
                lots = float(b_trade.get("lots") or b_trade.get("count", 1.0))
                reason = b_trade.get("close_reason") or "broker_close"
                close_time = _parse_to_utc_string(b_trade.get("close_time") or b_trade.get("time"))
                open_time = _parse_to_utc_string(b_trade.get("open_time") or b_trade.get("time"))

                cursor.execute("SELECT position_id, pnl FROM trades WHERE position_id = ?", (pid,))
                row = cursor.fetchone()

                if row:
                    # Update if pnl was NULL or unconfirmed
                    if row["pnl"] is None and pnl is not None:
                        cursor.execute("""
                            UPDATE trades SET
                                pnl = ?, exit_price = COALESCE(?, exit_price),
                                close_time_utc = COALESCE(?, close_time_utc),
                                exit_reason = COALESCE(?, exit_reason)
                            WHERE position_id = ?
                        """, (pnl, exit_px, close_time, reason, pid))
                        updated_count += 1
                else:
                    # Insert missing broker trade as Manual/Unattributed
                    cursor.execute("""
                        INSERT INTO trades (
                            position_id, strategy, symbol, side, lots, entry_price,
                            open_time_utc, close_time_utc, exit_price, pnl, exit_reason
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pid, "Manual/Unattributed",
                        str(b_trade.get("asset_id") or "UNKNOWN"),
                        side, lots, entry_px, open_time, close_time, exit_px, pnl, reason
                    ))
                    inserted_count += 1

            conn.commit()

        return {
            "ok": True,
            "partial_sync": partial_sync,
            "broker_trades_fetched": len(all_broker_history),
            "updated_pnl_count": updated_count,
            "inserted_manual_count": inserted_count
        }

    def export_csv(self, filepath: str, strategy: Optional[str] = None, period: str = "all") -> str:
        """Export ledger trades to CSV file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        trades = self.get_closed_trades(strategy=strategy, period=period, limit=10000, offset=0)

        fields = [
            "position_id", "group_id", "strategy", "symbol", "side", "lots",
            "entry_price", "exit_price", "pnl", "sl", "tp", "open_time_utc",
            "close_time_utc", "exit_reason", "balance_type"
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for t in trades:
                writer.writerow(t)

        logger.info(f"📊 [Ledger] Exported {len(trades)} trades to {filepath}")
        return filepath
