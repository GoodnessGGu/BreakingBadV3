"""
bot/telemetry_server.py - Strictly Read-Only WebSocket & Health Server for BreakingBad V3

Exposes:
  - GET / and GET /health : Lightweight HTTP health check (JSON)
  - GET /ws/live?token=... : Authenticated read-only WebSocket telemetry stream

Security & Sync:
  - Strictly read-only observer: no trade execution, closing, or settings mutation routes.
  - Requires DASHBOARD_API_TOKEN (or WS_API_TOKEN) matched via constant-time comparison.
  - Redacts bot tokens and bearer secrets from all streamed log lines.
  - Reconciles/reads live broker balance, open positions, and closed trade history
    (IQ Option Marginal CFD/Forex/Crypto + Blitz + shared TradeLedger) when clients connect.
"""

import os
import re
import hmac
import json
import time
import logging
import asyncio
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from aiohttp import web, WSMsgType

from bot.trade_ledger import TradeLedger
from bot.history_formatter import DEFAULT_SYMBOL_MAP, resolve_symbol_name

logger = logging.getLogger("TelemetryServer")


def _redact_secrets(text: str) -> str:
    text = re.sub(r"bot\d+:[A-Za-z0-9_-]+", "bot[REDACTED]", text)
    text = re.sub(r"(Bearer\s+)[A-Za-z0-9_\-\.]{16,}", r"\1[REDACTED]", text)
    return text


def _infer_log_engine_and_tag(logger_name: str, message: str) -> Dict[str, Optional[str]]:
    lname = logger_name.lower()
    if "ict" in lname:
        return {"engine": "ICT", "tag": "ICT_ENGINE"}
    if "snd" in lname:
        return {"engine": "S&D", "tag": "SND_ENGINE"}
    if "crt" in lname:
        return {"engine": "CRT", "tag": "CRT_ENGINE"}
    if "trendline" in lname:
        return {"engine": "TREND", "tag": "TRENDLINE"}
    if "channel" in lname or "copier" in lname or "callisto" in lname or "gold" in lname:
        return {"engine": "VIP", "tag": "VIP_FEED"}
    if "propfirm" in lname or "guardian" in lname:
        return {"engine": None, "tag": "GUARDIAN"}
    if "trailing" in message.lower() or "breakeven" in message.lower():
        return {"engine": None, "tag": "TRAILING"}
    return {"engine": None, "tag": logger_name.upper()[:16] or "SYSTEM"}


def _infer_log_level(record: logging.LogRecord, message: str) -> str:
    lower = message.lower()
    if record.levelno >= logging.WARNING:
        return "warn"
    if any(k in lower for k in ("won", "opened", "closed", "filled", "breakeven", "stage", "connected", "✅", "🏆", "🛡")):
        return "success"
    if any(k in lower for k in ("monitoring", "scan", "sweep", "fvg", "imbalance", "analyzed", "🔍")):
        return "scan"
    return "info"


class TelemetryLogHandler(logging.Handler):
    """Captures recent engine logs in memory for streaming to read-only dashboard clients."""

    def __init__(self, maxlen: int = 60):
        super().__init__(level=logging.INFO)
        self.buffer: deque = deque(maxlen=maxlen)
        self._seq = 0

    def emit(self, record: logging.LogRecord):
        try:
            msg = _redact_secrets(record.getMessage())
            meta = _infer_log_engine_and_tag(record.name, msg)
            self._seq += 1
            entry: Dict[str, Any] = {
                "id": f"srv-log-{self._seq}",
                "time": datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%H:%M:%S"),
                "tag": meta["tag"],
                "message": msg[:260],
                "level": _infer_log_level(record, msg),
            }
            if meta["engine"]:
                entry["engine"] = meta["engine"]
            self.buffer.appendleft(entry)
        except Exception:
            pass


def _calculate_pips(symbol: str, open_px: float, cur_px: float, side: str) -> float:
    """Matches TelegramTradingBot.calculate_pips exactly."""
    if open_px <= 0 or cur_px <= 0:
        return 0.0
    diff = (cur_px - open_px) if side.upper() in ("BUY", "LONG") else (open_px - cur_px)
    s = str(symbol).upper().replace("/", "").replace("-", "")
    if "XAU" in s or "GOLD" in s or s == "74":
        return round(diff * 10.0, 1)
    if "XAG" in s or "SILVER" in s or s in ("1487", "54"):
        return round(diff * 100.0, 1)
    if "JPY" in s:
        return round(diff * 100.0, 1)
    if any(k in s for k in ("BTC", "ETH", "SOL", "XRP", "NAS100", "US100", "US500", "OIL", "WTI", "BRENT")):
        return round(diff, 1)
    return round(diff * 10000.0, 1)


def _normalize_engine_code(raw_strategy: str) -> str:
    s = (raw_strategy or "").upper()
    if "TREND" in s or s == "TL":
        return "TREND"
    if "ICT" in s or "STRADDLE" in s:
        return "ICT"
    if "SND" in s or "S&D" in s:
        return "S&D"
    if "CRT" in s:
        return "CRT"
    if any(k in s for k in ("CALLISTO", "GOLD", "GSOCIETY", "KING", "POLY", "VIP", "COPIER", "BLITZ")):
        return "VIP"
    return "CFD"


def _parse_iso_or_epoch(val: Any) -> Optional[datetime]:
    if val is None or val == "":
        return None
    if isinstance(val, (int, float)):
        ts = val / 1000.0 if val > 1e11 else float(val)
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None
    s = str(val).strip()
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def _format_duration(open_dt: Optional[datetime], close_dt: Optional[datetime]) -> str:
    if not open_dt or not close_dt:
        return "Closed"
    secs = max(0, int((close_dt - open_dt).total_seconds()))
    if secs < 60:
        return f"{secs}s"
    mins = secs // 60
    if mins < 60:
        return f"{mins}m"
    hours = mins // 60
    rem_mins = mins % 60
    return f"{hours}h {rem_mins}m" if rem_mins else f"{hours}h"


class TelemetryServer:
    """Read-only HTTP health + authenticated WebSocket telemetry server."""

    def __init__(
        self,
        forex_mcp: Any,
        channel_mgr: Any = None,
        ict_engine: Any = None,
        crt_engine: Any = None,
        snd_engine: Any = None,
        trendline_engine: Any = None,
        tg_bot: Any = None,
        account_type: str = "training",
    ):
        self.forex_mcp = forex_mcp
        self.channel_mgr = channel_mgr
        self.ict_engine = ict_engine
        self.crt_engine = crt_engine
        self.snd_engine = snd_engine
        self.trendline_engine = trendline_engine
        self.tg_bot = tg_bot
        self.account_type = account_type

        self.clients: Set[web.WebSocketResponse] = set()
        self.log_handler = TelemetryLogHandler(maxlen=60)
        self.ledger = getattr(tg_bot, "trade_ledger", None) or TradeLedger()
        self._runner: Optional[web.AppRunner] = None
        self._running = False
        self._handler_attached = False

        # Short-TTL cache for broker trade history so we don't spam MCP every 3s
        self._history_cache_ts: float = 0.0
        self._history_cache_ttl: float = 20.0
        self._history_cache_trades: List[Dict[str, Any]] = []

    def _get_expected_token(self) -> str:
        return (
            os.getenv("DASHBOARD_API_TOKEN")
            or os.getenv("WS_API_TOKEN")
            or os.getenv("API_TOKEN")
            or ""
        ).strip()

    def _is_authorized(self, request: web.Request) -> bool:
        expected = self._get_expected_token()
        if not expected:
            return False

        candidate = (request.query.get("token") or "").strip()
        if not candidate:
            auth = request.headers.get("Authorization", "")
            if auth.lower().startswith("bearer "):
                candidate = auth[7:].strip()

        if not candidate:
            return False
        return hmac.compare_digest(candidate, expected)

    def _get_symbol_map(self) -> Dict[str, str]:
        if self.tg_bot and hasattr(self.tg_bot, "get_cached_symbol_map"):
            try:
                return self.tg_bot.get_cached_symbol_map()
            except Exception:
                pass
        return dict(DEFAULT_SYMBOL_MAP)

    def _resolve_symbol(self, raw_pos: Dict[str, Any], symbol_map: Dict[str, str]) -> str:
        raw_sym = raw_pos.get("symbol") or raw_pos.get("asset_name") or raw_pos.get("active")
        if raw_sym:
            return resolve_symbol_name(raw_sym, symbol_map)
        aid = raw_pos.get("asset_id")
        if aid is not None:
            resolved = resolve_symbol_name(str(aid), symbol_map)
            if resolved and not resolved.startswith("ID"):
                return resolved
        inst = raw_pos.get("instrument_id")
        if inst:
            return resolve_symbol_name(str(inst), symbol_map)
        return "UNKNOWN"

    def _resolve_engine_for_position(self, pos_id: Any, pair: str, ledger_open_map: Dict[str, str]) -> str:
        pid_str = str(pos_id)

        # 1. Check live in-memory engine active_trades
        for code, eng in (
            ("ICT", self.ict_engine),
            ("CRT", self.crt_engine),
            ("S&D", self.snd_engine),
            ("TREND", self.trendline_engine),
        ):
            if eng and hasattr(eng, "active_trades"):
                for _, t in (eng.active_trades or {}).items():
                    if not t:
                        continue
                    if str(t.get("position_id")) == pid_str or str(t.get("order_id")) == pid_str:
                        return code
                    for leg in t.get("tickets", []) or []:
                        if str(leg.get("position_id")) == pid_str or str(leg.get("order_id")) == pid_str:
                            return code

        # 2. Check channel copiers
        if self.channel_mgr and hasattr(self.channel_mgr, "copiers"):
            for _, copier in (self.channel_mgr.copiers or {}).items():
                open_pos = getattr(copier, "open_positions", {}) or {}
                for oid, opos in open_pos.items():
                    if str(oid) == pid_str or str(opos.get("position_id")) == pid_str:
                        return "VIP"

        # 3. Check persisted TradeLedger open trades (survives container restarts if volume/DB kept)
        if pid_str in ledger_open_map:
            return _normalize_engine_code(ledger_open_map[pid_str])

        # 4. Infer from exclusive/configured strategy symbol assignments
        if pair in ("NAS100", "US100"):
            return "TREND"
        for code, eng in (
            ("ICT", self.ict_engine),
            ("CRT", self.crt_engine),
            ("S&D", self.snd_engine),
            ("TREND", self.trendline_engine),
        ):
            syms = getattr(eng, "enabled_symbols", None) or set()
            if pair in syms:
                return code

        return "CFD"

    def _build_strategies_payload(self, mapped_positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ict_active = bool(self.ict_engine and getattr(self.ict_engine, "is_enabled", False))
        trend_active = bool(self.trendline_engine and getattr(self.trendline_engine, "is_enabled", False))
        snd_active = bool(self.snd_engine and getattr(self.snd_engine, "is_enabled", False))
        crt_active = bool(self.crt_engine and getattr(self.crt_engine, "is_enabled", False))
        vip_active = bool(self.channel_mgr and getattr(self.channel_mgr, "is_running", False))

        open_by_eng: Dict[str, int] = {}
        for p in mapped_positions:
            code = p.get("engine", "")
            open_by_eng[code] = open_by_eng.get(code, 0) + 1

        ict_syms = ", ".join(sorted(getattr(self.ict_engine, "enabled_symbols", []) or []))
        trend_syms = ", ".join(sorted(getattr(self.trendline_engine, "enabled_symbols", []) or []))
        snd_syms = ", ".join(sorted(getattr(self.snd_engine, "enabled_symbols", []) or []))
        crt_syms = ", ".join(sorted(getattr(self.crt_engine, "enabled_symbols", []) or []))
        copier_count = len(getattr(self.channel_mgr, "copiers", {}) or {})

        return [
            {
                "id": "ict",
                "code": "ICT",
                "name": "ICT Engine",
                "status": "ACTIVE" if open_by_eng.get("ICT", 0) > 0 else ("SCANNING" if ict_active else "IDLE"),
                "subModel": f"Smart Money FVG ({ict_syms or 'XAUUSD'})",
                "confidence": 86 if ict_active else 0,
            },
            {
                "id": "trend",
                "code": "TREND",
                "name": "Trendline Engine",
                "status": "ACTIVE" if open_by_eng.get("TREND", 0) > 0 else ("SCANNING" if trend_active else "IDLE"),
                "subModel": f"15M Bounce Model ({trend_syms or 'NAS100, XAUUSD'})",
                "confidence": 82 if trend_active else 0,
            },
            {
                "id": "crt",
                "code": "CRT",
                "name": "CRT Engine",
                "status": "ACTIVE" if open_by_eng.get("CRT", 0) > 0 else ("SCANNING" if crt_active else "IDLE"),
                "subModel": f"Candle Range Theory ({crt_syms or 'EURUSD'})",
                "confidence": 74 if crt_active else 0,
            },
            {
                "id": "snd",
                "code": "S&D",
                "name": "S&D Engine",
                "status": "ACTIVE" if open_by_eng.get("S&D", 0) > 0 else ("SCANNING" if snd_active else "IDLE"),
                "subModel": f"Supply/Demand Imbalances ({snd_syms or 'FX'})",
                "confidence": 81 if snd_active else 0,
            },
            {
                "id": "vip",
                "code": "VIP",
                "name": "VIP Channel Feeds",
                "status": "MONITORING" if vip_active else "IDLE",
                "subModel": f"Telethon Copier ({copier_count} channels)",
                "confidence": 78 if vip_active else 0,
            },
        ]

    def _fetch_broker_closed_trades(self, balance_id: Optional[int], symbol_map: Dict[str, str]) -> List[Dict[str, Any]]:
        now = time.time()
        if self._history_cache_trades and (now - self._history_cache_ts) < self._history_cache_ttl:
            return self._history_cache_trades

        # Build strategy lookup from TradeLedger (if available)
        ledger_by_pid: Dict[str, Dict[str, Any]] = {}
        try:
            for row in self.ledger.get_closed_trades(limit=100, offset=0):
                pid = str(row.get("position_id") or "")
                if pid:
                    ledger_by_pid[pid] = row
        except Exception as e:
            logger.debug(f"[Telemetry] Ledger read warning: {e}")

        merged: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()

        # 1. Direct IQ Option Marginal CFD/Forex/Crypto history (authoritative broker records)
        if balance_id and self.forex_mcp:
            try:
                broker_hist = self.forex_mcp.get_trade_history(balance_id=balance_id, skip=0, limit=100) or []
                for b in broker_hist:
                    pid = str(b.get("position_id") or b.get("id") or b.get("order_id") or "")
                    if not pid or pid in seen_ids:
                        continue
                    seen_ids.add(pid)

                    ledger_row = ledger_by_pid.get(pid, {})
                    pair = (
                        resolve_symbol_name(ledger_row["symbol"], symbol_map)
                        if ledger_row.get("symbol")
                        else self._resolve_symbol(b, symbol_map)
                    )
                    side_raw = (
                        b.get("side")
                        or b.get("direction")
                        or b.get("type")
                        or (b.get("raw_event") or {}).get("type")
                        or ledger_row.get("side")
                        or "BUY"
                    )
                    side = str(side_raw).upper()
                    if side == "LONG":
                        side = "BUY"
                    elif side == "SHORT":
                        side = "SELL"

                    entry_px = float(b.get("open_price") or b.get("entry_price") or ledger_row.get("entry_price") or 0.0)
                    exit_px = float(b.get("close_price") or b.get("exit_price") or ledger_row.get("exit_price") or entry_px)

                    pnl = 0.0
                    for k in ("pnl", "profit", "pnl_net", "isolated_pnl_net", "close_profit"):
                        if b.get(k) is not None:
                            try:
                                pnl = float(b.get(k))
                                break
                            except (ValueError, TypeError):
                                pass
                    if pnl == 0.0 and ledger_row.get("pnl") is not None:
                        pnl = float(ledger_row["pnl"])

                    pips = _calculate_pips(pair, entry_px, exit_px, side) if entry_px > 0 and exit_px > 0 else 0.0
                    result = "WIN" if pnl > 0.01 else ("LOSS" if pnl < -0.01 else "BREAKEVEN")
                    raw_strat = ledger_row.get("strategy") or ""
                    eng = _normalize_engine_code(raw_strat) if raw_strat else self._resolve_engine_for_position(pid, pair, {})

                    open_dt = _parse_iso_or_epoch(b.get("open_time") or ledger_row.get("open_time_utc"))
                    close_dt = _parse_iso_or_epoch(b.get("close_time") or b.get("time") or ledger_row.get("close_time_utc"))

                    merged.append({
                        "id": pid,
                        "pair": pair,
                        "engine": eng,
                        "type": side if side in ("BUY", "SELL") else "BUY",
                        "result": result,
                        "pnl": round(pnl, 2),
                        "pips": pips,
                        "rMultiple": round(pnl / 50.0, 2),
                        "exitReason": str(b.get("close_reason") or ledger_row.get("exit_reason") or "Closed"),
                        "closedAt": close_dt.strftime("%m-%d %H:%M:%S") if close_dt else "--:--:--",
                        "closedDate": close_dt.strftime("%Y-%m-%d") if close_dt else "",
                        "_sortTs": close_dt.timestamp() if close_dt else 0.0,
                        "duration": _format_duration(open_dt, close_dt),
                    })
            except Exception as e:
                logger.debug(f"[Telemetry] Broker trade history fetch warning: {e}")

        # 2. Include Blitz Options history if Blitz MCP is connected
        blitz_mcp = getattr(self.tg_bot, "blitz_mcp", None) if self.tg_bot else None
        if blitz_mcp:
            try:
                blitz_hist = blitz_mcp.get_trade_history(limit=25, use_cache=True) or []
                for bt in blitz_hist:
                    pid = f"blitz-{bt.get('position_id') or bt.get('id') or len(merged)}"
                    if pid in seen_ids:
                        continue
                    seen_ids.add(pid)
                    pair = resolve_symbol_name(
                        bt.get("asset_name") or bt.get("active") or bt.get("asset_id") or "BLITZ",
                        symbol_map,
                    )
                    profit = float(bt.get("profit") or 0.0)
                    res_str = str(bt.get("result") or "").lower()
                    result = "WIN" if (res_str == "win" or profit > 0.01) else ("LOSS" if (res_str in ("loss", "loose") or profit < -0.01) else "BREAKEVEN")
                    close_dt = _parse_iso_or_epoch(bt.get("close_time") or bt.get("open_time") or bt.get("created_at"))
                    direction = str(bt.get("direction") or bt.get("type") or "BUY").upper()
                    side = "SELL" if direction in ("PUT", "SELL", "SHORT") else "BUY"

                    merged.append({
                        "id": pid,
                        "pair": pair,
                        "engine": "VIP",
                        "type": side,
                        "result": result,
                        "pnl": round(profit, 2),
                        "pips": 0.0,
                        "rMultiple": round(profit / max(1.0, float(bt.get("amount") or bt.get("invest") or 2.0)), 2),
                        "exitReason": "Blitz Expiry",
                        "closedAt": close_dt.strftime("%m-%d %H:%M:%S") if close_dt else "--:--:--",
                        "closedDate": close_dt.strftime("%Y-%m-%d") if close_dt else "",
                        "_sortTs": close_dt.timestamp() if close_dt else 0.0,
                        "duration": "1m",
                    })
            except Exception as e:
                logger.debug(f"[Telemetry] Blitz history fetch warning: {e}")

        # 3. Fallback to TradeLedger rows if broker history failed
        if not merged and ledger_by_pid:
            for pid, t in ledger_by_pid.items():
                pnl = float(t.get("pnl") or 0.0)
                entry_px = float(t.get("entry_price") or 0.0)
                exit_px = float(t.get("exit_price") or entry_px)
                pair = resolve_symbol_name(str(t.get("symbol") or "UNKNOWN"), symbol_map)
                side = str(t.get("side") or "BUY").upper()
                close_dt = _parse_iso_or_epoch(t.get("close_time_utc"))
                open_dt = _parse_iso_or_epoch(t.get("open_time_utc"))
                merged.append({
                    "id": pid,
                    "pair": pair,
                    "engine": _normalize_engine_code(str(t.get("strategy") or "CFD")),
                    "type": side if side in ("BUY", "SELL") else "BUY",
                    "result": "WIN" if pnl > 0.01 else ("LOSS" if pnl < -0.01 else "BREAKEVEN"),
                    "pnl": round(pnl, 2),
                    "pips": _calculate_pips(pair, entry_px, exit_px, side) if entry_px > 0 else 0.0,
                    "rMultiple": round(pnl / 50.0, 2),
                    "exitReason": str(t.get("exit_reason") or "Closed"),
                    "closedAt": close_dt.strftime("%m-%d %H:%M:%S") if close_dt else "--:--:--",
                    "closedDate": close_dt.strftime("%Y-%m-%d") if close_dt else "",
                    "_sortTs": close_dt.timestamp() if close_dt else 0.0,
                    "duration": _format_duration(open_dt, close_dt),
                })

        merged.sort(key=lambda x: x.get("_sortTs", 0.0), reverse=True)
        for item in merged:
            item.pop("_sortTs", None)

        self._history_cache_trades = merged[:100]
        self._history_cache_ts = now
        return self._history_cache_trades

    def _collect_snapshot_sync(self) -> Dict[str, Any]:
        acct = getattr(self.tg_bot, "account_type", self.account_type) if self.tg_bot else self.account_type
        symbol_map = self._get_symbol_map()

        bal_info = None
        try:
            bal_info = (
                self.forex_mcp.get_training_balance()
                if acct == "training"
                else self.forex_mcp.get_real_balance()
            )
        except Exception as e:
            logger.debug(f"[Telemetry] Balance fetch warning: {e}")

        bid = bal_info.get("balance_id") if isinstance(bal_info, dict) else None
        raw_amount = float(bal_info.get("amount", 0.0)) if isinstance(bal_info, dict) else 0.0
        raw_equity = float(bal_info.get("equity", raw_amount)) if isinstance(bal_info, dict) else 0.0

        ledger_open_map: Dict[str, str] = {}
        try:
            for row in self.ledger.get_open_trades():
                pid = str(row.get("position_id") or "")
                if pid:
                    ledger_open_map[pid] = str(row.get("strategy") or "")
        except Exception:
            pass

        raw_positions = []
        if bid:
            try:
                raw_positions = self.forex_mcp.list_positions(balance_id=bid) or []
            except Exception as e:
                logger.debug(f"[Telemetry] Positions fetch warning: {e}")

        mapped_positions: List[Dict[str, Any]] = []
        for p in raw_positions:
            pid = p.get("position_id") or p.get("id")
            pair = self._resolve_symbol(p, symbol_map)
            side = str(p.get("side") or p.get("type", "BUY")).upper()
            if side == "LONG":
                side = "BUY"
            elif side == "SHORT":
                side = "SELL"

            lots = float(p.get("lots") or p.get("count") or 0.0)
            entry = float(p.get("open_price") or p.get("open_quote") or 0.0)
            current = float(p.get("current_price") or p.get("close_price") or p.get("price") or entry)
            sl = float(p.get("stop_lose_price") or p.get("stop_loss") or p.get("sl") or 0.0)
            tp = float(p.get("take_profit_price") or p.get("take_profit") or p.get("tp") or 0.0)

            pnl_val = None
            for k in ("expected_pnl", "pnl", "profit", "isolated_pnl_net", "net_profit"):
                if p.get(k) is not None:
                    try:
                        pnl_val = float(p.get(k))
                        break
                    except (ValueError, TypeError):
                        pass
            pnl = round(pnl_val or 0.0, 2)

            pips = _calculate_pips(pair, entry, current, side)
            if sl <= 0:
                sl = entry
            if tp <= 0:
                tp = entry

            dir_sign = 1 if side == "BUY" else -1
            risk_dist = abs(entry - sl)
            r_mult = round(((current - entry) * dir_sign) / risk_dist, 2) if (risk_dist > 0 and current > 0) else 0.0

            be_locked = (side == "BUY" and sl >= entry) or (side == "SELL" and sl <= entry and sl > 0)
            stage_level = 2 if be_locked else 1
            stage_label = "Stage 2: Breakeven" if be_locked else "Stage 1: Open"
            engine_code = self._resolve_engine_for_position(pid, pair, ledger_open_map)

            mapped_positions.append({
                "id": str(pid),
                "pair": pair,
                "engine": engine_code,
                "type": side,
                "lot": round(lots, 2),
                "entry": entry,
                "current": current,
                "stopLoss": sl,
                "initialStopLoss": sl,
                "takeProfit": tp,
                "pnl": pnl,
                "pips": pips,
                "rMultiple": r_mult,
                "stage": stage_label,
                "stageLevel": stage_level,
                "openedAt": str(p.get("open_time") or "")[:19] or datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "thesis": f"Live IQ Option {acct.upper()} ({engine_code}) position #{pid}",
            })

        mapped_closed = self._fetch_broker_closed_trades(bid, symbol_map)

        # Build equity curve chronologically ending at live equity
        rev_pts: List[Dict[str, Any]] = []
        running_bal = raw_amount if raw_amount > 0 else 4000.0
        for t in mapped_closed[:30]:
            rev_pts.append({
                "time": t["closedAt"],
                "equity": round(running_bal, 2),
                "pnl": t["pnl"],
                "drawdownPct": 0.0,
            })
            running_bal -= t["pnl"]
        rev_pts.reverse()
        if raw_equity > 0:
            rev_pts.append({
                "time": "Now",
                "equity": round(raw_equity, 2),
                "pnl": round(raw_equity - raw_amount, 2),
                "drawdownPct": 0.0,
            })

        peak = 0.0
        for pt in rev_pts:
            if pt["equity"] > peak:
                peak = pt["equity"]
            pt["drawdownPct"] = round(((peak - pt["equity"]) / peak) * 100, 2) if peak > 0 else 0.0

        return {
            "type": "snapshot",
            "accountType": acct,
            "balance": round(raw_amount, 2),
            "equity": round(raw_equity, 2),
            "strategies": self._build_strategies_payload(mapped_positions),
            "positions": mapped_positions,
            "closedTrades": mapped_closed,
            "equityCurve": rev_pts,
            "logs": list(self.log_handler.buffer),
            "timestamp": int(time.time() * 1000),
        }

    async def _handle_health(self, request: web.Request) -> web.Response:
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
        }
        return web.json_response(
            {
                "status": "ok",
                "service": "BreakingBadV3",
                "mode": "read-only",
                "wsPath": "/ws/live",
                "clients": len(self.clients),
                "tokenConfigured": bool(self._get_expected_token()),
            },
            headers=headers,
        )

    async def _handle_ws(self, request: web.Request) -> web.StreamResponse:
        if not self._get_expected_token():
            return web.json_response(
                {"error": "DASHBOARD_API_TOKEN is not configured on server"},
                status=503,
            )
        if not self._is_authorized(request):
            return web.json_response({"error": "Unauthorized"}, status=401)

        ws = web.WebSocketResponse(heartbeat=25.0)
        await ws.prepare(request)
        self.clients.add(ws)
        logger.info(f"📡 [Telemetry] Read-only dashboard client connected ({len(self.clients)} active)")

        try:
            snapshot = await asyncio.to_thread(self._collect_snapshot_sync)
            await ws.send_json(snapshot)

            async for msg in ws:
                if msg.type == WSMsgType.TEXT and msg.data.strip().lower() == "ping":
                    await ws.send_json({"type": "pong", "timestamp": int(time.time() * 1000)})
                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        except Exception as e:
            logger.debug(f"[Telemetry] Client stream closed: {e}")
        finally:
            self.clients.discard(ws)
            logger.info(f"📡 [Telemetry] Dashboard client disconnected ({len(self.clients)} active)")

        return ws

    async def _broadcast_loop(self):
        while self._running:
            try:
                await asyncio.sleep(3.0)
                if not self.clients:
                    continue

                snapshot = await asyncio.to_thread(self._collect_snapshot_sync)
                payload = json.dumps(snapshot)

                dead: List[web.WebSocketResponse] = []
                for ws in list(self.clients):
                    if ws.closed:
                        dead.append(ws)
                        continue
                    try:
                        await ws.send_str(payload)
                    except Exception:
                        dead.append(ws)
                for ws in dead:
                    self.clients.discard(ws)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[Telemetry] Broadcast loop warning: {e}")
                await asyncio.sleep(3.0)

    async def start(self):
        if not self._handler_attached:
            logging.root.addHandler(self.log_handler)
            self._handler_attached = True

        port = int(os.getenv("PORT", "8080"))
        app = web.Application()
        app.router.add_get("/", self._handle_health)
        app.router.add_get("/health", self._handle_health)
        app.router.add_get("/ws/live", self._handle_ws)

        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", port)
        await site.start()
        self._running = True
        logger.info(f"🌐 [Telemetry] Read-only HTTP/WebSocket server listening on 0.0.0.0:{port} (/health, /ws/live)")

        await self._broadcast_loop()

    async def stop(self):
        self._running = False
        for ws in list(self.clients):
            try:
                await ws.close()
            except Exception:
                pass
        self.clients.clear()
        if self._runner:
            try:
                await self._runner.cleanup()
            except Exception:
                pass
            self._runner = None
        if self._handler_attached:
            logging.root.removeHandler(self.log_handler)
            self._handler_attached = False
        logger.info("Telemetry server stopped.")
