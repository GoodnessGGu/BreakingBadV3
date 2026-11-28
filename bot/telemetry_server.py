"""
bot/telemetry_server.py - Strictly Read-Only WebSocket & Health Server for BreakingBad V3

Exposes:
  - GET / and GET /health : Lightweight HTTP health check (JSON)
  - GET /ws/live?token=... : Authenticated read-only WebSocket telemetry stream

Security:
  - Strictly read-only observer: no trade execution, closing, or settings mutation routes.
  - Requires DASHBOARD_API_TOKEN (or WS_API_TOKEN) matched via constant-time comparison.
  - Redacts bot tokens and bearer secrets from all streamed log lines.
  - Only polls broker MCP caches when at least one authenticated client is connected.
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

logger = logging.getLogger("TelemetryServer")

ASSET_ID_TO_SYMBOL = {
    1: "EURUSD",
    5: "GBPUSD",
    6: "USDJPY",
    8: "NZDUSD",
    74: "XAUUSD",
    99: "AUDUSD",
    100: "USDCAD",
    816: "BTCUSD",
    1471: "NAS100",
    1487: "XAGUSD",
}


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
        return {"engine": "ICT", "tag": "TRENDLINE"}
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
        self._pending: deque = deque(maxlen=40)

    def emit(self, record: logging.LogRecord):
        try:
            msg = _redact_secrets(record.getMessage())
            meta = _infer_log_engine_and_tag(record.name, msg)
            self._seq += 1
            entry = {
                "id": f"srv-log-{self._seq}",
                "time": datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%H:%M:%S"),
                "tag": meta["tag"],
                "message": msg[:260],
                "level": _infer_log_level(record, msg),
            }
            if meta["engine"]:
                entry["engine"] = meta["engine"]
            self.buffer.appendleft(entry)
            self._pending.append(entry)
        except Exception:
            pass

    def pop_pending(self) -> List[Dict[str, Any]]:
        items = list(self._pending)
        self._pending.clear()
        return items


def _pip_size_for(symbol: str) -> float:
    s = symbol.upper()
    if "XAU" in s or "XAG" in s:
        return 0.1
    if "JPY" in s:
        return 0.01
    if "BTC" in s or "NAS" in s:
        return 1.0
    return 0.0001


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
        self.ledger = TradeLedger()
        self._runner: Optional[web.AppRunner] = None
        self._running = False
        self._handler_attached = False

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

    def _resolve_symbol(self, raw_pos: Dict[str, Any]) -> str:
        aid = raw_pos.get("asset_id")
        if aid is not None:
            try:
                aid_int = int(aid)
                if aid_int in ASSET_ID_TO_SYMBOL:
                    return ASSET_ID_TO_SYMBOL[aid_int]
            except (ValueError, TypeError):
                pass
        if self.tg_bot and hasattr(self.tg_bot, "resolve_asset_name"):
            try:
                resolved = self.tg_bot.resolve_asset_name(aid)
                if resolved:
                    return str(resolved).replace("/", "").upper()
            except Exception:
                pass
        name = raw_pos.get("symbol") or raw_pos.get("instrument_id") or aid or "UNKNOWN"
        return str(name).replace("/", "").upper()

    def _resolve_engine_for_position(self, pos_id: Any) -> str:
        pid_str = str(pos_id)
        for code, eng in (
            ("ICT", self.ict_engine),
            ("CRT", self.crt_engine),
            ("S&D", self.snd_engine),
            ("ICT", self.trendline_engine),
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

        if self.channel_mgr and hasattr(self.channel_mgr, "copiers"):
            for _, copier in (self.channel_mgr.copiers or {}).items():
                open_pos = getattr(copier, "open_positions", {}) or {}
                for oid, opos in open_pos.items():
                    if str(oid) == pid_str or str(opos.get("position_id")) == pid_str:
                        return "VIP"
        return "BOT"

    def _build_strategies_payload(self) -> List[Dict[str, Any]]:
        ict_active = bool(self.ict_engine and getattr(self.ict_engine, "is_enabled", False))
        snd_active = bool(self.snd_engine and getattr(self.snd_engine, "is_enabled", False))
        crt_active = bool(self.crt_engine and getattr(self.crt_engine, "is_enabled", False))
        vip_active = bool(self.channel_mgr and getattr(self.channel_mgr, "is_running", False))

        def _count_open(eng: Any) -> int:
            if not eng or not hasattr(eng, "active_trades"):
                return 0
            return sum(1 for v in (eng.active_trades or {}).values() if v)

        ict_syms = ", ".join(sorted(getattr(self.ict_engine, "enabled_symbols", []) or []))
        snd_syms = ", ".join(sorted(getattr(self.snd_engine, "enabled_symbols", []) or []))
        crt_syms = ", ".join(sorted(getattr(self.crt_engine, "enabled_symbols", []) or []))
        copier_count = len(getattr(self.channel_mgr, "copiers", {}) or {})

        return [
            {
                "id": "ict",
                "code": "ICT",
                "name": "ICT Engine",
                "status": "ACTIVE" if _count_open(self.ict_engine) > 0 else ("SCANNING" if ict_active else "IDLE"),
                "subModel": f"Smart Money FVG ({ict_syms or 'XAUUSD'})",
                "confidence": 86 if ict_active else 0,
            },
            {
                "id": "snd",
                "code": "S&D",
                "name": "S&D Engine",
                "status": "ACTIVE" if _count_open(self.snd_engine) > 0 else ("SCANNING" if snd_active else "IDLE"),
                "subModel": f"Supply/Demand Imbalances ({snd_syms or 'FX'})",
                "confidence": 81 if snd_active else 0,
            },
            {
                "id": "crt",
                "code": "CRT",
                "name": "CRT Engine",
                "status": "ACTIVE" if _count_open(self.crt_engine) > 0 else ("SCANNING" if crt_active else "IDLE"),
                "subModel": f"Candle Range Theory ({crt_syms or 'EURUSD'})",
                "confidence": 74 if crt_active else 0,
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

    def _collect_snapshot_sync(self) -> Dict[str, Any]:
        acct = getattr(self.tg_bot, "account_type", self.account_type) if self.tg_bot else self.account_type
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

        raw_positions = []
        if bid:
            try:
                raw_positions = self.forex_mcp.list_positions(balance_id=bid) or []
            except Exception as e:
                logger.debug(f"[Telemetry] Positions fetch warning: {e}")

        mapped_positions: List[Dict[str, Any]] = []
        for p in raw_positions:
            pid = p.get("position_id") or p.get("id")
            pair = self._resolve_symbol(p)
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

            pip_sz = _pip_size_for(pair)
            dir_sign = 1 if side == "BUY" else -1
            pips = round(((current - entry) / pip_sz) * dir_sign, 1) if entry > 0 and current > 0 else 0.0

            if sl <= 0:
                sl = entry
            if tp <= 0:
                tp = entry

            risk_dist = abs(entry - sl)
            if risk_dist > 0 and current > 0:
                r_mult = round(((current - entry) * dir_sign) / risk_dist, 2)
            else:
                r_mult = 0.0

            be_locked = (side == "BUY" and sl >= entry) or (side == "SELL" and sl <= entry and sl > 0)
            stage_level = 2 if be_locked else 1
            stage_label = "Stage 2: Breakeven" if be_locked else "Stage 1: Open"

            mapped_positions.append({
                "id": str(pid),
                "pair": pair,
                "engine": self._resolve_engine_for_position(pid),
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
                "thesis": f"Live IQ Option {acct.upper()} position #{pid}",
            })

        # Closed trades & equity curve from TradeLedger
        closed_rows = []
        try:
            closed_rows = self.ledger.get_closed_trades(limit=35, offset=0)
        except Exception as e:
            logger.debug(f"[Telemetry] Ledger fetch warning: {e}")

        mapped_closed: List[Dict[str, Any]] = []
        for t in closed_rows:
            pnl = float(t.get("pnl") or 0.0)
            entry_px = float(t.get("entry_price") or 0.0)
            exit_px = float(t.get("exit_price") or entry_px)
            sym = str(t.get("symbol") or "UNKNOWN").replace("/", "").upper()
            side = str(t.get("side") or "BUY").upper()
            pip_sz = _pip_size_for(sym)
            dir_sign = 1 if side == "BUY" else -1
            pips = round(((exit_px - entry_px) / pip_sz) * dir_sign, 1) if entry_px > 0 else 0.0
            result = "WIN" if pnl > 0.01 else ("LOSS" if pnl < -0.01 else "BREAKEVEN")
            strat_raw = str(t.get("strategy") or "BOT").upper()
            eng = "ICT" if "ICT" in strat_raw else ("S&D" if "SND" in strat_raw else ("CRT" if "CRT" in strat_raw else "VIP"))

            mapped_closed.append({
                "id": str(t.get("position_id") or len(mapped_closed)),
                "pair": sym,
                "engine": eng,
                "type": side if side in ("BUY", "SELL") else "BUY",
                "result": result,
                "pnl": round(pnl, 2),
                "pips": pips,
                "rMultiple": round(pnl / 50.0, 2),
                "exitReason": str(t.get("exit_reason") or "Closed"),
                "closedAt": str(t.get("close_time_utc") or "")[-8:] or "--:--:--",
                "duration": "Live",
            })

        # Build equity curve backwards from current balance using closed trades
        curve: List[Dict[str, Any]] = []
        running_bal = raw_amount if raw_amount > 0 else 10000.0
        rev_pts: List[Dict[str, Any]] = []
        for t in mapped_closed[:25]:
            rev_pts.append({
                "time": t["closedAt"],
                "equity": round(running_bal, 2),
                "pnl": t["pnl"],
                "drawdownPct": 0.0,
            })
            running_bal -= t["pnl"]
        rev_pts.reverse()
        peak = 0.0
        for pt in rev_pts:
            if pt["equity"] > peak:
                peak = pt["equity"]
            pt["drawdownPct"] = round(((peak - pt["equity"]) / peak) * 100, 2) if peak > 0 else 0.0
        if not rev_pts:
            rev_pts.append({
                "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "equity": round(raw_equity or raw_amount, 2),
                "pnl": 0.0,
                "drawdownPct": 0.0,
            })
        curve = rev_pts

        return {
            "type": "snapshot",
            "accountType": acct,
            "balance": round(raw_amount, 2),
            "equity": round(raw_equity, 2),
            "strategies": self._build_strategies_payload(),
            "positions": mapped_positions,
            "closedTrades": mapped_closed,
            "equityCurve": curve,
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
                    self.log_handler.pop_pending()
                    continue

                snapshot = await asyncio.to_thread(self._collect_snapshot_sync)
                # Pop pending incremental logs to avoid re-queuing
                self.log_handler.pop_pending()
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
