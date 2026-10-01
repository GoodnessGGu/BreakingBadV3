"""
forex_mcp_client.py - IQ Option Official Marginal Forex, CFD & Crypto MCP Client

Direct integration with IQ Option's official Model Context Protocol (MCP) streamable-HTTP servers:
- CFD / Commodities (Gold, Silver, Indices, Stocks): https://marginal-cfd.mcp.iqoption.com
- Forex Currencies (EUR/USD, GBP/USD, USD/JPY, etc.): https://marginal-forex.mcp.iqoption.com
- Crypto (Bitcoin, Ethereum, etc.): https://marginal-crypto.mcp.iqoption.com

Features:
- Multi-endpoint smart routing based on asset_id, instrument_id, or symbol ticker.
- Unified session initialization and Mcp-Session-Id header management across endpoints.
- Automatic aggregation of open positions and trade history across all marginal engines.
- Robust error handling and fallback execution for SL/TP stop-level constraints.
"""

import os
import json
import logging
import math
import requests
from typing import Dict, Any, Optional, List, Tuple
from dotenv import load_dotenv

logger = logging.getLogger("IQForexMCP")

FOREX_ENDPOINTS = {
    "cfd": "https://marginal-cfd.mcp.iqoption.com",
    "forex": "https://marginal-forex.mcp.iqoption.com",
    "crypto": "https://marginal-crypto.mcp.iqoption.com"
}

class IQForexMCPClient:
    def __init__(self, token: Optional[str] = None, base_url: str = "https://marginal-cfd.mcp.iqoption.com"):
        load_dotenv()
        self.token = token or os.getenv("IQ_AI_TOKEN") or os.getenv("IQ_MCP_TOKEN")
        self.default_base_url = base_url.rstrip("/")
        self.request_id = 0
        
        # Multi-endpoint sessions
        self.endpoint_urls = dict(FOREX_ENDPOINTS)
        if self.default_base_url not in self.endpoint_urls.values():
            self.endpoint_urls["custom"] = self.default_base_url

        self.sessions: Dict[str, requests.Session] = {k: requests.Session() for k in self.endpoint_urls}
        self.session_ids: Dict[str, Optional[str]] = {k: None for k in self.endpoint_urls}
        self.tools: Dict[str, Dict[str, Any]] = {k: {} for k in self.endpoint_urls}
        self.asset_cache: Dict[str, Any] = {}
        self.instrument_cache: Dict[int, Any] = {}
        
        if self.token:
            self._update_auth_headers()

    @property
    def base_url(self) -> str:
        return self.default_base_url

    @property
    def session(self) -> requests.Session:
        return self.sessions.get("cfd", list(self.sessions.values())[0])

    @property
    def session_id(self) -> Optional[str]:
        return self.session_ids.get("cfd")

    @session_id.setter
    def session_id(self, val: Optional[str]):
        self.session_ids["cfd"] = val

    def _update_auth_headers(self):
        for ep_key, sess in self.sessions.items():
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "User-Agent": "SmartTrailForex/1.0 (Antigravity-AI)"
            }
            if self.session_ids.get(ep_key):
                headers["Mcp-Session-Id"] = self.session_ids[ep_key]
            sess.headers.update(headers)

    def set_token(self, token: str):
        self.token = token.strip()
        self._update_auth_headers()

    def _next_id(self) -> int:
        self.request_id += 1
        return self.request_id

    def resolve_endpoint(self, asset_id: Optional[int] = None, instrument_id: Optional[str] = None, symbol: Optional[str] = None) -> str:
        """Resolve which MCP server endpoint hosts the given asset/instrument/symbol."""
        if instrument_id:
            i = str(instrument_id).lower()
            if i.startswith("mf."):
                return "forex"
            if i.startswith("mcrpt."):
                return "crypto"
            if i.startswith("mcfd."):
                return "cfd"
        if symbol:
            s = str(symbol).upper().replace("/", "").replace("-", "")
            if s in ("BTCUSD", "BTC", "ETHUSD", "ETH", "BCHUSD"):
                return "crypto"
            if s in ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "EURGBP", "EURJPY", "GBPJPY", "EURCAD", "EURAUD", "EURNZD", "EURCHF"):
                return "forex"
            if s in ("XAUUSD", "GOLD", "XAGUSD", "SILVER", "XAU", "XAG"):
                return "cfd"
        if asset_id is not None:
            try:
                aid = int(asset_id)
                if aid in (816, 824, 836, 886, 1979):
                    return "crypto"
                if aid in (1, 2, 4, 5, 6, 99, 105, 108, 212, 946, 951, 955, 1011):
                    return "forex"
                return "cfd"
            except (ValueError, TypeError):
                pass
        return "cfd"

    def rpc_call(self, method: str, params: Optional[Dict[str, Any]] = None, endpoint: str = "cfd", retry_init: bool = True) -> Dict[str, Any]:
        """Execute a JSON-RPC 2.0 call against the resolved streamable-HTTP MCP endpoint."""
        if not self.token:
            raise ValueError("Missing IQ Option AI Integration Token. Please set IQ_AI_TOKEN in .env")

        if endpoint not in self.endpoint_urls:
            endpoint = "cfd"
        url = self.endpoint_urls[endpoint]
        sess = self.sessions[endpoint]

        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params or {}
        }

        try:
            resp = sess.post(url, json=payload, timeout=20)

            # Check for Mcp-Session-Id in response headers
            sess_hdr = resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
            if sess_hdr and sess_hdr != self.session_ids[endpoint]:
                self.session_ids[endpoint] = sess_hdr
                sess.headers["Mcp-Session-Id"] = self.session_ids[endpoint]

            if resp.status_code == 401:
                logger.error(f"❌ 401 Unauthorized on {endpoint}: Invalid or expired IQ Option AI token.")
                return {"error": {"code": 401, "message": "Unauthorized: Invalid or expired token"}}
            elif resp.status_code == 403:
                logger.error(f"❌ 403 Forbidden on {endpoint}: Token lacks permission.")
                return {"error": {"code": 403, "message": "Forbidden: Token lacks permission"}}
            elif resp.status_code in (400, 404, 410) and retry_init and method != "initialize":
                logger.warning(f"⚠️ Forex MCP [{endpoint}] HTTP {resp.status_code} (session invalid/expired). Re-initializing...")
                if self.initialize(endpoint=endpoint):
                    return self.rpc_call(method, params, endpoint=endpoint, retry_init=False)

            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            data = None
            if "application/json" in content_type:
                data = resp.json()
            else:
                lines = resp.text.strip().splitlines()
                for line in lines:
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    try:
                        data = json.loads(line)
                        break
                    except Exception:
                        continue
                if data is None:
                    data = {"result": resp.text}

            # Check if error indicates expired/invalid session
            err = data.get("error") if isinstance(data, dict) else None
            if err and retry_init and ("session" in str(err).lower() or "not found" in str(err).lower()):
                logger.warning(f"⚠️ Forex MCP [{endpoint}] session expired. Re-initializing...")
                if self.initialize(endpoint=endpoint):
                    return self.rpc_call(method, params, endpoint=endpoint, retry_init=False)

            return data

        except requests.RequestException as e:
            if retry_init and method != "initialize":
                logger.warning(f"⚠️ Forex MCP [{endpoint}] request exception: {e}. Re-initializing...")
                if self.initialize(endpoint=endpoint):
                    return self.rpc_call(method, params, endpoint=endpoint, retry_init=False)
            logger.error(f"HTTP error during MCP rpc_call({method}) on {endpoint}: {e}")
            return {"error": {"code": -1, "message": str(e)}}

    def initialize(self, endpoint: Optional[str] = None) -> bool:
        """Initialize MCP sessions with IQ Option servers and exchange session handshakes."""
        targets = [endpoint] if endpoint else list(self.endpoint_urls.keys())
        any_success = False
        for ep in targets:
            sess = self.sessions[ep]
            if "Mcp-Session-Id" in sess.headers:
                del sess.headers["Mcp-Session-Id"]
            self.session_ids[ep] = None

            init_res = self.rpc_call("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "SmartTrailForexBot", "version": "1.0.0"}
            }, endpoint=ep, retry_init=False)

            if init_res and "error" not in init_res and self.session_ids[ep]:
                try:
                    sess.post(self.endpoint_urls[ep], json={
                        "jsonrpc": "2.0",
                        "method": "notifications/initialized"
                    }, timeout=10)
                except Exception as e:
                    logger.debug(f"Error sending notifications/initialized to {ep}: {e}")
                logger.info(f"✅ IQ Option MCP [{ep.upper()}] session initialized! Session ID: {self.session_ids[ep]}")
                self.fetch_tools(endpoint=ep)
                any_success = True
            else:
                logger.warning(f"⚠️ Could not initialize MCP endpoint [{ep.upper()}]: {init_res.get('error') if isinstance(init_res, dict) else init_res}")
        return any_success

    def fetch_tools(self, endpoint: str = "cfd") -> Dict[str, Any]:
        """Query tools/list from server for specific endpoint."""
        res = self.rpc_call("tools/list", {}, endpoint=endpoint)
        result = res.get("result", {})
        tools_list = result.get("tools", [])
        self.tools[endpoint] = {t["name"]: t for t in tools_list}
        return self.tools[endpoint]

    def call_tool(self, name: str, arguments: Dict[str, Any], endpoint: Optional[str] = None) -> Dict[str, Any]:
        """Execute a specific MCP tool and unpack structuredContent or content JSON."""
        if not endpoint:
            endpoint = self.resolve_endpoint(
                asset_id=arguments.get("asset_id"),
                instrument_id=arguments.get("instrument_id")
            )

        res = self.rpc_call("tools/call", {
            "name": name,
            "arguments": arguments
        }, endpoint=endpoint)

        if "error" in res:
            logger.error(f"Tool call error [{name}] on {endpoint}: {res['error']}")
            return {"error": res["error"]}

        result = res.get("result", {})
        if result.get("isError"):
            err_msg = ""
            for c in result.get("content", []):
                if isinstance(c, dict) and "text" in c:
                    err_msg += c["text"] + " "
            logger.error(f"Tool execution returned error [{name}] on {endpoint}: {err_msg.strip()}")
            return {"error": {"message": err_msg.strip()}}

        # Extract structuredContent if present
        if "structuredContent" in result and result["structuredContent"]:
            return result["structuredContent"]

        # Fallback to parsing content[0].text
        content_items = result.get("content", [])
        if content_items and isinstance(content_items, list):
            first = content_items[0]
            if isinstance(first, dict) and "text" in first:
                txt = first["text"]
                try:
                    return json.loads(txt)
                except Exception:
                    return {"text": txt}

        return result

    # ==========================================
    # High-Level Forex, CFD & Crypto API Methods
    # ==========================================

    def list_balances(self, types: str = "ALL") -> List[Dict[str, Any]]:
        """List marginal balances (equity, margin, free_margin, pnl)."""
        res = self.call_tool("list_balances", {"types": types.upper()}, endpoint="cfd")
        if isinstance(res, dict) and "balances" in res:
            return res["balances"]
        return []

    def get_real_balance(self) -> Optional[Dict[str, Any]]:
        """Retrieve real money marginal balance profile."""
        balances = self.list_balances("REAL")
        return balances[0] if balances else None

    def get_training_balance(self) -> Optional[Dict[str, Any]]:
        """Retrieve training practice marginal balance profile."""
        balances = self.list_balances("TRAINING")
        return balances[0] if balances else None

    def list_assets(self, endpoint: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all available tradeable assets across endpoints."""
        target_eps = [endpoint] if endpoint else ["cfd", "forex", "crypto"]
        all_assets = []
        for ep in target_eps:
            res = self.call_tool("list_assets", {}, endpoint=ep)
            if isinstance(res, dict) and "assets" in res:
                for a in res["assets"]:
                    clean_name = a.get("name", "").replace("/", "").upper()
                    self.asset_cache[clean_name] = a
                    self.asset_cache[a.get("name", "").upper()] = a
                    self.asset_cache[str(a.get("asset_id"))] = a
                    all_assets.append(a)
        return all_assets

    def get_asset(self, asset_name_or_id: Any) -> Optional[Dict[str, Any]]:
        """Get asset metadata by name or asset_id."""
        if not self.asset_cache:
            self.list_assets()

        key = str(asset_name_or_id).replace("/", "").upper()
        if key in self.asset_cache:
            return self.asset_cache[key]
        if str(asset_name_or_id) in self.asset_cache:
            return self.asset_cache[str(asset_name_or_id)]
        return None

    def get_instruments(self, asset_id: int) -> Optional[Dict[str, Any]]:
        """Get tradable instruments, lot_size, min_quantity, stop_levels, etc."""
        if asset_id in self.instrument_cache:
            return self.instrument_cache[asset_id]

        ep = self.resolve_endpoint(asset_id=asset_id)
        res = self.call_tool("get_instruments", {"asset_id": asset_id}, endpoint=ep)
        if isinstance(res, dict) and "instruments" in res:
            self.instrument_cache[asset_id] = res
            return res
        return None

    def get_candles(self, asset_id: Any, size: int = 60, count: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Get historical candles from the appropriate server endpoint."""
        if isinstance(asset_id, str):
            if "." in asset_id:
                try:
                    asset_id = int(asset_id.split(".")[-1])
                except ValueError:
                    pass
            elif asset_id.isdigit():
                asset_id = int(asset_id)
        if "period" in kwargs:
            size = kwargs["period"]
        if size == 180:
            size = 120
        ep = self.resolve_endpoint(asset_id=asset_id)
        res = self.call_tool("get_candles", {
            "asset_id": int(asset_id),
            "size": size,
            "count": count
        }, endpoint=ep)
        if isinstance(res, dict) and "candles" in res:
            return res["candles"] or []
        return []

    def calculate_order_size(self, asset_id: int, balance_currency: str = "USD",
                             lots: Optional[float] = None, margin: Optional[float] = None,
                             units: Optional[float] = None, leverage: int = 50) -> Dict[str, Any]:
        """Preview lots, required margin, notional, and current prices."""
        ep = self.resolve_endpoint(asset_id=asset_id)
        args = {
            "asset_id": asset_id,
            "balance_currency": balance_currency.upper(),
            "leverage": leverage
        }
        if lots is not None:
            args["lots"] = lots
        elif margin is not None:
            args["margin"] = margin
        elif units is not None:
            args["units"] = units
        else:
            raise ValueError("Must provide exactly one of: lots, margin, units")

        return self.call_tool("calculate_order_size", args, endpoint=ep)

    def place_market_order(self, side: str, balance_id: int, instrument_id: str, asset_id: int,
                           lots: float, leverage: int = 50, stop_loss: Optional[float] = None,
                           take_profit: Optional[float] = None, is_margin_isolated: bool = True,
                           keep_position_open: bool = False) -> Dict[str, Any]:
        """Place an immediate market order on the appropriate marginal engine."""
        ep = self.resolve_endpoint(asset_id=asset_id, instrument_id=instrument_id)

        # Enforce asset-specific constraints
        if asset_id == 1487:  # Silver (XAGUSD)
            lots = max(10.0, float(lots))
            instrument_id = "mcfd.1487"
        elif asset_id == 816:  # Bitcoin (BTCUSD)
            leverage = min(leverage, 20)
            lots = max(0.001, float(lots))
            instrument_id = "mcrpt.816"
        elif asset_id == 1:   # EURUSD
            instrument_id = "mf.1"
        elif asset_id == 5:   # GBPUSD
            instrument_id = "mf.5"

        active_bal_id = balance_id
        if not active_bal_id:
            try:
                bal = self.get_training_balance() or self.get_real_balance()
                if bal and bal.get("balance_id"):
                    active_bal_id = bal["balance_id"]
            except Exception:
                pass

        args = {
            "side": side.lower(),
            "balance_id": active_bal_id,
            "instrument_id": instrument_id,
            "asset_id": asset_id,
            "lots": lots,
            "leverage": leverage,
            "is_margin_isolated": is_margin_isolated,
            "keep_position_open": keep_position_open
        }
        if stop_loss is not None and stop_loss > 0:
            args["stop_loss"] = round(float(stop_loss), 5)
        if take_profit is not None and take_profit > 0:
            args["take_profit"] = round(float(take_profit), 5)

        logger.info(f"🚀 [MCP {ep.upper()}] Placing {side.upper()} order: Asset={asset_id} ({instrument_id}), Lots={lots}, Lev={leverage}x, SL={stop_loss}, TP={take_profit}")
        res = self.call_tool("place_market_order", args, endpoint=ep)
        if "error" in res or res.get("isError"):
            err_str = str(res.get("error", res))
            if "deadline" in err_str.lower() or "timeout" in err_str.lower():
                logger.warning(f"⚠️ [MCP {ep.upper()}] Order placement encountered deadline/timeout ({err_str}). Retrying once after 1s...")
                time.sleep(1.0)
                res = self.call_tool("place_market_order", args, endpoint=ep)

        if ("error" in res or res.get("isError")) and ("stop_loss" in args or "take_profit" in args):
            err_str = str(res.get("error", res))
            if "not_filled" in err_str or "stop_levels" in err_str:
                logger.warning(f"⚠️ [MCP {ep.upper()}] Market order fill failed due to SL/TP constraints ({err_str}). Retrying immediately without initial SL/TP for guaranteed market fill...")
                clean_args = {k: v for k, v in args.items() if k not in ("stop_loss", "take_profit")}
                retry_res = self.call_tool("place_market_order", clean_args, endpoint=ep)
                return retry_res
        return res

    def _resolve_actual_position_id(self, candidate_id: int) -> int:
        """Resolve a candidate position_id or order_id to an active broker position_id."""
        if not candidate_id:
            return candidate_id
        try:
            positions = self.list_positions()
            for p in positions:
                pid = p.get("position_id") or p.get("id")
                oid = p.get("order_id")
                if pid and int(pid) == int(candidate_id):
                    return int(pid)
                if oid and int(oid) == int(candidate_id):
                    logger.info(f"🔄 [MCP] Resolved Order ID #{candidate_id} -> Position ID #{pid}")
                    return int(pid)
        except Exception as e:
            logger.debug(f"Could not resolve position_id: {e}")
        return candidate_id

    def change_position_stop_loss(self, position_id: int, level: float, balance_id: Optional[int] = None) -> Dict[str, Any]:
        """Move or set the Stop Loss trigger price for an open position across endpoints."""
        actual_pid = self._resolve_actual_position_id(int(position_id))
        logger.info(f"🔄 [MCP] Updating SL on Position #{actual_pid} (input: #{position_id}) to {level:.5f}")
        for ep in ("cfd", "forex", "crypto"):
            try:
                res = self.call_tool("change_position_stop_loss", {
                    "position_id": int(actual_pid),
                    "level": round(float(level), 5)
                }, endpoint=ep)
                if res and not res.get("error") and not res.get("isError"):
                    return res
            except Exception:
                pass
        return {"error": {"message": f"Failed to update SL for position {actual_pid}"}}

    def change_position_take_profit(self, position_id: int, level: float) -> Dict[str, Any]:
        """Move or set the Take Profit trigger price for an open position across endpoints."""
        actual_pid = self._resolve_actual_position_id(int(position_id))
        logger.info(f"🎯 [MCP] Updating TP on Position #{actual_pid} (input: #{position_id}) to {level:.5f}")
        for ep in ("cfd", "forex", "crypto"):
            try:
                res = self.call_tool("change_position_take_profit", {
                    "position_id": int(actual_pid),
                    "level": round(float(level), 5)
                }, endpoint=ep)
                if res and not res.get("error") and not res.get("isError"):
                    return res
            except Exception:
                pass
        return {"error": {"message": f"Failed to update TP for position {actual_pid}"}}

    def list_positions(self, balance_id: Optional[int] = None, skip: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        """List currently open marginal positions across all engines (CFD, Forex, Crypto)."""
        active_bal_id = balance_id
        if not active_bal_id:
            try:
                bal = self.get_training_balance() or self.get_real_balance()
                if bal and bal.get("balance_id"):
                    active_bal_id = bal["balance_id"]
            except Exception:
                pass

        all_positions = []
        seen = set()
        for ep in ("cfd", "forex", "crypto"):
            try:
                args = {"skip": skip, "limit": limit}
                if active_bal_id:
                    args["balance_id"] = active_bal_id
                res = self.call_tool("list_positions", args, endpoint=ep)
                if isinstance(res, dict) and "positions" in res:
                    for p in (res["positions"] or []):
                        pid = p.get("position_id") or p.get("id")
                        if pid and pid not in seen:
                            seen.add(pid)
                            p["_endpoint"] = ep
                            all_positions.append(p)
            except Exception as e:
                logger.debug(f"Error listing positions on {ep}: {e}")
        return all_positions

    def get_orders(self, balance_id: int) -> List[Dict[str, Any]]:
        """List active pending orders across endpoints."""
        all_orders = []
        for ep in ("cfd", "forex", "crypto"):
            try:
                res = self.call_tool("get_orders", {"balance_id": balance_id}, endpoint=ep)
                if isinstance(res, dict) and "orders" in res:
                    all_orders.extend(res["orders"] or [])
            except Exception:
                pass
        return all_orders

    def get_trade_history(self, balance_id: int, skip: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        """List closed marginal positions across all engines."""
        all_history = []
        seen = set()
        for ep in ("cfd", "forex", "crypto"):
            try:
                res = self.call_tool("get_trade_history", {
                    "balance_id": balance_id,
                    "skip": skip,
                    "limit": limit
                }, endpoint=ep)
                if isinstance(res, dict) and "history" in res:
                    for h in (res["history"] or []):
                        hid = h.get("position_id") or h.get("id")
                        if hid and hid not in seen:
                            seen.add(hid)
                            all_history.append(h)
            except Exception:
                pass
        return all_history

    def close_position(self, position_id: int) -> Dict[str, Any]:
        """Close an open marginal position by trying endpoints until accepted."""
        actual_pid = self._resolve_actual_position_id(int(position_id))
        logger.info(f"🔒 [MCP] Closing position #{actual_pid} (input: #{position_id})")
        for ep in ("cfd", "forex", "crypto"):
            try:
                res = self.call_tool("close_position", {"position_id": int(actual_pid)}, endpoint=ep)
                if res and not res.get("error") and not res.get("isError"):
                    return res
            except Exception:
                pass
        return self.call_tool("close_position", {"position_id": int(actual_pid)}, endpoint="cfd")
