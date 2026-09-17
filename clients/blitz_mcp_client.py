"""
clients/blitz_mcp_client.py - IQ Option Official Blitz Options MCP Client

Direct integration with IQ Option's official Blitz Options MCP streamable-HTTP server
at https://blitz-options.mcp.iqoption.com using Bearer AI token authentication.
"""

import os
import json
import logging
import time
import requests
from typing import Dict, Any, Optional, List, Tuple
from dotenv import load_dotenv

logger = logging.getLogger("IQBlitzMCP")

class IQBlitzMCPClient:
    def __init__(self, token: Optional[str] = None, base_url: str = "https://blitz-options.mcp.iqoption.com"):
        load_dotenv()
        self.token = token or os.getenv("IQ_AI_TOKEN") or os.getenv("IQ_MCP_TOKEN")
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.request_id = 0
        self.session_id: Optional[str] = None
        self.tools: Dict[str, Any] = {}
        self.asset_cache: Dict[str, Any] = {}
        self.last_asset_fetch = 0

        if self.token:
            self._update_auth_header()

    def _update_auth_header(self):
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "User-Agent": "SmartTrailBlitz/1.0 (Antigravity-AI)"
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        self.session.headers.update(headers)

    def set_token(self, token: str):
        self.token = token.strip()
        self._update_auth_header()

    def _next_id(self) -> int:
        self.request_id += 1
        return self.request_id

    def rpc_call(self, method: str, params: Optional[Dict[str, Any]] = None, retry_init: bool = True) -> Dict[str, Any]:
        """Execute JSON-RPC 2.0 call against streamable-HTTP MCP endpoint."""
        if not self.token:
            raise ValueError("Missing IQ Option AI Token. Please set IQ_AI_TOKEN in .env")

        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params or {}
        }

        try:
            resp = self.session.post(self.base_url, json=payload, timeout=20)
            sess_hdr = resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
            if sess_hdr and sess_hdr != self.session_id:
                self.session_id = sess_hdr
                self.session.headers["Mcp-Session-Id"] = self.session_id

            if resp.status_code == 401:
                logger.error("❌ 401 Unauthorized: Invalid or expired IQ Option AI token.")
                return {"error": {"code": 401, "message": "Unauthorized: Invalid or expired token"}}
            elif resp.status_code == 403:
                logger.error("❌ 403 Forbidden: Token lacks Blitz Options permission.")
                return {"error": {"code": 403, "message": "Forbidden: Token lacks Blitz Options permission"}}

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

            err = data.get("error") if isinstance(data, dict) else None
            if err and retry_init and "session" in str(err).lower():
                logger.warning("⚠️ MCP session expired or uninitialized. Re-initializing...")
                if self.initialize():
                    return self.rpc_call(method, params, retry_init=False)

            return data

        except requests.RequestException as e:
            logger.error(f"HTTP error during Blitz MCP rpc_call({method}): {e}")
            return {"error": {"code": -1, "message": str(e)}}

    def initialize(self) -> bool:
        """Initialize MCP session with server."""
        if "Mcp-Session-Id" in self.session.headers:
            del self.session.headers["Mcp-Session-Id"]
        self.session_id = None

        init_res = self.rpc_call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "BlitzOptionsBot", "version": "1.0.0"}
        }, retry_init=False)

        if not init_res or "error" in init_res:
            logger.error(f"Blitz initialization failed: {init_res.get('error') if init_res else 'No response'}")
            return False

        if not self.session_id:
            logger.error("No Mcp-Session-Id returned during initialize.")
            return False

        try:
            self.session.post(self.base_url, json={
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }, timeout=10)
        except Exception as e:
            logger.warning(f"Error sending notifications/initialized: {e}")

        logger.info(f"✅ IQ Option Blitz MCP session initialized! Session ID: {self.session_id}")
        self.fetch_tools()
        return True

    def fetch_tools(self) -> Dict[str, Any]:
        res = self.rpc_call("tools/list", {})
        result = res.get("result", {})
        tools_list = result.get("tools", [])
        self.tools = {t["name"]: t for t in tools_list}
        return self.tools

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        res = self.rpc_call("tools/call", {"name": name, "arguments": arguments})
        if "error" in res:
            logger.error(f"Blitz Tool error [{name}]: {res['error']}")
            return {"error": res["error"]}

        result = res.get("result", {})
        if result.get("isError"):
            err_msg = ""
            for c in result.get("content", []):
                if isinstance(c, dict) and "text" in c:
                    err_msg += c["text"] + " "
            logger.error(f"Blitz Tool execution error [{name}]: {err_msg.strip()}")
            return {"error": {"message": err_msg.strip()}}

        content_items = result.get("content", [])
        if content_items and isinstance(content_items, list):
            first = content_items[0]
            if isinstance(first, dict) and "text" in first:
                try:
                    return json.loads(first["text"])
                except Exception:
                    return {"text": first["text"]}

        if "structuredContent" in result:
            return result["structuredContent"]
        return result

    # ==========================================
    # High-Level Blitz Options API
    # ==========================================

    def list_balances(self, types: str = "ALL") -> List[Dict[str, Any]]:
        res = self.call_tool("list_balances", {"types": types.upper()})
        if isinstance(res, dict) and "balances" in res:
            return res["balances"]
        return []

    def get_training_balance(self) -> Optional[Dict[str, Any]]:
        balances = self.list_balances("TRAINING")
        for b in balances:
            if b.get("type") == "training":
                return b
        for b in self.list_balances("ALL"):
            if b.get("type") == "training":
                return b
        return None

    def get_real_balance(self) -> Optional[Dict[str, Any]]:
        balances = self.list_balances("NORMAL")
        for b in balances:
            if b.get("type") == "regular":
                return b
        return None

    def list_assets(self, only_enabled: bool = False) -> List[Dict[str, Any]]:
        """List all blitz options assets."""
        now = time.time()
        if self.asset_cache and (now - self.last_asset_fetch < 60) and not only_enabled:
            return list(self.asset_cache.values())

        res = self.call_tool("list_assets", {"only_enabled": only_enabled})
        assets = []
        if isinstance(res, dict):
            assets = res.get("assets", [])
        elif isinstance(res, list):
            assets = res

        self.asset_cache.clear()
        for a in assets:
            name = a.get("name", "")
            clean = name.replace("/", "").replace(" ", "").replace("(", "").replace(")", "").replace("-", "").upper()
            self.asset_cache[clean] = a
            self.asset_cache[name.upper()] = a
            self.asset_cache[str(a.get("asset_id"))] = a
        self.last_asset_fetch = now
        return assets

    def find_asset(self, pair_str: str) -> Optional[Dict[str, Any]]:
        """
        Smart asset matcher for Polycarp pairs.
        e.g. 'AUD/JPY (OTC)', 'AUDJPY-OTC', 'AUD/JPY', 'AUDJPY'
        """
        if not self.asset_cache or (time.time() - self.last_asset_fetch > 60):
            self.list_assets(only_enabled=False)

        clean_query = pair_str.replace("/", "").replace(" ", "").replace("(", "").replace(")", "").replace("-", "").upper()
        if clean_query in self.asset_cache:
            return self.asset_cache[clean_query]

        is_otc = "OTC" in pair_str.upper()
        # Search substrings
        for key, a in self.asset_cache.items():
            aname = a.get("name", "").upper()
            if is_otc and "OTC" in aname:
                # Compare base/quote e.g. AUDJPY in AUD/JPY (OTC)
                clean_name = aname.replace("/", "").replace(" ", "").replace("(", "").replace(")", "").replace("-", "")
                if clean_query in clean_name or clean_name in clean_query:
                    return a
            elif not is_otc and "OTC" not in aname:
                clean_name = aname.replace("/", "").replace(" ", "").replace("(", "").replace(")", "").replace("-", "")
                if clean_query in clean_name or clean_name in clean_query:
                    return a

        # Fallback to general search
        for key, a in self.asset_cache.items():
            clean_name = a.get("name", "").replace("/", "").replace(" ", "").replace("(", "").replace(")", "").replace("-", "").upper()
            if clean_query in clean_name:
                return a

        return None

    def place_trade(self, balance_id: int, asset_id: int, direction: str,
                    amount: float, profit_percent: int, expiration_size: int = 300) -> Dict[str, Any]:
        """
        Open a new blitz-options position.
        direction: 'call' or 'put'
        expiration_size: duration in seconds (e.g. 300 for 5m)
        """
        args = {
            "balance_id": int(balance_id),
            "asset_id": int(asset_id),
            "direction": direction.lower(),
            "amount": float(amount),
            "profit_percent": int(profit_percent),
            "expiration_size": int(expiration_size)
        }
        logger.info(f"⚡ [Blitz MCP] Placing {direction.upper()} trade: Asset={asset_id}, Amount=${amount}, Payout={profit_percent}%, Exp={expiration_size}s")
        return self.call_tool("place_trade", args)

    def list_positions(self, balance_id: int) -> List[Dict[str, Any]]:
        res = self.call_tool("list_positions", {"balance_id": int(balance_id)})
        if isinstance(res, dict) and "positions" in res:
            return res["positions"]
        elif isinstance(res, list):
            return res
        return []

    def get_trade_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        res = self.call_tool("get_trade_history", {"limit": limit})
        if isinstance(res, dict) and "history" in res:
            return res["history"]
        elif isinstance(res, list):
            return res
        return []

    def sell_position(self, position_id: int) -> Dict[str, Any]:
        return self.call_tool("sell_position", {"position_id": int(position_id)})
