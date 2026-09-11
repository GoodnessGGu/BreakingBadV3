"""
forex_mcp_client.py - IQ Option Official Marginal Forex MCP Client

Direct integration with IQ Option's official Model Context Protocol (MCP) streamable-HTTP server
at https://marginal-forex.mcp.iqoption.com using Bearer AI token authentication.

Features:
- Robust session initialization with Mcp-Session-Id header management.
- Transparent structured JSON response parsing.
- Dynamic balance & margin queries (Practice/Training vs Real).
- Instrument metadata & leverage profiles inspection.
- Candlestick history fetching directly via MCP.
- Mathematical risk-based lot sizing (1% equity risk).
- Market order execution with Stop Loss & Take Profit.
- Dynamic Stop Loss adjustment for trailing stops.
- Open positions and closed trade history tracking.
"""

import os
import json
import logging
import math
import requests
from typing import Dict, Any, Optional, List, Tuple
from dotenv import load_dotenv

logger = logging.getLogger("IQForexMCP")

class IQForexMCPClient:
    def __init__(self, token: Optional[str] = None, base_url: str = "https://marginal-forex.mcp.iqoption.com"):
        load_dotenv()
        self.token = token or os.getenv("IQ_AI_TOKEN") or os.getenv("IQ_MCP_TOKEN")
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.request_id = 0
        self.session_id: Optional[str] = None
        self.tools: Dict[str, Any] = {}
        self.asset_cache: Dict[str, Any] = {}
        self.instrument_cache: Dict[int, Any] = {}
        
        if self.token:
            self._update_auth_header()

    def _update_auth_header(self):
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "User-Agent": "SmartTrailForex/1.0 (Antigravity-AI)"
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
        """Execute a JSON-RPC 2.0 call against the streamable-HTTP MCP endpoint."""
        if not self.token:
            raise ValueError("Missing IQ Option AI Integration Token. Please set IQ_AI_TOKEN in .env")
            
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params or {}
        }
        
        try:
            resp = self.session.post(self.base_url, json=payload, timeout=20)
            
            # Check for Mcp-Session-Id in response headers
            sess_hdr = resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
            if sess_hdr and sess_hdr != self.session_id:
                self.session_id = sess_hdr
                self.session.headers["Mcp-Session-Id"] = self.session_id
                
            if resp.status_code == 401:
                logger.error("❌ 401 Unauthorized: Invalid or expired IQ Option AI token.")
                return {"error": {"code": 401, "message": "Unauthorized: Invalid or expired token"}}
            elif resp.status_code == 403:
                logger.error("❌ 403 Forbidden: Token lacks Margin Forex trading permission.")
                return {"error": {"code": 403, "message": "Forbidden: Token lacks Margin Forex permission"}}
                
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
            if err and retry_init and "session" in str(err).lower():
                logger.warning("⚠️ MCP session expired or uninitialized. Re-initializing...")
                if self.initialize():
                    return self.rpc_call(method, params, retry_init=False)

            return data
                
        except requests.RequestException as e:
            logger.error(f"HTTP error during MCP rpc_call({method}): {e}")
            return {"error": {"code": -1, "message": str(e)}}

    def initialize(self) -> bool:
        """Initialize MCP session with IQ Option server and exchange session handshake."""
        # Clear existing session ID for fresh handshake
        if "Mcp-Session-Id" in self.session.headers:
            del self.session.headers["Mcp-Session-Id"]
        self.session_id = None

        init_res = self.rpc_call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "SmartTrailForexBot", "version": "1.0.0"}
        }, retry_init=False)
        
        if not init_res or "error" in init_res:
            logger.error(f"Initialization failed: {init_res.get('error') if init_res else 'No response'}")
            return False
            
        if not self.session_id:
            logger.error("No Mcp-Session-Id returned by server during initialize.")
            return False

        # Send notifications/initialized notification
        try:
            self.session.post(self.base_url, json={
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }, timeout=10)
        except Exception as e:
            logger.warning(f"Error sending notifications/initialized: {e}")
            
        logger.info(f"✅ IQ Option MCP session initialized successfully! Session ID: {self.session_id}")
        self.fetch_tools()
        return True

    def fetch_tools(self) -> Dict[str, Any]:
        """Query tools/list from server to dynamically inspect available tools."""
        res = self.rpc_call("tools/list", {})
        result = res.get("result", {})
        tools_list = result.get("tools", [])
        self.tools = {t["name"]: t for t in tools_list}
        logger.info(f"Available MCP Tools ({len(self.tools)}): {list(self.tools.keys())}")
        return self.tools

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a specific MCP tool and unpack structuredContent or content JSON.
        """
        res = self.rpc_call("tools/call", {
            "name": name,
            "arguments": arguments
        })
        
        if "error" in res:
            logger.error(f"Tool call error [{name}]: {res['error']}")
            return {"error": res["error"]}
            
        result = res.get("result", {})
        if result.get("isError"):
            err_msg = ""
            for c in result.get("content", []):
                if isinstance(c, dict) and "text" in c:
                    err_msg += c["text"] + " "
            logger.error(f"Tool execution returned error [{name}]: {err_msg.strip()}")
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
    # High-Level Forex API Methods
    # ==========================================

    def list_balances(self, types: str = "ALL") -> List[Dict[str, Any]]:
        """List marginal-forex balances (equity, margin, free_margin, pnl)."""
        res = self.call_tool("list_balances", {"types": types.upper()})
        if isinstance(res, dict) and "balances" in res:
            return res["balances"]
        return []

    def get_training_balance(self) -> Optional[Dict[str, Any]]:
        """Get the practice/training balance details."""
        balances = self.list_balances("TRAINING")
        for b in balances:
            if b.get("type") == "training":
                return b
        # Fallback to ALL
        for b in self.list_balances("ALL"):
            if b.get("type") == "training":
                return b
        return None

    def get_real_balance(self) -> Optional[Dict[str, Any]]:
        """Get the real/regular balance details."""
        balances = self.list_balances("NORMAL")
        for b in balances:
            if b.get("type") == "regular":
                return b
        return None

    def list_assets(self) -> List[Dict[str, Any]]:
        """List all tradable marginal-forex assets."""
        if self.asset_cache:
            return list(self.asset_cache.values())
            
        res = self.call_tool("list_assets", {})
        if isinstance(res, dict) and "assets" in res:
            for a in res["assets"]:
                # Key by ticker without slash e.g. EURUSD and with slash EUR/USD
                clean_name = a.get("name", "").replace("/", "").upper()
                self.asset_cache[clean_name] = a
                self.asset_cache[a.get("name", "").upper()] = a
                self.asset_cache[str(a.get("asset_id"))] = a
            return res["assets"]
        return []

    def get_asset(self, asset_name_or_id: Any) -> Optional[Dict[str, Any]]:
        """Get asset metadata by name (e.g. 'EURUSD', 'EUR/USD') or asset_id (1)."""
        if not self.asset_cache:
            self.list_assets()
            
        key = str(asset_name_or_id).replace("/", "").upper()
        if key in self.asset_cache:
            return self.asset_cache[key]
        if str(asset_name_or_id) in self.asset_cache:
            return self.asset_cache[str(asset_name_or_id)]
        return None

    def get_instruments(self, asset_id: int) -> Optional[Dict[str, Any]]:
        """
        Get tradable instruments, lot_size, min_quantity, quantity_step, stop_levels, and leverage profiles.
        """
        if asset_id in self.instrument_cache:
            return self.instrument_cache[asset_id]
            
        res = self.call_tool("get_instruments", {"asset_id": asset_id})
        if isinstance(res, dict) and "instruments" in res:
            self.instrument_cache[asset_id] = res
            return res
        return None

    def get_candles(self, asset_id: int, size: int = 60, count: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical candles directly from MCP server.
        Candle size in seconds: 60 (1m), 180 (3m - if supported or resampled), 300 (5m).
        """
        res = self.call_tool("get_candles", {
            "asset_id": asset_id,
            "size": size,
            "count": count
        })
        if isinstance(res, dict) and "candles" in res:
            return res["candles"] or []
        return []

    def calculate_order_size(self, asset_id: int, balance_currency: str = "USD",
                             lots: Optional[float] = None, margin: Optional[float] = None,
                             units: Optional[float] = None, leverage: int = 50) -> Dict[str, Any]:
        """
        Preview lots, required margin, notional, and current bid/ask prices.
        Exactly one of lots, margin, or units must be provided.
        """
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
            
        return self.call_tool("calculate_order_size", args)

    def calculate_lot_size(self, asset_id: int, entry_price: float, sl_price: float,
                           risk_usd: float, balance_currency: str = "USD",
                           leverage: int = 50, free_margin: float = 100.0) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the exact lot size mathematically based on account risk in USD.
        Ensures the loss if SL is hit is bounded by risk_usd.
        Respects min_quantity, quantity_step, and available free margin.
        """
        inst_info = self.get_instruments(asset_id)
        if not inst_info or not inst_info.get("instruments"):
            logger.error(f"Cannot get instruments for asset_id {asset_id}")
            return 0.0, {}
            
        inst = inst_info["instruments"][0]
        lot_size = inst.get("lot_size", 100000)
        min_qty = inst.get("min_quantity", 0.001)
        qty_step = inst.get("quantity_step", 0.001)
        
        # Risk distance in quote currency
        risk_dist = abs(entry_price - sl_price)
        if risk_dist <= 0:
            return 0.0, {}
            
        # Preview 1 min-lot to get the exact quote-to-balance exchange conversion
        preview = self.calculate_order_size(asset_id=asset_id, balance_currency=balance_currency,
                                            lots=min_qty, leverage=leverage)
        if "error" in preview:
            logger.warning(f"Preview failed for asset {asset_id}: {preview['error']}")
            return min_qty, {}
            
        # In Forex: Loss = lots * lot_size * risk_dist (for USD quote currency)
        # For non-USD quote currency, calculate conversion factor from preview
        notional_per_lot = (preview.get("notional", entry_price * lot_size * min_qty) / min_qty)
        loss_per_lot = (lot_size * risk_dist) if "USD" in balance_currency else (risk_dist * notional_per_lot / entry_price)
        
        if loss_per_lot <= 0:
            calculated_lots = min_qty
        else:
            raw_lots = risk_usd / loss_per_lot
            # Step down to nearest quantity_step
            steps = math.floor(raw_lots / qty_step)
            calculated_lots = max(min_qty, steps * qty_step)
            
        # Double check required margin
        margin_check = self.calculate_order_size(asset_id=asset_id, balance_currency=balance_currency,
                                                lots=calculated_lots, leverage=leverage)
        req_margin = margin_check.get("margin", 0.0)
        while req_margin > (free_margin * 0.5) and calculated_lots > min_qty:
            calculated_lots = round(calculated_lots - qty_step, 6)
            margin_check = self.calculate_order_size(asset_id=asset_id, balance_currency=balance_currency,
                                                    lots=calculated_lots, leverage=leverage)
            req_margin = margin_check.get("margin", 0.0)
            
        return calculated_lots, margin_check

    def place_market_order(self, side: str, balance_id: int, instrument_id: str, asset_id: int,
                           lots: float, leverage: int = 50, stop_loss: Optional[float] = None,
                           take_profit: Optional[float] = None, is_margin_isolated: bool = True,
                           keep_position_open: bool = False) -> Dict[str, Any]:
        """
        Place an immediate market order on IQ Option Marginal Forex engine.
        """
        args = {
            "side": side.lower(),
            "balance_id": balance_id,
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

        logger.info(f"🚀 [MCP Forex] Placing {side.upper()} order: Asset={asset_id} ({instrument_id}), Lots={lots}, Lev={leverage}x, SL={stop_loss}, TP={take_profit}")
        return self.call_tool("place_market_order", args)

    def change_position_stop_loss(self, position_id: int, level: float) -> Dict[str, Any]:
        """
        Move or set the Stop Loss trigger price for an open position.
        Level 0 cancels the existing stop-loss.
        """
        logger.info(f"🔄 [MCP Forex] Updating SL on Position #{position_id} to {level:.5f}")
        return self.call_tool("change_position_stop_loss", {
            "position_id": int(position_id),
            "level": round(float(level), 5)
        })

    def change_position_take_profit(self, position_id: int, level: float) -> Dict[str, Any]:
        """
        Move or set the Take Profit trigger price for an open position.
        Level 0 cancels the existing take-profit.
        """
        logger.info(f"🎯 [MCP Forex] Updating TP on Position #{position_id} to {level:.5f}")
        return self.call_tool("change_position_take_profit", {
            "position_id": int(position_id),
            "level": round(float(level), 5)
        })

    def list_positions(self, balance_id: int, skip: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        """List currently open marginal-forex positions for a balance."""
        res = self.call_tool("list_positions", {
            "balance_id": balance_id,
            "skip": skip,
            "limit": limit
        })
        if isinstance(res, dict) and "positions" in res:
            return res["positions"] or []
        return []

    def get_orders(self, balance_id: int) -> List[Dict[str, Any]]:
        """List active pending orders for a balance."""
        res = self.call_tool("get_orders", {"balance_id": balance_id})
        if isinstance(res, dict) and "orders" in res:
            return res["orders"] or []
        return []

    def get_trade_history(self, balance_id: int, skip: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        """List closed marginal-forex positions (trade history) with realized PnL and close reasons."""
        res = self.call_tool("get_trade_history", {
            "balance_id": balance_id,
            "skip": skip,
            "limit": limit
        })
        if isinstance(res, dict) and "history" in res:
            return res["history"] or []
        return []

    def close_position(self, position_id: int) -> Dict[str, Any]:
        """Close an open marginal-forex position by position_id."""
        logger.info(f"🔒 [MCP Forex] Closing position #{position_id}")
        return self.call_tool("close_position", {"position_id": int(position_id)})
