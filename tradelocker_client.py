"""
tradelocker_client.py - TradeLocker API Client for BreakingBad V3
Provides native cloud execution for Prop Firms (FundedVerse, etc.) on TradeLocker.
Supports full trade lifecycle:
  - Authentication (JWT via REST)
  - Account state & balance querying
  - Symbol lookup & real-time quotes (Forex, Metals, Indices, Crypto)
  - Dual-Ticket Order Placement (Leg 1: TP1, Leg 2: Runner TP2)
  - Instant Stop Loss Modification (Breakeven Snapping)
  - Position Closing & Listing
"""

import os
import time
import logging
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("TradeLockerClient")

try:
    from tradelocker import TLAPI
    TRADELOCKER_AVAILABLE = True
except ImportError:
    TRADELOCKER_AVAILABLE = False
    logger.error("tradelocker package not installed. Run: pip install tradelocker")

class TradeLockerClient:
    def __init__(
        self,
        environment: Optional[str] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
        server: Optional[str] = None
    ):
        self.environment = environment or os.getenv("TRADELOCKER_ENV", "https://demo.tradelocker.com")
        self.email = email or os.getenv("TRADELOCKER_EMAIL")
        self.password = password or os.getenv("TRADELOCKER_PASSWORD")
        self.server = server or os.getenv("TRADELOCKER_SERVER", "demo.tradelocker.com")
        self.access_token = os.getenv("TRADELOCKER_ACCESS_TOKEN")
        self.refresh_token = os.getenv("TRADELOCKER_REFRESH_TOKEN")
        
        self.api: Optional[Any] = None
        self.is_connected: bool = False
        self._instrument_cache: Dict[str, int] = {}

    def initialize(self) -> bool:
        """Connect and authenticate with TradeLocker."""
        if not TRADELOCKER_AVAILABLE:
            logger.error("❌ TradeLocker library is missing.")
            return False

        try:
            if self.access_token and self.refresh_token:
                logger.info(f"🔌 Connecting to TradeLocker using Token Authentication ({self.environment})...")
                self.api = TLAPI(
                    environment=self.environment,
                    access_token=self.access_token,
                    refresh_token=self.refresh_token
                )
            elif self.email and self.password and self.server:
                logger.info(f"🔌 Connecting to TradeLocker using Credentials ({self.environment} | Server: {self.server})...")
                self.api = TLAPI(
                    environment=self.environment,
                    username=self.email,
                    password=self.password,
                    server=self.server
                )
            else:
                logger.error("❌ Missing TradeLocker credentials. Provide either (EMAIL + PASSWORD + SERVER) or (ACCESS_TOKEN + REFRESH_TOKEN).")
                return False

            self.is_connected = True
            logger.info("✅ TradeLocker connected & authenticated successfully.")
            return True
        except Exception as e:
            logger.error(f"❌ TradeLocker authentication failed: {e}")
            self.is_connected = False
            return False

    def get_account_state(self) -> Optional[Dict[str, Any]]:
        """Fetch current account balance, equity, and margin."""
        if not self.is_connected or not self.api:
            return None
        try:
            state = self.api.get_account_state()
            return state
        except Exception as e:
            logger.error(f"Failed to fetch account state: {e}")
            return None

    def get_instrument_id(self, symbol: str) -> Optional[int]:
        """Resolve a symbol name (e.g., 'BTCUSD', 'XAUUSD', 'NAS100') to instrument_id."""
        if not self.is_connected or not self.api:
            return None
            
        sym_clean = symbol.upper().replace("/", "").replace(" ", "")
        if sym_clean in self._instrument_cache:
            return self._instrument_cache[sym_clean]

        try:
            # 1. Direct symbol name lookup
            inst_id = self.api.get_instrument_id_from_symbol_name(sym_clean)
            if inst_id:
                self._instrument_cache[sym_clean] = inst_id
                return inst_id

            # 2. Fallback search through all instruments
            all_inst = self.api.get_all_instruments()
            if hasattr(all_inst, "itertuples"):
                for row in all_inst.itertuples():
                    name = getattr(row, "name", "") or getattr(row, "tradableInstrumentId", "")
                    if sym_clean in str(name).upper().replace("/", ""):
                        inst_id = getattr(row, "id", None) or getattr(row, "instrumentId", None)
                        if inst_id:
                            self._instrument_cache[sym_clean] = int(inst_id)
                            return int(inst_id)
            return None
        except Exception as e:
            logger.error(f"Failed to resolve instrument ID for {symbol}: {e}")
            return None

    def get_latest_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """Fetch latest bid & ask price."""
        inst_id = self.get_instrument_id(symbol)
        if not inst_id or not self.api:
            return None
        try:
            bid = self.api.get_latest_bid_price(inst_id)
            ask = self.api.get_latest_asking_price(inst_id)
            return {"bid": float(bid) if bid else 0.0, "ask": float(ask) if ask else 0.0}
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return None

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[int]:
        """Place a single market order with stop loss and take profit."""
        inst_id = self.get_instrument_id(symbol)
        if not inst_id or not self.api:
            logger.error(f"Cannot place order: symbol {symbol} not found.")
            return None

        side_clean = side.lower()
        if side_clean not in ("buy", "sell"):
            logger.error(f"Invalid side: {side}")
            return None

        try:
            logger.info(f"📤 Placing {side_clean.upper()} market order on {symbol}: Qty={quantity}, SL={stop_loss}, TP={take_profit}")
            order_id = self.api.create_order(
                instrument_id=inst_id,
                quantity=float(quantity),
                side=side_clean,
                type_="market",
                stop_loss=float(stop_loss) if stop_loss else None,
                stop_loss_type="absolute" if stop_loss else None,
                take_profit=float(take_profit) if take_profit else None,
                take_profit_type="absolute" if take_profit else None
            )
            logger.info(f"✅ Order placed successfully! Order ID: {order_id}")
            return order_id
        except Exception as e:
            logger.error(f"❌ Failed to place order on {symbol}: {e}")
            return None

    def place_dual_ticket_order(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        min_qty: float = 0.01
    ) -> Dict[str, Optional[int]]:
        """
        Dual-Ticket Order Execution:
          - Leg 1 (50%): Take Profit at TP1 (+1.0R guaranteed banking).
          - Leg 2 (50%): Take Profit at TP2 (+2.5R runner).
        """
        half_qty = max(min_qty, round(total_quantity / 2.0, 2))
        logger.info(f"🚀 Executing Dual-Ticket Setup on {symbol}: Total={total_quantity} | Leg 1={half_qty}, Leg 2={half_qty}")

        # Leg 1: Banks profit at TP1
        order_leg1 = self.place_market_order(
            symbol=symbol,
            side=side,
            quantity=half_qty,
            stop_loss=stop_loss,
            take_profit=tp1
        )

        # Leg 2: Runner to TP2
        order_leg2 = self.place_market_order(
            symbol=symbol,
            side=side,
            quantity=half_qty,
            stop_loss=stop_loss,
            take_profit=tp2
        )

        return {"leg1_order_id": order_leg1, "leg2_order_id": order_leg2}

    def modify_position_sl(self, position_id: int, new_sl: float) -> bool:
        """Modify an active position's Stop Loss (e.g. Move SL to Breakeven)."""
        if not self.is_connected or not self.api:
            return False
        try:
            logger.info(f"🔒 Moving SL on Position #{position_id} to Breakeven ({new_sl})...")
            success = self.api.modify_position(
                position_id=position_id,
                modification_params={
                    "stopLoss": float(new_sl),
                    "stopLossType": "absolute"
                }
            )
            if success:
                logger.info(f"✅ Position #{position_id} SL updated to {new_sl} successfully.")
            return bool(success)
        except Exception as e:
            logger.error(f"❌ Failed to modify SL for position #{position_id}: {e}")
            return False

    def list_positions(self) -> List[Any]:
        """Fetch all currently open positions."""
        if not self.is_connected or not self.api:
            return []
        try:
            positions = self.api.get_all_positions()
            return positions if positions is not None else []
        except Exception as e:
            logger.error(f"Failed to list positions: {e}")
            return []

    def close_position(self, position_id: int) -> bool:
        """Close an open position immediately at market price."""
        if not self.is_connected or not self.api:
            return False
        try:
            logger.info(f"🛑 Closing position #{position_id}...")
            res = self.api.close_position(position_id=position_id)
            logger.info(f"✅ Position #{position_id} closed: {res}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to close position #{position_id}: {e}")
            return False
