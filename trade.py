import asyncio
import time
import logging
import random
from datetime import datetime, timezone
from options_assests import UNDERLYING_ASSESTS
from utilities import get_expiration, get_remaining_secs

logger = logging.getLogger(__name__)


# Custom exceptions for better error categorization
class TradeExecutionError(Exception):
    """Base exception for trade execution errors"""
    pass


class InvalidTradeParametersError(TradeExecutionError):
    """Raised when trade parameters are invalid"""
    pass


class TradeManager:
    """
    Manages IQOption trading operations
    
    Handles trade parameter validation, order execution, confirmation waiting,
    and trade outcome tracking.
    """
    def __init__(self, websocket_manager, message_handler, account_manager, market_manager=None):
        self.ws_manager = websocket_manager
        self.message_handler = message_handler
        self.account_manager = account_manager
        self.market_manager = market_manager

    def get_current_price(self, asset: str) -> float:
        """Fetch the latest price for an asset."""
        if not self.market_manager:
            return None
        try:
            # Fetch 1 candle of 1s to get latest tick
            candles = self.market_manager.get_candle_history(asset, 1, 1)
            if candles:
                return candles[-1]['close']
        except Exception:
            pass
        return None



    def get_asset_id(self, asset_name: str) -> int:
        if asset_name in UNDERLYING_ASSESTS:
            return UNDERLYING_ASSESTS[asset_name]
        raise KeyError(f'{asset_name} not found!')

    # ========== DIGITAL OPTIONS ==========
    async def _execute_digital_option_trade(self, asset:str, amount:float, direction:str, expiry:int=1):
        try:
            direction = direction.lower()
            self._validate_options_trading_parameters(asset, amount, direction, expiry)

            direction_map = {'put': 'P', 'call': 'C'}        
            direction_code = direction_map[direction]

            from random import randint
            request_id = str(randint(0, 100000))

            msg = self._build_options_body(asset, amount, expiry, direction_code)
            
            # Create a future to wait for result
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            self.message_handler.pending_digital_orders[request_id] = future
            
            self.ws_manager.send_message("sendMessage", msg, request_id)

            # Wait for future with timeout
            try:
                result = await asyncio.wait_for(future, timeout=10)
                if isinstance(result, int):
                    expires_in = get_remaining_secs(self.message_handler.server_time, expiry)
                    logger.info(f'Order Executed Successfully, Order ID: {result}, Expires in: {expires_in} Seconds')
                    return True, result
                else:
                    logger.error(f'Order Execution Failed, Reason: !!! {result} !!!')
                    return False, result
                    
            except asyncio.TimeoutError:
                self.message_handler.pending_digital_orders.pop(request_id, None)
                logger.error(f"Order Confirmation timed out after 10 seconds")
                return False, "Order confirmation timed out"
                
        except (InvalidTradeParametersError, TradeExecutionError, KeyError) as e:
            logger.error(f"Trade execution failed: {e}")
            return False, str(e)
        except Exception as e:
            logger.error(f"Unexpected error during trade execution: {e}", exc_info=True)
            return False, f"Unexpected error: {str(e)}"
                
    # async def wait_for_order_confirmation - REMOVED (No longer needed)

    def _build_options_body(self, asset: str, amount: float, expiry: int, direction: str) -> str:
        active_id = str(self.get_asset_id(asset))
        expiration = get_expiration(self.message_handler.server_time, expiry)
        date_formatted = datetime.fromtimestamp(expiration, timezone.utc).strftime("%Y%m%d%H%M")

        instrument_id = f"do{active_id}A{date_formatted[:8]}D{date_formatted[8:]}00T{expiry}M{direction}SPT"

        return {
            "name": "digital-options.place-digital-option",
            "version": "3.0",
            "body": {
                "user_balance_id": int(self.account_manager.current_account_id),
                "instrument_id": str(instrument_id),
                "amount": str(amount),
                "asset_id": int(active_id),
                "instrument_index": 0,
            }
        }
    
    # ========== PARAM VALIDATION ==========
    def _validate_options_trading_parameters(self, asset: str, amount: float, direction: str, expiry: int) -> None:
        if not isinstance(asset, str) or not asset.strip():
            raise InvalidTradeParametersError("Asset name cannot be empty")
        if not isinstance(amount, (int, float)) or amount < 1:
            raise InvalidTradeParametersError(f"Minimum Bet Amount is $1, got: {amount}")
        direction = direction.lower().strip()
        if direction not in ['put', 'call']:
            raise InvalidTradeParametersError(f"Direction must be 'put' or 'call', got: {direction}")
        if not isinstance(expiry, int) or expiry < 1:
            raise InvalidTradeParametersError(f"Expiry must be positive integer, got: {expiry}")
        if not self.account_manager.current_account_id:
            raise TradeExecutionError("No active account available")
            
    # ========== TRADE OUTCOME ==========
    async def get_trade_outcome(self, order_id: int, expiry:int=1, asset_name: str = None, direction: str = None):
        start_time = time.time()
        timeout = get_remaining_secs(self.message_handler.server_time, expiry)

        while time.time() - start_time < timeout + 30:
            order_data = self.message_handler.position_info.get(order_id, {})
            if order_data and order_data.get("status") == "closed":
                pnl = order_data.get('pnl', 0)
                result_type = "WIN" if pnl > 0 else "LOSS"
                logger.info(f"Trade closed - Order ID: {order_id}, Result: {result_type}, PnL: ${pnl:.2f}")
                return True, pnl
            await asyncio.sleep(.5)

        logger.warning(f"Digital Trade Outcome Timed Out (ID: {order_id})")
        
        # Shadow Verification for Digital
        if self.market_manager and asset_name and direction:
             current_price = self.get_current_price(asset_name)
             if current_price:
                 order_data = self.message_handler.position_info.get(order_id, {})
                 # For Digital, 'msg' -> 'price' might be open price or 'open_underlying_price'
                 open_price = float(order_data.get('open_underlying_price', 0))
                 if not open_price:
                      # Try to find it in msg
                      open_price = float(order_data.get('msg', {}).get('price', 0))

                 if open_price:
                     is_call = direction.lower() == 'call'
                     calculated_win = (current_price > open_price) if is_call else (current_price < open_price)
                     logger.warning(f"🕵️ Timeout Shadow Verification (Digital): {asset_name} {direction} | Open: {open_price} | Close: {current_price} | Win: {calculated_win}")
                     if calculated_win:
                         return True, 0.85 
                     else:
                         return True, -1.0

        return False, None

    # ========== BINARY OPTIONS ==========
    async def _execute_binary_option_trade(self, asset:str, amount:float, direction:str, expiry:int=1):
        """
        Executes a binary/turbo option trade.
        """
        try:
            direction = direction.lower()
            self._validate_options_trading_parameters(asset, amount, direction, expiry)

            # Determine option type (turbo vs binary) based on expiry
            # usually <= 5m is turbo (3), > 5m is binary (1)
            option_type_id = 3 if expiry <= 5 else 1  
            
            from random import randint
            request_id = str(randint(0, 100000))

            start_time = time.time() # Capture time before sending
            msg = self._build_binary_body(asset, amount, expiry, direction, option_type_id)
            self.ws_manager.send_message("sendMessage", msg, request_id)

            active_id = self.get_asset_id(asset)
            return await self.wait_for_binary_order_confirmation(active_id, amount, direction, start_time, expiry)
        
        except (InvalidTradeParametersError, TradeExecutionError, KeyError) as e:
            logger.error(f"Binary Trade execution failed: {e}")
            return False, str(e)
        except Exception as e:
            logger.error(f"Unexpected error during binary trade execution: {e}", exc_info=True)
            return False, f"Unexpected error: {str(e)}"

    def _build_binary_body(self, asset: str, amount: float, expiry: int, direction: str, option_type_id: int) -> dict:
        active_id = self.get_asset_id(asset)
        expiration = get_expiration(self.message_handler.server_time, expiry)
        
        return {
            "name": "binary-options.open-option",
            "version": "1.0",
            "body": {
                "user_balance_id": int(self.account_manager.current_account_id),
                "active_id": int(active_id),
                "option_type_id": option_type_id,
                "direction": direction, # 'call' or 'put'
                "expired": int(expiration),
                "price": float(amount),
                "profit_percent": 0 # Usually 0 or queried, server handles it
            }
        }

    async def wait_for_binary_order_confirmation(self, active_id:int, amount:float, direction:str, start_time:float, expiry:int, timeout:int=10):
        # Poll recent_binary_opens for the matching trade
        # Matching criteria: active_id, close amount, direction, and timestamp >= start_time
        
        end_time = time.time() + timeout
        
        while time.time() < end_time:
            # 1. Check existing list first
            current_list = list(self.message_handler.recent_binary_opens)
            for order in current_list:
                 created_at_ms = order.get("created_at") or order.get("open_time_millisecond", 0)
                 created_at = created_at_ms / 1000.0
                 
                 if created_at >= (start_time - 5): 
                     try:
                         oa_id = int(order.get("active_id"))
                         o_amt = float(order.get("amount"))
                         o_dir = order.get("direction")
                         
                         if oa_id == active_id and abs(o_amt - amount) < 0.01 and o_dir == direction:
                             result_id = order.get("id") or order.get("option_id")
                             expires_in = get_remaining_secs(self.message_handler.server_time, expiry)
                             logger.info(f'Binary Order Executed, ID: {result_id}, Expires in: {expires_in}s')
                             return True, result_id
                     except Exception:
                         continue

            # 2. Wait for NEW event (instead of sleep)
            # Calculate remaining time
            remaining = end_time - time.time()
            if remaining <= 0:
                break
                
            try:
                self.message_handler.binary_order_event.clear()
                await asyncio.wait_for(self.message_handler.binary_order_event.wait(), timeout=min(remaining, 0.5))
            except asyncio.TimeoutError:
                pass # Just loop check again
            
        return False, "Binary order confirmation timed out (No match found)"
    
    async def get_binary_trade_outcome(self, order_id: int, expiry: int = 1, asset_name: str = None, direction: str = None):
        start_time = time.time()
        # Increase timeout buffer for OTC/delayed server responses
        timeout = get_remaining_secs(self.message_handler.server_time, expiry) + 30

        while time.time() - start_time < timeout:
            order_data = self.message_handler.position_info.get(order_id, {})
            
            # Check if Closed or Time Elapsed significantly (>5s past expiry)
            # This 'expected_end' needs to be derived. 
            # We don't have exact expiry time variable here, but we know 'expiry' duration.
            # Assuming 'start_time' matches roughly expiry start? No.
            # We can use server_time logic but simplified:
            # If (current time > start + expiry + 5) AND status != closed -> Force Check
            
            is_closed = order_data and (order_data.get("status") == "closed" or order_data.get("close_time"))
            force_check = False
            
            # Approximate elapsed time since check started
            # Note: 'expiry' arg is duration (e.g. 60). 
            # If we called this *after* trade placed, we might wait up to 60s?
            # Usually this method is called immediately after placement.
            # So 'start_time + expiry' is roughly the close time.
            time_since_start = time.time() - start_time
            if not is_closed and time_since_start > (expiry + random.randint(1, 5)):
                force_check = True
                logger.warning(f"⚠️ Trade {order_id} server timeout (>5s delay). Forcing Shadow Verification.")
            
            if is_closed or force_check:
                # Check outcome
                result = order_data.get('win')
                
                invest = float(order_data.get('amount', 0))
                profit_amount = float(order_data.get('profit_amount', 0) or 0) 
                
                pnl = 0.0
                
                # Robust status check
                is_win = result in ['win', 'won'] or (result is None and profit_amount > invest)
                
                # Shadow Verification logic...
                if (result is None or force_check) and self.market_manager and asset_name and direction:
                    try:
                        current_price = self.get_current_price(asset_name)
                        open_price = float(order_data.get('value', 0) or order_data.get('open_price', 0)) 
                        if not open_price and force_check:
                             # If we forced check, maybe we didn't get open price from order_data yet?
                             # Try to fetch from msg if available
                             open_price = float(order_data.get('msg', {}).get('price', 0))

                        if current_price and open_price:
                            is_call = direction.lower() == 'call'
                            calculated_win = (current_price > open_price) if is_call else (current_price < open_price)
                            
                            logger.info(f"🕵️ Shadow Verification: {asset_name} {direction} | Open: {open_price} | Close (Tick): {current_price} | Calc Win: {calculated_win}")
                            
                            if calculated_win:
                                is_win = True
                                try:
                                    payout = self.market_manager.get_binary_payout(asset_name)
                                    pnl = invest * (payout / 100.0)
                                except:
                                    pnl = invest * 0.85 
                            else:
                                is_win = False
                                pnl = -invest
                                
                            # If forced check, break loop with this result
                            return {'result': 'WIN' if is_win else 'LOSS', 'profit': pnl, 'win': is_win}
                            
                    except Exception as e:
                        logger.warning(f"Shadow Verification failed: {e}")

                if is_win:
                    if profit_amount >= invest:
                         pnl = profit_amount - invest
                    pnl = -invest

                # Log for debugging
                logger.info(f"Binary Outcome: {result} | Invest: {invest} | Return: {profit_amount} | PnL: {pnl}")
                return True, pnl

            # Calculate remaining wait time
            remaining = (start_time + timeout) - time.time()
            if remaining <= 0:
                break

            # Wait for NEXT event (or timeout) instead of sleep
            try:
                self.message_handler.binary_outcome_event.clear()
                # fast reaction: if event triggers, we loop and check 'position_info' immediately
                await asyncio.wait_for(self.message_handler.binary_outcome_event.wait(), timeout=min(remaining, 1.0))
            except asyncio.TimeoutError:
                pass # Loop again to check timeout or poll
            
        logger.warning(f"Binary Trade Outcome Timed Out (ID: {order_id})")
        
        # Last Resort Shadow Verification on Timeout
        if self.market_manager and asset_name and direction:
             current_price = self.get_current_price(asset_name)
             if current_price:
                 # We lack open_price if order_data is missing completely...
                 # But usually order_data exists but Status is not Closed.
                 # Let's hope order_data is populated via 'buy_complete' at least.
                 order_data = self.message_handler.position_info.get(order_id, {})
                 open_price = float(order_data.get('value', 0) or order_data.get('open_price', 0))
                 if open_price:
                     is_call = direction.lower() == 'call'
                     calculated_win = (current_price > open_price) if is_call else (current_price < open_price)
                     logger.warning(f"🕵️ Timeout Shadow Verification: {asset_name} {direction} | Open: {open_price} | Close: {current_price} | Win: {calculated_win}")
                     if calculated_win:
                         return True, 0.85 # Return arbitrary +PnL to signal Win
                     else:
                         return True, -1.0 # Return arbitrary -PnL to signal Loss
        
        return False, 0.0
