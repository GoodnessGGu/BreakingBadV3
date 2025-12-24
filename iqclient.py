#iqclient.py
import sys
import time
import logging
import requests
import asyncio
from settings import *
from typing import Optional, List

from trade import TradeManager
from markets import MarketManager
from accounts import AccountManager
from wsmanager.iqwebsocket import WebSocketManager
from wsmanager.message_handler import MessageHandler

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IQOptionAPI:
    """
    Main API class for IQOption automated trading.

    Provides a unified interface for account management, market data,
    and trade execution through websocket connections.
    """
    def __init__(self, email=None, password=None, account_type=None):
        """
        Initialize the IQOption API client.

        Args:
            email (str, optional): Login email. Defaults to settings.EMAIL
            password (str, optional): Login password. Defaults to settings.PASSWORD
            account_type (str, optional): Account type. Defaults to settings.DEFAULT_ACCOUNT_TYPE
        """
        # Prefer args → else environment → else crash
        self.email = email or os.getenv("IQ_EMAIL")
        self.password = password or os.getenv("IQ_PASSWORD")
        self.account_mode = account_type or os.getenv("IQ_ACCOUNT_TYPE", "practice")

        # Validate required credentials
        if not self.email or not self.password:
            logger.error("❌ Email and password are required! Check environment variables IQ_EMAIL and IQ_PASSWORD.")
            sys.exit(1)
        # self.email = email or EMAIL
        # self.password = password or PASSWORD
        # self.account_mode = account_type or DEFAULT_ACCOUNT_TYPE

        # Initialize HTTP session for login requests
        self.session = requests.Session()
        self._connected = False

        # Initialize core managers
        self.message_handler = MessageHandler()
        self.websocket = WebSocketManager(self.message_handler)
        self.account_manager = AccountManager(self.websocket, self.message_handler)
        self.market_manager = MarketManager(self.websocket, self.message_handler)
        self.trade_manager = TradeManager(self.websocket, self.message_handler, self.account_manager, self.market_manager)
        logger.info('ALGO BOT initialized successfully')

    def check_connect(self):
        """Check if the API session is still active and socket is alive."""
        if self._connected:
            # Verify actual socket state
            if self.websocket and self.websocket.websocket and self.websocket.websocket.sock:
                return self.websocket.websocket.sock.connected
            else:
                 # Flag is true but socket is missing/dead
                 self._connected = False
                 return False
        return False

    async def ensure_connect(self):
        """Checks connection and reconnects if necessary."""
        if not self.check_connect():
            logger.warning("⚠️ Connection check failed. Reconnecting...")
            self._connected = False # Force reset
            try:
                await self._connect()
                logger.info("✅ Reconnected successfully.")
                return True
            except Exception as e:
                logger.error(f"❌ Reconnection failed: {e}")
                return False
        return True


    def _login(self):
        """
        Authenticate with IQOption using email/password.

        Returns:
            bool: True if login successful, None otherwise
        """

        # Validate required credentials
        if not all([ self.email, self.password]):
            raise ValueError("Email and password are required!")

        if self._connected:
            logger.warning('Already connected to iqoption')
            return

        try:
            # Send login request
            response = self.session.post(url=LOGIN_URL,
                data={'identifier': self.email,'password': self.password})
            response.raise_for_status()

            # Check if session ID was received (login success indicator)
            if self.get_session_id():
                logger.info(f'Successfully logged into an account - SSID: {self.get_session_id()}')
                return True
        except Exception as e:
            logger.warning(e)


    def _logout(self, data=None):
        """
        Log out from IQOption and close session.

        Args:
            data (dict, optional): Additional logout data
        """
        if self.session.post(url=LOGOUT_URL, data=data).status_code == 200:
            self._connected = False
            logger.info(f'Logged out Successfully')

    def get_session_id(self):
        """
        Get the current session ID (SSID) from cookies.

        Returns:
            str: Session ID if available, None otherwise
        """
        return self.session.cookies.get('ssid')

    async def _connect(self):
        """
        Establish full connection: login + websocket + authentication.

        Sets up the complete connection pipeline including websocket
        authentication and account initialization.
        """
        if await asyncio.to_thread(self._login):
            # Start websocket connection
            self.websocket.start_websocket()

            # Authenticate websocket using session ID
            # Retry sending SSID a few times if socket is flaky
            for _ in range(3):
                try:
                     self.websocket.send_message('ssid', self.get_session_id())
                     break
                except Exception as e:
                     logger.warning(f"Failed to send SSID ({e}), retrying...")
                     await asyncio.sleep(0.5)

            ## Wait for profile confirmation (indicates successful auth)
            while self.message_handler.profile_msg is None:
                await asyncio.sleep(.1)

            # Set default account and mark as connected
            self.account_manager.set_default_account()
            self._connected = True
        else:
            raise ConnectionError("Login failed. Check credentials.")

    # Expose manager methods for convenience
    def get_current_account_balance(self):
        """
        Get the balance of the currently active account.

        Returns:
            float: Current account balance
        """
        self._ensure_connected()
        return self.account_manager.get_active_account_balance()

    def refill_demo_account(self, amount=10000):
        """
        Refill demo account with specified amount.

        Args:
            amount (int): Amount to add to demo account. Defaults to 10000

        Returns:
            bool: True if refill successful
        """
        self._ensure_connected()
        return self.account_manager.refill_demo_balance(amount)

    def get_tournament_accounts(self):
        """
        Retrieve list of available tournament accounts.

        Returns:
            list: Available tournament accounts
        """
        self._ensure_connected()
        return self.account_manager.get_tournament_accounts()

    def switch_account(self, account_type:str):
        """
        Switch to a different account type (demo/real/tournament).

        Args:
            account_type (str): Target account type

        Returns:
            bool: True if switch successful, False if already on target account
        """
        self._ensure_connected()
        if account_type.lower() == self.account_manager.current_account_type:
            logger.warning(f'Already on {account_type.lower()} account. No switch needed.')
            return False  # or True, depending on how you want to handle this
        return self.account_manager.switch_account(account_type)

    # Market Data Methods
    def get_candle_history(self, asset_name='EURUSD-op', count=50, timeframe=60, end_time=None):
        """
        Retrieve historical candlestick data for an asset.

        Args:
            asset_name (str): Asset symbol. Defaults to 'EURUSD-op'
            count (int): Number of candles to retrieve. Defaults to 50
            timeframe (int): Timeframe in seconds. Defaults to 60
            end_time (int): Optional timestamp to fetch up to.

        Returns:
            list: Historical candle data
        """
        self._ensure_connected()
        return self.market_manager.get_candle_history(asset_name, count, timeframe, end_time=end_time)

    def save_candles_to_csv(self, candles_data=None, filename='candles'):
        """
        Export candlestick data to CSV file.

        Args:
            candles_data (list, optional): Candle data to export
            filename (str): Output filename. Defaults to 'candles'

        Returns:
            bool: True if save successful
        """
        return self.market_manager.save_candles_to_csv(candles_data, filename)

    def _ensure_connected(self):
        """
        Verify that the bot is connected before executing operations.

        Raises:
            Exception: If bot is not connected
        """
        if not self._connected:
            raise Exception("Bot is not connected. Call connect() first.")

    def get_position_history_by_time(self, instrument_type: List[str],
                                    start_time: Optional[str] = None,
                                    end_time: Optional[str] = None):
        """
        Retrieve position history within a specific time range.

        Args:
            instrument_type (List[str]): Types of instruments to include
            start_time (str, optional): Start time filter
            end_time (str, optional): End time filter

        Returns:
            list: Position history within specified time range
        """
        self._ensure_connected()
        return self.account_manager.get_position_history_by_time(instrument_type, start_time=start_time, end_time=end_time)

    def get_position_history_by_page(self, instrument_type: List[str],
                                    limit: int = 300,
                                    offset: int = 0):
        """
        Retrieve paginated position history.

        Args:
            instrument_type (List[str]): Types of instruments to include
            limit (int): Maximum records per page. Defaults to 300
            offset (int): Number of records to skip. Defaults to 0

        Returns:
            list: Paginated position history
        """
        self._ensure_connected()
        return self.account_manager.get_position_history_by_page(instrument_type, limit=limit, offset=offset)

    # Trade methods (digital kept as-is)
    async def execute_digital_option_trade(self, asset: str, amount: int, direction: str,
                                    expiry: Optional[int] = 1):
        """
        Execute a digital options trade.

        Args:
            asset (str): Asset symbol to trade
            amount (int): Trade amount
            direction (str): Trade direction ('call' or 'put')
            expiry (int, optional): Expiry time in minutes. Defaults to 1

        Returns:
            dict: Trade execution result with order ID
        """
        self._ensure_connected()
        return await self.trade_manager._execute_digital_option_trade(asset, amount, direction, expiry=expiry)

    async def get_trade_outcome(self, order_id: int ,expiry:int, asset_name: str = None, direction: str = None):
        """
        Get the outcome of a completed trade.

        Args:
            order_id (int): ID of the trade order
            expiry (int): Expiry time in minutes
            asset_name (str): Asset name for shadow verification
            direction (str): Trade direction for shadow verification

        Returns:
            dict: Trade outcome (win/loss/refund) and payout details
        """
        self._ensure_connected()
        return await self.trade_manager.get_trade_outcome(order_id, expiry=expiry, asset_name=asset_name, direction=direction)

    # ---- New binary option wrappers ----
    async def execute_binary_option_trade(self, asset: str, amount: int, direction: str,
                                    expiry: Optional[int] = 1):
        """
        Execute a binary options trade.

        Args:
            asset (str): Asset symbol to trade
            amount (int): Trade amount
            direction (str): Trade direction ('call' or 'put')
            expiry (int, optional): Expiry time in minutes. Defaults to 1

        Returns:
            tuple: (success: bool, order_id_or_error)
        """
        self._ensure_connected()
        return await self.trade_manager._execute_binary_option_trade(asset, amount, direction, expiry=expiry)

    async def get_binary_trade_outcome(self, order_id: int, expiry: int = 1, asset_name: str = None, direction: str = None):
        """
        Get the outcome of a binary options trade.

        Args:
            order_id (int): Order ID of the binary trade
            expiry (int): Expiry time in minutes
            asset_name (str): Asset name for shadow verification
            direction (str): Trade direction for shadow verification

        Returns:
            tuple: (success: bool, pnl: float or None)
        """
        self._ensure_connected()
        return await self.trade_manager.get_binary_trade_outcome(order_id, expiry=expiry, asset_name=asset_name, direction=direction)

    async def get_open_positions(self):
        """
        Retrieve a list of all open positions.

        Returns:
            list: A list of dictionaries, each representing an open position.
        """
        self._ensure_connected()
        open_positions = []
        for position in self.message_handler.position_info.values():
            if position.get("status") != "closed":
                raw_event = position.get("raw_event", {})
                open_positions.append({
                    "asset": raw_event.get("instrument_underlying"),
                    "direction": raw_event.get("instrument_dir"),
                    "amount": raw_event.get("buy_amount"),
                })
        return open_positions


from settings import config

# Global set to track active trades to prevent overlapping signals
ACTIVE_TRADES = set()

async def run_trade(api, asset, direction, expiry, amount, max_gales=None, notification_callback=None):
    """
    Executes a trade (digital only) and handles up to a configurable number of martingale attempts.
    """
    # Use config if max_gales is not explicitly provided
    if max_gales is None:
        max_gales = config.max_martingale_gales

    # Check for suppression first
    trade_key = (asset, direction)
    if config.suppress_overlapping_signals and trade_key in ACTIVE_TRADES:
        msg = f"🚫 Trade suppressed: {asset} {direction.upper()} is already active."
        logger.warning(msg)
        return {
            "asset": asset,
            "direction": direction,
            "expiry": expiry,
            "result": "SUPPRESSED",
            "gales": 0,
            "profit": 0.0
        }

    if config.paused:
        logger.info(f"🚫 Trade skipped (bot paused): {asset} {direction.upper()}")
        return {
            "asset": asset,
            "direction": direction,
            "expiry": expiry,
            "result": "SKIPPED",
            "gales": 0,
            "profit": 0.0
        }

    ACTIVE_TRADES.add(trade_key)
    try:
        current_amount = amount
        total_pnl = 0.0  # Track total PnL across all attempts
        
        # Track which type worked to avoid retrying failed types (e.g. Digital on OTC)
        # Track which type worked to avoid retrying failed types (e.g. Digital on OTC)
        # Optimization: Start with configured preference (BINARY/DIGITAL/AUTO)
        preferred_type = "binary" if config.preferred_trading_type == "BINARY" else "digital" 

        for gale in range(max_gales + 1):
            trade_type = preferred_type
            success = False
            result_data = None
            
            # Attempt 1: Try Preferred Type
            if trade_type == "digital":
                success, result_data = await api.execute_digital_option_trade(asset, current_amount, direction, expiry=expiry)
            else:
                success, result_data = await api.execute_binary_option_trade(asset, current_amount, direction, expiry=expiry)
            
            # Fallback Logic: If Digital failed, try Binary
            if not success and trade_type == "digital":
                logger.warning(f"⚠️ Digital trade failed: {result_data}. Switching to Binary/Turbo option...")
                trade_type = "binary"
                success, result_data = await api.execute_binary_option_trade(asset, current_amount, direction, expiry=expiry)
                
                # If Binary worked, make it the preferred type for next gales
                if success:
                    preferred_type = "binary"
            
            if not success:
                error_msg = str(result_data)
                logger.error(f"❌ Failed to place trade on {asset} (Digital & Binary): {error_msg}")
                return {
                    "asset": asset,
                    "direction": direction,
                    "expiry": expiry,
                    "result": "ERROR",
                    "error_message": error_msg,
                    "gales": gale,
                    "profit": total_pnl
                }

            order_id = result_data
            logger.info(f"🎯 Placed trade: {asset} {direction.upper()} ${current_amount} ({expiry}m expiry)")

            pnl_ok, pnl = False, None
            if trade_type == "digital":
                 pnl_ok, pnl = await api.get_trade_outcome(order_id, expiry=expiry, asset_name=asset, direction=direction)
            else:
                 pnl_ok, pnl = await api.get_binary_trade_outcome(order_id, expiry=expiry, asset_name=asset, direction=direction)

            balance = api.get_current_account_balance()

            # Accumulate PnL (pnl is negative on loss, positive on win)
            if pnl is not None:
                total_pnl += pnl

            if pnl_ok and pnl > 0:
                logger.info(f"✅ WIN on {asset} | Profit: ${pnl:.2f} | Net PnL: ${total_pnl:.2f} | Balance: ${balance:.2f}")
                if notification_callback:
                    await notification_callback(f"✅ WIN on {asset} | Net PnL: ${total_pnl:.2f}")
                return {
                    "asset": asset,
                    "direction": direction,
                    "expiry": expiry,
                    "result": "WIN",
                    "gales": gale,
                    "profit": total_pnl
                }
            else:
                logger.warning(f"⚠️ LOSS on {asset} (Gale {gale}) | PnL: {pnl} | Net PnL: ${total_pnl:.2f}")
                
                if gale < max_gales:
                    next_amount = current_amount * config.martingale_multiplier
                    msg = f"⚠️ LOSS on {asset} (Gale {gale}). Martingale to Gale {gale+1}: ${next_amount:.2f}"
                    logger.info(msg)
                    if notification_callback:
                        await notification_callback(msg)
                    current_amount = next_amount
                else:
                    if notification_callback:
                        await notification_callback(f"💀 LOSS on {asset} after {max_gales} gales. Net PnL: ${total_pnl:.2f}")

        logger.error(f"💀 Lost all attempts ({max_gales} gales) on {asset}")
        return {
            "asset": asset,
            "direction": direction,
            "expiry": expiry,
            "result": "LOSS",
            "gales": max_gales,
            "profit": total_pnl
        }
    finally:
        ACTIVE_TRADES.discard(trade_key)
