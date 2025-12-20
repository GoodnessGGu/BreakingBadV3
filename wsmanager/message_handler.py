import json
import logging

logger = logging.getLogger(__name__)

class MessageHandler:
    def __init__(self):
        self.server_time = None
        self.profile_msg = None
        self.balance_data = None
        self.candles = None
        self.underlying_list = None
        self.initialization_data = None
        self._underlying_assests = None
        self.hisory_positions = None
        self.open_positions = {
            'digital_options': {},
            'binary_options': {}
        }
        # Optimization: Event-driven confirmation
        import asyncio
        self.pending_digital_orders = {} # {request_id: asyncio.Future}
        self.pending_digital_orders = {} # {request_id: asyncio.Future}
        self.binary_order_event = asyncio.Event()
        self.binary_outcome_event = asyncio.Event() # NEW: For instant close detection
        
        self.recent_binary_opens = []
        self.position_info = {}

    def handle_message(self, message):
        message_name = message.get('name')
        handlers = {
            'profile': self._handle_profile,
            'candles': self._handle_candles,
            'balances': self._handle_balances,
            'timeSync': self._handle_server_time,
            'underlying-list': self._handle_underlying_list,
            'initialization-data': self._handle_initialization_data,
            'training-balance-reset': self._handle_training_balance_reset,
            "history-positions": self._handle_position_history,
            "digital-option-placed": self._handle_digital_option_placed,
            "position-changed": self._handle_position_changed,
            "option-opened": self._handle_binary_option_opened,
            "option-closed": self._handle_binary_option_closed,
        }
        handler = handlers.get(message_name)
        if handler:
            handler(message)

    def _handle_server_time(self, message):
        self.server_time = message['msg']

    def _handle_profile(self, message):
        self.profile_msg = message
        balances = message['msg']['balances']
        for balance in balances:
            if balance['type'] == 4:
                self.active_balance_id = balance['id']
                break

    def _handle_balances(self, message):
        self.balance_data = message['msg']

    def _handle_training_balance_reset(self, message):
        if message['status'] == 2000:
            logger.info('Demo Account Balance Reset Successfully')
        elif message['status'] == 4001:
            logger.warning(message['msg']['message'])
        else:
            logger.info(message)

    def _handle_initialization_data(self, message):
        self._underlying_assests = message['msg']

    def _handle_candles(self, message):
        self.candles = message['msg']['candles']

    def _handle_underlying_list(self, message):
        if message['msg'].get('type', None) == 'digital-option':
            self._underlying_assests = message['msg']['underlying']
        else:
            self._underlying_assests = message['msg']['items']

    def _handle_position_history(self, message):
        self.hisory_positions = message['msg']['positions']

    def _handle_digital_option_placed(self, message):
        req_id = message["request_id"]
        
        # 1. Update fallback storage
        if message["msg"].get("id") is not None:
            self.open_positions['digital_options'][req_id] = message["msg"].get("id")
        else:
            self.open_positions['digital_options'][req_id] = message["msg"].get("message")
            
        # 2. Trigger Event-Driven Future if waiting
        if req_id in self.pending_digital_orders:
            future = self.pending_digital_orders.pop(req_id)
            if not future.done():
                # Pass the result directly (either ID or error message)
                result = message["msg"].get("id") or message["msg"].get("message")
                future.set_result(result)

    def _handle_position_changed(self, message):
        try:
            if "raw_event" in message["msg"] and "order_ids" in message["msg"]["raw_event"]:
                self.position_info[int(message["msg"]["raw_event"]["order_ids"][0])] = message['msg']
                self._save_data(message['msg'], 'positions')
                self.binary_outcome_event.set() # ✅ Signal update immediately
        except Exception as e:
            logger.warning(f"Error handling position changed: {e}")

    def _handle_binary_option_opened(self, message):
        try:
            msg_data = message.get("msg", {})
            self.recent_binary_opens.append(msg_data)
            # Keep list small
            if len(self.recent_binary_opens) > 20:
                self.recent_binary_opens.pop(0)
            
            # Trigger Event for anyone waiting
            self.binary_order_event.set()
                
            # Legacy/Debug logic (optional, keeping for safety if request_id ever appears)
            if "request_id" in message:
                req_id = message["request_id"]
                option_id = msg_data.get("id") or msg_data.get("option_id")
                if option_id:
                     self.open_positions['binary_options'][req_id] = option_id
        except Exception as e:
            logger.error(f"Error handling binary option opened: {e} | Msg: {message}")

    def _handle_binary_option_closed(self, message):
        try:
            msg = message["msg"]
            # logger.info(f"Binary Option Closed: {msg}")
            
            option_id = msg.get("id") or msg.get("option_id")
            if option_id:
                self.position_info[int(option_id)] = msg
                self._save_data(msg, 'binary_positions')
                self.binary_outcome_event.set() # ✅ Signal trade.py immediately
            else:
                logger.warning(f"Binary option closed without ID: {message}")
        except Exception as e:
            logger.error(f"Error handling binary option closed: {e}")

    # Utility
    def _save_data(self, message, filename):
        with open(f'{filename}.json', 'w') as file:
            json.dump(message, file, indent=4)
