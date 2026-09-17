import asyncio
import os
import json
import logging
from dotenv import load_dotenv
from iqclient import IQOptionAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ForexOrderTest")

async def test():
    load_dotenv()
    email = os.getenv("email") or os.getenv("IQ_EMAIL")
    password = os.getenv("password") or os.getenv("IQ_PASSWORD")
    
    client = IQOptionAPI(email=email, password=password)
    await client._connect()
        
    user_balance_id = int(client.account_manager.current_account_id)
    logger.info(f"Balance ID: {user_balance_id}")
    
    # Catch all incoming messages
    original_handle = client.message_handler.handle_message
    def debug_handle_message(message):
        name = message.get("name")
        if name != "timeSync":
            logger.info(f"⚡ INCOMING MSG [{name}]:\n{json.dumps(message, indent=2)}")
        original_handle(message)
    client.message_handler.handle_message = debug_handle_message
    
    # Try sending place-order-temp with instrument_type="marginal-forex" and instrument_id="1"
    order_payload = {
        "name": "place-order-temp",
        "version": "4.0",
        "body": {
            "instrument_type": "marginal-forex",
            "instrument_id": "1",
            "side": "buy",
            "amount": 1.0,
            "leverage": 50,
            "type": "market",
            "limit_price": None,
            "stop_price": None,
            "stop_lose_kind": None,
            "stop_lose_value": None,
            "take_profit_kind": None,
            "take_profit_value": None,
            "use_trail_stop": False,
            "auto_margin_call": False,
            "use_token_for_commission": False,
            "user_balance_id": user_balance_id,
            "client_platform_id": "9"
        }
    }
    
    logger.info("Sending place-order-temp (marginal-forex)...")
    client.websocket.send_message("sendMessage", order_payload)
    await asyncio.sleep(5)
    
    # Also test with instrument_type="forex"
    order_payload2 = dict(order_payload)
    order_payload2["body"] = dict(order_payload["body"])
    order_payload2["body"]["instrument_type"] = "forex"
    order_payload2["body"]["instrument_id"] = "EURUSD"
    logger.info("Sending place-order-temp (forex EURUSD)...")
    client.websocket.send_message("sendMessage", order_payload2)
    await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(test())
