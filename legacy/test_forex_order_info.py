import asyncio
import os
import json
import logging
from dotenv import load_dotenv
from iqclient import IQOptionAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ForexTest")

async def test():
    load_dotenv()
    email = os.getenv("email") or os.getenv("IQ_EMAIL")
    password = os.getenv("password") or os.getenv("IQ_PASSWORD")
    
    client = IQOptionAPI(email=email, password=password)
    await client._connect()
        
    order_id = 100966324502
    
    received = []
    def debug_handle_message(message):
        name = message.get("name")
        logger.info(f"Incoming: {name}")
        if name in ["order", "orders", "position-changed"]:
            logger.info(f"Order data: {json.dumps(message, indent=2)}")
        received.append(message)
        
    client.message_handler.handle_message = debug_handle_message
    
    # Try get-order
    client.websocket.send_message("sendMessage", {
        "name": "get-order",
        "version": "1.0",
        "body": {"order_id": order_id}
    })
    
    # Try order.get-order
    client.websocket.send_message("sendMessage", {
        "name": "orders.get-order",
        "version": "1.0",
        "body": {"order_id": order_id}
    })
    
    await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(test())
