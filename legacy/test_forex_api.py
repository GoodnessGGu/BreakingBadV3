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
    logger.info("Connecting to IQ Option...")
    await client._connect()
        
    logger.info(f"Balance: {client.get_current_account_balance()} (Demo ID: {client.account_manager.current_account_id})")
    
    # 1. Test get_underlying_assests('forex')
    logger.info("Fetching forex underlying assets from market_manager...")
    try:
        forex_assets = client.market_manager.get_underlying_assests('forex')
        logger.info(f"Received {len(forex_assets)} forex assets!")
        if forex_assets:
            logger.info(f"Sample asset data: {json.dumps(forex_assets[0], indent=2)}")
    except Exception as e:
        logger.error(f"Error fetching forex assets: {e}")

    # 2. Test position history for marginal-forex
    logger.info("Testing get_position_history_by_page for marginal-forex...")
    try:
        history = client.account_manager.get_position_history_by_page(["marginal-forex", "marginal-cfd"], limit=10)
        logger.info(f"Position history received: {len(history)} positions")
        if history:
            logger.info(f"Sample position: {json.dumps(history[0], indent=2)}")
    except Exception as e:
        logger.error(f"Error fetching position history: {e}")

    client.disconnect()

if __name__ == "__main__":
    asyncio.run(test())
