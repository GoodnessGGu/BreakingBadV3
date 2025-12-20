import asyncio
import logging
import os
from iqclient import IQOptionAPI, run_trade
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    email = os.getenv("IQ_EMAIL")
    password = os.getenv("IQ_PASSWORD")
    
    if not email or not password:
        logger.error("Please set IQ_EMAIL and IQ_PASSWORD in .env file")
        return

    logger.info("Initializing API...")
    api = IQOptionAPI(email=email, password=password, account_type="PRACTICE")

    try:
        logger.info("Connecting...")
        await api._connect()
        logger.info("Connected!")

        # Balance check
        balance = api.get_current_account_balance()
        logger.info(f"Current Balance: ${balance}")

        # Execute Trade
        asset = "EURUSD-OTC"
        direction = "CALL"
        expiry = 1
        amount = 1

        logger.info(f"Attempting BINARY/TURBO trade: {asset} {direction} ${amount} {expiry}m")
        
        # Explicitly testing the binary option method
        success, result_data = await api.execute_binary_option_trade(
            asset=asset,
            amount=amount,
            direction=direction,
            expiry=expiry
        )

        if success:
            order_id = result_data
            logger.info(f"Binary Trade Placed! Order ID: {order_id}")
            logger.info("Waiting for outcome...")
            pnl_ok, pnl = await api.get_binary_trade_outcome(order_id, expiry=expiry)
            logger.info(f"Trade Outcome: pnl_ok={pnl_ok}, PnL=${pnl}")
        else:
            logger.error(f"Binary Trade Failed: {result_data}")

        # result = await run_trade(
        #     api=api,
        #     asset=asset,
        #     direction=direction, 
        #     expiry=expiry, 
        #     amount=amount, 
        #     max_gales=0 # No martingale for this test
        # )

        # logger.info(f"Trade Result: {result}")

    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
