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
    await api._connect()
    
    # Wait for connection
    await asyncio.sleep(3)
    
    asset = "EURUSD" # Standard pair usually has binary
    # If standard is closed, try OTC
    # asset = "EURUSD-OTC" 
    
    # Test cases: 2m (Turbo), 5m (Binary/Turbo edge), 15m (Binary)
    test_expiries = [2, 5, 15] 
    
    for expiry in test_expiries:
        logger.info(f"\n--- Testing Expiry: {expiry}m ---")
        try:
             # Force fallback logic via run_trade OR call execute_binary directly
             # Let's use execute_binary_option_trade directly to control params better
             
            success, result = await api.execute_binary_option_trade(
                asset=asset,
                amount=1,
                direction="call",
                expiry=expiry
            )
            
            if success:
                logger.info(f"✅ Success! Order ID: {result}")
            else:
                logger.error(f"❌ Failed: {result}")
                
        except Exception as e:
            logger.error(f"Error testing {expiry}m: {e}")
            
        await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())
