
import asyncio
import logging
import time
from iqclient import IQOptionAPI, run_trade
from strategies import analyze_strategy
from settings import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def bot_loop(api):
    assets = ["EURUSD-OTC", "GBPUSD-OTC", "USDJPY-OTC"]
    timeframe = 60 # 1 minute
    
    logger.info("🚀 Starting Realtime Bot Loop...")
    logger.info(f"Assets: {assets}")
    logger.info(f"Timeframe: {timeframe}s")
    
    while True:
        try:
            if not api.check_connect():
                logger.warning("Bot disconnected. Reconnecting...")
                api._connect()
                await asyncio.sleep(5)
                continue
                
            for asset in assets:
                # 1. Fetch latest candles
                # We need enough history for indicators (EMA200, RSI, etc.)
                # 250-300 is safe.
                candles = api.get_candle_history(asset, 280, timeframe)
                
                if not candles:
                    logger.warning(f"No candles for {asset}")
                    continue
                    
                # 2. Analyze Strategy
                # analyze_strategy expects a list of dicts (candles)
                signal = analyze_strategy(candles)
                
                if signal:
                    logger.info(f"🔔 SIGNAL FOUND: {asset} {signal}")
                    
                    # 3. Execute Trade
                    # Using run_trade from iqclient which handles martingale & logic
                    # Amount from settings or default 1
                    amount = 1 
                    expiry = 1 # 1 minute
                    
                    # Run trade (this might block if not careful, but run_trade is async in iqclient? yes)
                    result = await run_trade(
                        api=api, 
                        asset=asset, 
                        direction=signal.lower(), 
                        expiry=expiry, 
                        amount=amount
                    )
                    
                    logger.info(f"Trade Result: {result}")
            
            # Sleep to prevent API spam and wait for next candle?
            # 1-second interval is fine for checking, but we should probably sync with candle close.
            # For simplicity, just sleep 2s.
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Error in bot loop: {e}")
            await asyncio.sleep(5)

async def main():
    api = IQOptionAPI()
    await api._connect()
    
    # Wait for connection
    while not api.check_connect():
        await asyncio.sleep(1)
        
    await bot_loop(api)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
