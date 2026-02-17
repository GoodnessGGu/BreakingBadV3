import asyncio
import sys
import os
import logging

# Add parent dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from iqclient import IQOptionAPI
from ml_utils import calculate_orderflow_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyOrderflow")

async def test_orderflow():
    api = IQOptionAPI()
    try:
        logger.info("Connecting to IQ Option...")
        await api._connect()
        logger.info("Connected.")
        
        asset = "XAUUSD"
        logger.info(f"Subscribing to orderflow for {asset}...")
        api.subscribe_orderflow(asset)
        
        logger.info("Waiting for data (30 seconds)...")
        active_id = api.market_manager.get_marginal_asset_id(asset)
        
        for _ in range(30):
            await asyncio.sleep(1)
            orderbook = api.message_handler.orderbook.get(active_id)
            if orderbook:
                metrics = calculate_orderflow_features(orderbook)
                logger.info(f"[{_}] DATA RECEIVED: Imbalance={metrics['of_imbalance']:.2f}, Delta={metrics['of_delta']:.0f}, Spread={metrics['of_spread']:.5f}")
                if metrics['of_imbalance'] != 1.0 or metrics['of_delta'] != 0:
                    logger.info("✅ SUCCESS: Orderflow data validated with non-zero metrics.")
                    return
            else:
                logger.info(f"[{_}] No orderbook data yet...")
        
        logger.warning("❌ FAILURE: No orderbook data received within timeout.")
        
    except Exception as e:
        logger.error(f"Test Error: {e}")
    finally:
        # Close connection if possible (though ssid remains)
        pass

if __name__ == "__main__":
    asyncio.run(test_orderflow())
