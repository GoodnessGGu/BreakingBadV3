import time
import json
import logging
from forex_mcp_client import IQForexMCPClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TestMCPTrade")

def test_trade_lifecycle():
    client = IQForexMCPClient()
    logger.info("Initializing MCP client...")
    if not client.initialize():
        logger.error("Failed to initialize")
        return
        
    training_bal = client.get_training_balance()
    if not training_bal:
        logger.error("No training balance found!")
        return
        
    balance_id = training_bal["balance_id"]
    logger.info(f"Using Training Balance ID: {balance_id} (Equity: ${training_bal['equity']})")
    
    # 1. Preview current price on EURUSD (asset_id 1)
    preview = client.calculate_order_size(asset_id=1, balance_currency="USD", lots=0.001, leverage=50)
    logger.info(f"Preview: {preview}")
    buy_price = preview.get("buy_price", 1.1600)
    
    # Set SL 25 pips below, TP 50 pips above (1:2 R:R)
    sl_price = round(buy_price - 0.0025, 5)
    tp_price = round(buy_price + 0.0050, 5)
    
    logger.info(f"Targeting BUY: Price={buy_price}, SL={sl_price}, TP={tp_price}")
    
    # 2. Place market order
    order_res = client.place_market_order(
        side="buy",
        balance_id=balance_id,
        instrument_id="mf.1",
        asset_id=1,
        lots=0.001,
        leverage=50,
        stop_loss=sl_price,
        take_profit=tp_price
    )
    logger.info(f"Order Result: {json.dumps(order_res, indent=2)}")
    
    # 3. Check open positions
    time.sleep(2)
    positions = client.list_positions(balance_id=balance_id)
    logger.info(f"Open Positions ({len(positions)}): {json.dumps(positions, indent=2)}")
    
    if positions:
        pos = positions[0]
        pos_id = pos.get("position_id") or pos.get("id")
        logger.info(f"Testing Trailing SL update on Position #{pos_id}...")
        
        # Move SL up by 5 pips (closer)
        new_sl = round(sl_price + 0.0005, 5)
        sl_res = client.change_position_stop_loss(position_id=pos_id, level=new_sl)
        logger.info(f"SL Update Result: {sl_res}")
        
        time.sleep(2)
        # 4. Close position
        logger.info(f"Closing position #{pos_id}...")
        close_res = client.close_position(position_id=pos_id)
        logger.info(f"Close Result: {close_res}")
        
    # 5. Check Trade History
    time.sleep(2)
    history = client.get_trade_history(balance_id=balance_id, limit=5)
    logger.info(f"Recent Trade History ({len(history)}): {json.dumps(history[:2], indent=2)}")

if __name__ == "__main__":
    test_trade_lifecycle()
