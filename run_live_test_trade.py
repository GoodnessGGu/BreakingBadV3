"""
run_live_test_trade.py - Execute a Live Test Trade on IQ Option Marginal Forex

Calculates current Smart Trail parameters on EUR/USD, sizes position with 1% risk,
and executes a real margin order with protective Stop Loss and 1:2 Take Profit.
"""

import time
import json
import logging
import pandas as pd
from forex_mcp_client import IQForexMCPClient
from smart_trail_forex_bot import calculate_smart_trail
from gsheet_logger import gsheet_logger

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("LiveTestTrade")

def main():
    client = IQForexMCPClient()
    logger.info("Connecting to IQ Option Marginal Forex engine via MCP...")
    if not client.initialize():
        logger.error("Failed to initialize MCP session.")
        return

    bal = client.get_training_balance()
    if not bal:
        logger.error("No training balance found!")
        return
        
    balance_id = bal["balance_id"]
    equity = float(bal["equity"])
    free_margin = float(bal["free_margin"])
    logger.info(f"Using Practice Account ID: {balance_id} | Equity: ${equity:.2f} | Free Margin: ${free_margin:.2f}")

    # 1. Fetch candles & calculate Smart Trail on EUR/USD (asset_id 1)
    candles = client.get_candles(asset_id=1, size=60, count=60)
    df = pd.DataFrame(candles)
    for c in ['open', 'close', 'min', 'max', 'high', 'low']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c])

    df = calculate_smart_trail(df, length=14, multiplier=2.0, sensitivity=3)
    curr = df.iloc[-1]
    
    smart_trend = int(curr['smart_trend'])
    smart_trail_val = float(curr['smart_trail_value'])
    atr = float(curr['atr'])
    
    # 2. Preview prices
    preview = client.calculate_order_size(asset_id=1, balance_currency="USD", lots=0.001, leverage=50)
    buy_price = float(preview.get("buy_price", curr['close']))
    sell_price = float(preview.get("sell_price", curr['close']))

    # Minimum stop distance on EUR/USD is 0.0002 (2 pips)
    min_dist = 0.00025

    if smart_trend == 1:
        side = "buy"
        entry_price = buy_price
        sl_price = round(min(entry_price - min_dist, smart_trail_val), 5)
        # Ensure at least min_dist
        if (entry_price - sl_price) < min_dist:
            sl_price = round(entry_price - min_dist, 5)
        risk = entry_price - sl_price
        tp_price = round(entry_price + (risk * 2.0), 5)
    else:
        side = "sell"
        entry_price = sell_price
        sl_price = round(max(entry_price + min_dist, smart_trail_val), 5)
        if (sl_price - entry_price) < min_dist:
            sl_price = round(entry_price + min_dist, 5)
        risk = sl_price - entry_price
        tp_price = round(entry_price - (risk * 2.0), 5)

    pips_risk = round(risk / 0.0001, 1)
    pips_target = round(pips_risk * 2.0, 1)
    
    logger.info("========================================")
    logger.info(f"📊 SIGNAL: EUR/USD | Smart Trend: {'BULLISH' if smart_trend == 1 else 'BEARISH'}")
    logger.info(f"🎯 DIRECTION: {side.upper()}")
    logger.info(f"💵 ENTRY PRICE: {entry_price:.5f}")
    logger.info(f"🛡️ STOP LOSS:   {sl_price:.5f} ({pips_risk} pips risk at Smart Trail)")
    logger.info(f"🏆 TAKE PROFIT: {tp_price:.5f} ({pips_target} pips target, 1:2 R:R)")
    logger.info("========================================")

    # 3. Calculate lot size for 1% equity risk
    lots, sizing = client.calculate_lot_size(
        asset_id=1,
        entry_price=entry_price,
        sl_price=sl_price,
        risk_usd=max(0.50, equity * 0.01),
        balance_currency="USD",
        leverage=50,
        free_margin=free_margin
    )
    
    margin_req = sizing.get("margin", 2.32)
    logger.info(f"📈 SIZING: {lots} lots | Required Margin: ${margin_req:.2f} | Leverage: 50x")

    # 4. Place order
    logger.info("🚀 Sending market order to IQ Option MCP Gateway...")
    order_res = client.place_market_order(
        side=side,
        balance_id=balance_id,
        instrument_id="mf.1",
        asset_id=1,
        lots=lots,
        leverage=50,
        stop_loss=sl_price,
        take_profit=tp_price,
        is_margin_isolated=True,
        keep_position_open=False
    )

    if "order_id" not in order_res:
        logger.error(f"Order failed: {order_res}")
        return

    order_id = order_res["order_id"]
    logger.info(f"✅ ORDER ACCEPTED! Order ID: #{order_id}")
    
    # 5. Confirm open position
    time.sleep(2)
    positions = client.list_positions(balance_id=balance_id)
    target_pos = None
    for p in positions:
        if p.get("asset_id") == 1:
            target_pos = p
            break

    if target_pos:
        pos_id = target_pos.get("position_id")
        open_px = target_pos.get("open_price")
        cur_px = target_pos.get("current_price")
        pnl = target_pos.get("expected_pnl")
        logger.info("========================================")
        logger.info(f"🎉 POSITION CONFIRMED OPEN on IQ Option!")
        logger.info(f"   Position ID: #{pos_id}")
        logger.info(f"   Instrument:  EUR/USD ({target_pos.get('instrument_id')})")
        logger.info(f"   Side:        {target_pos.get('type').upper()}")
        logger.info(f"   Filled At:   {open_px}")
        logger.info(f"   Current Px:  {cur_px}")
        logger.info(f"   Stop Loss:   {target_pos.get('stop_lose_price')}")
        logger.info(f"   Take Profit: {target_pos.get('take_profit_price')}")
        logger.info(f"   Live PnL:    ${pnl:.2f}")
        logger.info("========================================")
    else:
        logger.warning("Order placed, but position not immediately returned in list_positions.")

if __name__ == "__main__":
    main()
