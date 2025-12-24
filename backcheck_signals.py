
import asyncio
import logging
import re
from datetime import datetime, timedelta
import pytz
from iqclient import IQOptionAPI
from settings import config

# --- CONFIG ---
SIGNALS_INPUT = """
00:10 USDCHF PUT
00:15 USDCHF PUT
00:20 EURGBP CALL
00:25 EURGBP CALL
01:00 EURAUD CALL
01:05 EURAUD CALL
01:35 EURGBP CALL
01:40 EURGBP CALL
02:15 NZDUSD PUT
02:20 NZDUSD PUT
03:00 AUDCHF PUT
03:05 AUDCHF PUT
04:25 EURJPY PUT
04:30 EURJPY PUT
05:50 NZDJPY CALL
05:55 EURGBP CALL
05:55 NZDJPY CALL
06:00 EURGBP CALL
06:45 EURAUD CALL
06:50 EURAUD CALL
06:55 EURAUD PUT
07:00 EURAUD PUT
07:30 NZDCHF CALL
07:35 NZDCHF CALL
08:25 NZDJPY CALL
08:30 NZDJPY CALL
09:00 EURCHF CALL
09:05 EURCHF CALL
09:40 AUDCHF PUT
09:45 AUDCHF PUT
09:50 GBPUSD PUT
09:55 GBPJPY CALL
09:55 GBPUSD PUT
10:00 GBPJPY CALL
10:15 NZDUSD CALL
10:20 NZDUSD CALL
10:55 NZDUSD CALL
11:00 NZDUSD CALL
11:10 EURGBP CALL
11:15 EURGBP CALL
"""

TARGET_DATE = datetime.now().date() # Check signals for TODAY. Change if checking yesterday.
TIMEFRAME = 60 # 1 minute candles (Signal Expiry = 1m usually) - ADJUST IF 5M
MAX_GALES = 2 # Check up to 2 Martingales

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

async def main():
    api = IQOptionAPI()
    await api._connect()
    
    # 1. Parse Signals
    signals = []
    lines = SIGNALS_INPUT.strip().split('\n')
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            time_str = parts[0]
            asset = parts[1].upper()
            if "-OTC" not in asset and "OTC" not in asset:
                # Naive check for OTC vs Regular. 
                # If checking weekends, might need to force OTC.
                # For now assume input is correct or simple pairs.
                # NOTE: If user provides "USDCHF" but market is Closed, we might need "USDCHF-OTC".
                # Let's try regular first, fallback to OTC if needed? 
                pass
            
            direction = parts[2].upper()
            signals.append({'time': time_str, 'asset': asset, 'direction': direction})

    print(f"\n[CHECK] Checking {len(signals)} signals for date: {TARGET_DATE}\n")
    print(f"{'Time':<8} {'Asset':<10} {'Dir':<5} -> {'Result'}")
    print("-" * 50)

    for sig in signals:
        # Construct Datetime
        sig_dt_str = f"{TARGET_DATE} {sig['time']}"
        sig_dt = datetime.strptime(sig_dt_str, "%Y-%m-%d %H:%M")
        
        # Localize? Assuming signals are in same timezone as bot (Lagos/Sao Paulo)
        # IQ Option candles are queried by UTC timestamp usually or local computer time if simple API.
        # iqclient uses seconds timestamp.
        # We need to convert Local Time -> Timestamp.
        # Assuming script runs in same timezone as signals.
        
        ts = int(sig_dt.timestamp())
        
        # Calculate Outcome
        outcome = "UNKNOWN"
        icon = "❓"
        
        # We need candles starting from 'ts'. 
        # Check Base Trade (Gale 0)
        # Check Gale 1 (ts + timeframe)
        # Check Gale 2 (ts + 2*timeframe)
        
        # Fetch 3 candles starting from this time
        # Note: api.get_candle_history ends at a time? Or is it 'from'?
        # get_candle_history(asset, count, timeframe) usually gets LAST count candles.
        # To get specific time, we might need a different retrieval method or just get a chunk.
        # But iqclient.get_candle_history only supports 'count'.
        # We need to use underlying API method 'get_candles' which supports 'to' or 'from'.
        # api.api is the IQOptionProto.
        # api.api.get_candles(asset, timeframe, count, to_timestamp)
        # So we ask for candles ending shortly after our target.
        
        # We need candles covering: ts, ts+60, ts+120 (for 3 attempts)
        # Request ending at ts + 300 (buffer)
        
        
        # Manually fetch candles via WebSocket (since iqclient helper is for 'now')
        api.message_handler.candles = None
        
        try:
            active_id = api.market_manager.get_asset_id(sig['asset'])
        except KeyError:
             # Try OTC if regular failed lookup (naive retry)
             if "OTC" not in sig['asset']:
                 sig['asset'] = sig['asset'] + "-OTC"
                 try:
                     active_id = api.market_manager.get_asset_id(sig['asset'])
                 except KeyError:
                     # Skip if totally invalid
                     print(f"{sig['time']:<8} {sig['asset']:<10} {sig['direction']:<5} -> INVALID ASSET ⚠️")
                     continue
        
        # Calculate end time for query (cover all gales)
        end_query = ts + (MAX_GALES + 2) * TIMEFRAME
        
        msg = {
            "name": "get-candles",
            "version": "2.0",
            "body": {
                "active_id": active_id,
                "size": TIMEFRAME,
                "count": 10,
                "to": end_query,
                "only_closed": True,
                "split_normalization": True
            }
        }
        api.websocket.send_message("sendMessage", msg)
        
        # Wait for response
        retries = 20
        while api.message_handler.candles is None and retries > 0:
            await asyncio.sleep(0.1)
            retries -= 1
            
        candles = api.message_handler.candles
        if not candles:
             # Try OTC Fallback logic moved up or handled here
             pass

        gale_win = -1 # -1: Loss, 0: Direct Win, 1: Gale 1, 2: Gale 2
        
        target_ts = ts
        
        # Check Gales
        for g in range(MAX_GALES + 1):
            # Find candle opening at target_ts
            candle = next((c for c in candles if c['from'] == target_ts), None)
            
            if not candle:
                outcome = "NO DATA"
                break
                
            open_price = candle['open']
            close_price = candle['close']
            
            win = False
            if sig['direction'] == 'CALL' and close_price > open_price:
                win = True
            elif sig['direction'] == 'PUT' and close_price < open_price:
                win = True
            
            if win:
                gale_win = g
                break
            else:
                target_ts += TIMEFRAME # Move to next candle for Gale
        
        # Format Output
        if gale_win == 0:
            outcome = "WIN"
        elif gale_win > 0:
            outcome = f"WIN-G{gale_win}"
        elif outcome != "NO DATA":
            outcome = "LOSS"
            
        print(f"{sig['time']:<8} {sig['asset']:<10} {sig['direction']:<5} -> {outcome}")
        
        # Be nice to API speed limit
        await asyncio.sleep(0.2)

if __name__ == "__main__":
    asyncio.run(main())
