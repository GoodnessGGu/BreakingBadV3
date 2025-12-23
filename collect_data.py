import asyncio
import pandas as pd
import logging
from iqclient import IQOptionAPI
from strategies import analyze_strategy
from ml_utils import prepare_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def collect_and_label_data(api, asset, count=5000, timeframe=60):
    """
    Fetches data, runs the strategy to find signals, and labels them (Win/Loss).
    """
    logger.info(f"Fetching {count} candles for {asset}...")
    candles = api.get_candle_history(asset, count, timeframe)
    
    if not candles:
        logger.error("No candles received.")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(candles)
    cols = ['open', 'close', 'min', 'max', 'volume']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c])
            
    if 'from' in df.columns:
        df['time'] = pd.to_datetime(df['from'], unit='s')
        
    # --- 1. Prepare Features (RSI, Bollinger, etc.) ---
    # We do this on the whole DF first for efficiency
    df_features = prepare_features(df)
    
    # Re-align original candles to feature DF (features might drop initial rows due to NaN)
    # df_features contains the indicators + original columns
    
    # --- 2. Iterate and Find Signals ---
    labeled_data = []
    
    records = df_features.to_dict('records')
    min_candles = 35 # Minimum required by strategy
    
    logger.info("Labeling data...")
    for i in range(min_candles, len(records) - 1):
        # Current candle at 'i'. We want to see if strategy gives a signal at 'i'
        # Strategy input: History up to 'i'
        history_slice = records[:i+1]
        
        signal = analyze_strategy(history_slice, use_ai=False)
        
        if signal:
            # Check Outcome
            # Signal at 'i' means we trade on 'i+1'
            # If confirmed at close of 'i', we enter Open of 'i+1' or immediately.
            # Usually simplified: Compare Close of 'i' vs Close of 'i+1'? 
            # Or Open 'i+1' vs Close 'i+1'?
            # strategies.py usually implies entering on next candle.
            
            entry_candle = records[i] # Signal generated here
            next_candle = records[i+1] # Trade result here
            
            # Simple Outcome: 
            # CALL Win: Close(i+1) > Open(i+1)
            # PUT Win: Close(i+1) < Open(i+1)
            
            is_win = False
            if signal == "CALL":
                is_win = next_candle['close'] > next_candle['open']
            elif signal == "PUT":
                is_win = next_candle['close'] < next_candle['open']
                
            # Create data point
            # We want the MODEL to predict using features from `entry_candle` (i)
            # Target is `is_win`
            
            row = entry_candle.copy()
            row['signal'] = signal
            row['outcome'] = 1 if is_win else 0
            
            labeled_data.append(row)
            
    if not labeled_data:
        logger.warning(f"No signals found in {len(df)} candles.")
        return None
        
    logger.info(f"Found {len(labeled_data)} training examples.")
    return pd.DataFrame(labeled_data)

async def run_collection_cycle(api_instance=None):
    """
    Runs the full data collection pipeline.
    """
    logger.info("🚀 Starting Data Collection Cycle...")
    
    # Use provided API or create new one
    api = api_instance
    should_disconnect = False
    
    if api is None:
        logger.info("No API provided, creating new connection...")
        api = IQOptionAPI()
        await api._connect()
        should_disconnect = True
        
    try:
        if not api.check_connect():
            logger.warning("API not connected. Attempting reconnect...")
            await api._connect()

        # Collect data for a few assets to generalize better
        assets = [
            "EURUSD-OTC", "USDJPY-OTC", "GBPUSD-OTC", 
            "AUDCAD-OTC", "NZDUSD-OTC", "USDCHF-OTC", 
            "EURGBP-OTC", "USDCAD-OTC"
        ]
        all_data = []
        
        for asset in assets:
            df = await collect_and_label_data(api, asset, count=10000)
            if df is not None:
                 df['asset'] = asset
                 all_data.append(df)
            else:
                 logger.warning(f"Skipping {asset} (No data).")
                 
            # Small delay to be nice to API
            await asyncio.sleep(2)
        
        if all_data:
            new_df = pd.concat(all_data, ignore_index=True)
            
            # Load existing data if available to Append
            try:
                import os
                if os.path.exists("training_data.csv"):
                    existing_df = pd.read_csv("training_data.csv")
                    # Ensure time is datetime for accurate deduplication
                    if 'time' in existing_df.columns:
                        existing_df['time'] = pd.to_datetime(existing_df['time'])
                    if 'time' in new_df.columns:
                        new_df['time'] = pd.to_datetime(new_df['time'])
                        
                    combined_df = pd.concat([existing_df, new_df])
                    
                    # Remove duplicates (Same Asset + Same Time)
                    # Keep 'last' (newest version of candle) or 'first' doesn't matter much if data is same
                    final_df = combined_df.drop_duplicates(subset=['time', 'asset'], keep='last')
                else:
                    final_df = new_df
            except Exception as e:
                logger.error(f"Error merging data: {e}")
                final_df = new_df # Fallback to new data only
                
            final_df.to_csv("training_data.csv", index=False)
            logger.info(f"✅ Collection Complete. Total Database Size: {len(final_df)} records.")
            
            # Trigger Training Immediately
            from ml_utils import train_model
            logger.info("🧠 Starting Model Training...")
            train_model("training_data.csv")
            
            return True
        else:
            logger.error("❌ Failed to collect any data.")
            return False
            
    except Exception as e:
        logger.error(f"Collection Cycle Error: {e}")
        return False
    finally:
        if should_disconnect:
            pass # We could disconnect, but IQOptionAPI doesn't strictly need explicit close usually

async def main():
    await run_collection_cycle()

if __name__ == "__main__":
    asyncio.run(main())
