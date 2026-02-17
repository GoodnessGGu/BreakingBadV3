
print("STARTING TEST SCRIPT...", flush=True)
import sys
import os
print("Importing modules...", flush=True)
import time
import logging
import asyncio
import joblib
import pandas as pd
import torch
import numpy as np
from datetime import datetime

# Add parent dir to sys.path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    print("Importing IQClient...", flush=True)
    from iqclient import IQOptionAPI as IQClient, run_trade
    print("Importing ML Utils...", flush=True)
    from ml_utils import prepare_features
    print("Importing NN Utils...", flush=True)
    from nn_utils import BinaryOptionsNN
    print("Importing Settings...", flush=True)
    from settings import config
except ImportError as e:
    print(f"IMPORT ERROR: {e}", flush=True)
    sys.exit(1)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Config
ASSETS = [
    "EURUSD-OTC", 
    "GBPUSD-OTC", 
    "USDJPY-OTC", 
    "AUDUSD-OTC", 
    "USDCAD-OTC", 
    "NZDUSD-OTC"
]
TIMEFRAMES = {1: "1m", 5: "5m"} # Minutes to string
THRESHOLD = 0.55 # Minimum probability to trade
TRADE_AMOUNT = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Root directory for absolute paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

async def load_models():
    """Load models and scaler."""
    try:
        scaler_path = os.path.join(ROOT_DIR, "scaler.pkl")
        model_1m_path = os.path.join(ROOT_DIR, "model_1m.pth")
        model_5m_path = os.path.join(ROOT_DIR, "model_5m.pth")

        scaler = joblib.load(scaler_path)
        
        # Load 1m Model
        model_1m = BinaryOptionsNN(input_size=scaler.mean_.shape[0]).to(DEVICE)
        model_1m.load_state_dict(torch.load(model_1m_path, map_location=DEVICE))
        model_1m.eval()
        
        # Load 5m Model
        model_5m = BinaryOptionsNN(input_size=scaler.mean_.shape[0]).to(DEVICE)
        model_5m.load_state_dict(torch.load(model_5m_path, map_location=DEVICE))
        model_5m.eval()
        
        logger.info(f"✅ Models loaded. Scaler features: {scaler.mean_.shape[0]}")
        return scaler, model_1m, model_5m
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        return None, None, None

async def get_prediction(api, asset, tf_min, model, scaler):
    """Fetch candles and return model prediction (prob, direction, confidence, features)."""
    try:
        # Fetch candles (need enough for indicators)
        candles = api.get_candle_history(asset, count=300, timeframe=tf_min*60)
        if not candles:
            return None

        # Prepare Features
        df = pd.DataFrame(candles)
        
        # Use existing ml_utils to ensure consistency
        df_features = prepare_features(df)
        if df_features.empty:
            return None

        # Add ema_200 if missing (required by training data)
        if 'ema_200' not in df_features.columns:
            df_features['ema_200'] = df_features['close'].ewm(span=200, adjust=False).mean()

        # Drop non-feature columns (must match training cleanup EXACTLY)
        drop_cols = ["id", "signal", "pair", "asset", "market_type", "from", "to", "time", "outcome", "outcome_1m", "outcome_5m"]
        df_final = df_features.drop(columns=[c for c in drop_cols if c in df_features.columns], errors="ignore")
        
        # Get last row
        last_row = df_final.iloc[[-1]]
        
        # Scale (Ensure columns match scaler features and ORDER)
        if hasattr(scaler, "feature_names_in_"):
            last_row = last_row[scaler.feature_names_in_]
            
        X_scaled = scaler.transform(last_row)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            prob = model(X_tensor).item()
            
        direction = "CALL" if prob > 0.5 else "PUT"
        confidence = prob if prob > 0.5 else 1 - prob
        
        return {
            "prob": prob,
            "direction": direction,
            "confidence": confidence,
            "features": last_row.to_dict('records')[0]
        }
            
    except Exception as e:
        logger.error(f"Error in prediction for {asset} ({tf_min}m): {e}")
        return None

async def main():
    # Connect
    api = IQClient()
    try:
        print("Connecting to IQ Option...", flush=True)
        await api._connect()
    except Exception as e:
        logger.error(f"❌ Failed to connect to IQ Option: {e}")
        return
    logger.info("✅ Connected to IQ Option")
    
    # Load Brains
    scaler, model_1m, model_5m = await load_models()
    if not scaler:
        return

    logger.info(f"🧪 STARING MTF ALIGNMENT TEST on {len(ASSETS)} pairs...")
    logger.info(f"Rule: Trade ONLY when 1m and 5m directions MATCH")
    logger.info(f"Threshold: 1m or 5m conf >= {THRESHOLD} | Amount: ${TRADE_AMOUNT}")

    while True:
        for asset in ASSETS:
            logger.info(f"--- Analyzing {asset} ---")
            
            # 1. Get 1m Prediction
            pred_1m = await get_prediction(api, asset, 1, model_1m, scaler)
            
            # 2. Get 5m Prediction
            pred_5m = await get_prediction(api, asset, 5, model_5m, scaler)
            
            if not pred_1m or not pred_5m:
                continue

            dir1, conf1 = pred_1m["direction"], pred_1m["confidence"]
            dir5, conf5 = pred_5m["direction"], pred_5m["confidence"]

            logger.info(f"1m: {dir1} {conf1:.2%} | 5m: {dir5} {conf5:.2%}")

            # MTF ALIGNMENT RULE:
            # - Directions must match (both CALL or both PUT)
            # - At least one must be above the confidence threshold
            if dir1 == dir5:
                if conf1 >= THRESHOLD or conf5 >= THRESHOLD:
                    logger.info(f"🚀 MTF ALIGNMENT FOUND: {asset} {dir1} (1m+5m)")
                    
                    # Execute Trade (defaulting to 1m expiry for faster testing)
                    await run_trade(
                        api=api, 
                        asset=asset, 
                        direction=dir1, 
                        expiry=1, 
                        amount=TRADE_AMOUNT, 
                        auto_martingale=False,
                        features=pred_1m["features"]
                    )
                else:
                    logger.info("⚠️ Alignment found but confidence too low.")
            else:
                logger.info("❌ No alignment (Mismatching directions).")
            
            # Small delay between assets to avoid rate limits
            await asyncio.sleep(2)
            
        logger.info("💤 Waiting for next cycle...")
        await asyncio.sleep(45) 

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Test Stopped by User")
