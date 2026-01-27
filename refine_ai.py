# refine_ai.py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import os
import numpy as np
from nn_utils import TradingNN, get_nn_features

# --- CONFIG ---
FEEDBACK_FILE = "live_trade_feedback.csv"
LEARNING_RATE = 0.0001 # Very low for fine-tuning
EPOCHS = 10
BATCH_SIZE = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def refine_models():
    if not os.path.exists(FEEDBACK_FILE):
        print(f"❌ No live feedback data found at {FEEDBACK_FILE}")
        return

    # 1. Load Feedback Data
    df = pd.read_csv(FEEDBACK_FILE)
    if len(df) < BATCH_SIZE:
        print(f"⚠️ Not enough live trade data yet (Need at least {BATCH_SIZE} records, have {len(df)})")
        return

    print(f"🧠 Refining models with {len(df)} live trade records...")

    # 2. Prepare Data
    # NN features are a specific subset from get_nn_features()
    nn_feats = get_nn_features()
    
    # Ensure all required features exist in feedback
    missing = [f for f in nn_feats if f not in df.columns]
    if missing:
        print(f"❌ Feedback file missing required NN features: {missing}")
        return

    # Load Scaler
    try:
        scaler = joblib.load("scaler.pkl")
    except:
        print("❌ Could not load scaler.pkl")
        return

    # Process per timeframe
    for tf in ["1m", "5m"]:
        model_path = f"model_{tf}.pth"
        if not os.path.exists(model_path):
            continue

        # Filter feedback for this expiry
        expiry_val = 1 if tf == "1m" else 5
        tf_data = df[df['expiry'] == expiry_val].copy()
        
        if tf_data.empty:
            print(f"ℹ️ No live trades found for {tf}")
            continue

        print(f"🚀 Fine-tuning {tf} model with {len(tf_data)} samples...")

        # Feature preparation
        X_raw = tf_data[nn_feats].values
        X_scaled = scaler.transform(X_raw)
        y = tf_data['result'].values

        # Convert to Tensors
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        y_tensor = torch.FloatTensor(y).view(-1, 1).to(device)

        # 3. Load Model
        input_dim = len(nn_feats)
        model = TradingNN(input_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.train()

        # 4. Fine-Tune Loop
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                 print(f"  [{tf}] Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f}")

        # 5. Save Updated Model
        torch.save(model.state_dict(), model_path)
        print(f"✅ {tf} Model updated with latest live experience!")

    # Optional: Archive or clear feedback after learning?
    # os.rename(FEEDBACK_FILE, f"feedback_archive_{int(time.time())}.csv")
    print("\n🎉 AI Refinement Complete!")

if __name__ == "__main__":
    refine_models()
