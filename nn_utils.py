try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        class Module: pass
        class Sequential:
            def __init__(self, *args): pass
        def Linear(*args): pass
        def ReLU(*args): pass
        def BatchNorm1d(*args): pass
        def Dropout(*args): pass
        def Sigmoid(*args): pass
import joblib
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

if TORCH_AVAILABLE:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"

# =========================
# MODEL
# =========================
class BinaryOptionsNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# =========================
# UTILS
# =========================
def get_scaler():
    if os.path.exists("scaler.pkl"):
        return joblib.load("scaler.pkl")
    return None

def load_nn_model(expiry, input_size):
    path = f"model_{expiry}m.pth"
    if not os.path.exists(path):
        return None
        
    if not TORCH_AVAILABLE:
        logger.warning(f"NN Confirmation skipped (Torch not installed) for {path}")
        return None
        
    model = BinaryOptionsNN(input_size)
    try:
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Failed to load NN model {path}: {e}")
        return None

# =========================
# PREDICT
# =========================
def predict_nn_trade(features_df, expiry=1):
    """
    features_df = DataFrame of latest candle features
    expiry = 1 or 5
    """
    scaler = get_scaler()
    if not scaler:
        return 0.5, False

    try:
        # Align features with training (StandardScaler needs exact same columns)
        # Drop columns that were dropped during training
        drop_cols = ["signal", "pair", "asset", "market_type", "from", "to", "time", "outcome", "outcome_1m", "outcome_5m"]
        X = features_df.drop(columns=[c for c in drop_cols if c in features_df.columns], errors="ignore")
        
        # Scaling
        X_scaled = scaler.transform(X)
        
        if not TORCH_AVAILABLE:
            return 0.5

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

        model = load_nn_model(expiry, X_tensor.shape[1])
        if not model:
            return 0.5, False

        with torch.no_grad():
            prob = model(X_tensor).item()

        # Custom threshold can be applied in strategies.py
        return prob
    except Exception as e:
        logger.error(f"NN Prediction error: {e}")
        return 0.5
