import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
MODEL_PATH_TEMPLATE = os.path.join(MODELS_DIR, "trade_model_{}.pkl")
# Legacy path for migration/fallback
LEGACY_MODEL_PATH = os.path.join(MODELS_DIR, "trade_model.pkl")

# --- Indicators ---
def calculate_rsi(series, period=14):
    """Calculates Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(series, period=20, std_dev=2):
    """Calculates Bollinger Bands."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, lower

def calculate_adx(df, period=14):
    """Calculates ADX."""
    alpha = 1/period
    # True Range
    h_l = df['max'] - df['min']
    h_yc = abs(df['max'] - df['close'].shift(1))
    l_yc = abs(df['min'] - df['close'].shift(1))
    tr = pd.concat([h_l, h_yc, l_yc], axis=1).max(axis=1)
    
    # Directional Movement
    up = df['max'] - df['max'].shift(1)
    down = df['min'].shift(1) - df['min']
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    # Smoothing
    tr_s = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean()
    
    dx = 100 * abs(plus_dm_s - minus_dm_s) / (plus_dm_s + minus_dm_s)
    return dx.ewm(alpha=alpha, adjust=False).mean()

def calculate_atr(df, period=14):
    """Calculates Average True Range (Volatility)."""
    high = df['max']
    low = df['min']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def prepare_features(df):
    """
    Generates technical indicators as features for the ML model.
    """
    df = df.copy()
    
    # Ensure numeric
    cols = ['open', 'close', 'min', 'max', 'volume']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c])
            
    if 'time' in df.columns:
        df['hour'] = df['time'].dt.hour
        # Encoding cyclical time features can be better, but raw hour is a good start
    
    # 1. Momentum & Trend
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['adx'] = calculate_adx(df, 14)
    df['atr'] = calculate_atr(df, 14)
    
    # NEW: Rate of Change (Momentum)
    df['roc_5'] = df['close'].pct_change(5)
    df['roc_10'] = df['close'].pct_change(10)
    
    # 2. Moving Averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_100'] = df['close'].ewm(span=100, adjust=False).mean()
    
    # NEW: Price Distance from MAs
    df['dist_sma20'] = (df['close'] - df['sma_20']) / df['close']
    df['dist_sma50'] = (df['close'] - df['sma_50']) / df['close'] 
    
    # 3. Bollinger Bands
    df['bb_upper'], df['bb_lower'] = calculate_bollinger_bands(df['close'], 20, 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
    df['bb_pos'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # NEW: Volume Indicators
    if 'volume' in df.columns:
        df['vol_sma20'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / (df['vol_sma20'] + 1e-9)
        
    # NEW: Pivot Points (Classic)
    # Using rolling window to estimate daily high/low roughly or just local swing
    # For 1m candles, we can use local pivots (e.g. 60 min)
    # Better to use recent 50-candle High/Low/Close for support/resistance context
    df['recent_high'] = df['max'].rolling(50).max()
    df['recent_low'] = df['min'].rolling(50).min()
    df['pivot'] = (df['recent_high'] + df['recent_low'] + df['close']) / 3
    df['res1'] = (2 * df['pivot']) - df['recent_low']
    df['sup1'] = (2 * df['pivot']) - df['recent_high']
    
    # 4. Price Action
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['max'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['min']
    
    # Pattern: Engulfing (1 = Bullish, -1 = Bearish, 0 = None)
    # Bullish Engulfing: Prev Red, Curr Green, Curr Open < Prev Close, Curr Close > Prev Open
    # Bearish Engulfing: Prev Green, Curr Red, Curr Open > Prev Close, Curr Close < Prev Open
    
    prev_open = df['open'].shift(1)
    prev_close = df['close'].shift(1)
    curr_open = df['open']
    curr_close = df['close']
    
    # Vectorized Engulfing Logic
    is_bullish_engulfing = (prev_close < prev_open) & (curr_close > curr_open) & \
                           (curr_open < prev_close) & (curr_close > prev_open)
                           
    is_bearish_engulfing = (prev_close > prev_open) & (curr_close < curr_open) & \
                           (curr_open > prev_close) & (curr_close < prev_open)
                           
    df['pattern_engulfing'] = 0
    df.loc[is_bullish_engulfing, 'pattern_engulfing'] = 1
    df.loc[is_bearish_engulfing, 'pattern_engulfing'] = -1

    # NEW: Signal Indicators as Features
    # AM_IQ logic
    df['sma_fast'] = df['close'].rolling(window=1).mean()
    df['sma_slow'] = df['close'].rolling(window=34).mean()
    df['buffer1'] = df['sma_fast'] - df['sma_slow']
    df['buffer2'] = df['buffer1'].rolling(4).apply(
        lambda x: ((x * np.arange(1, 5)).sum()) / np.arange(1, 5).sum(), 
        raw=True
    )
    df['amiq_call'] = ((df['buffer1'] > df['buffer2']) & (df['buffer1'].shift(1) < df['buffer2'].shift(1))).astype(int)
    df['amiq_put'] = ((df['buffer1'] < df['buffer2']) & (df['buffer1'].shift(1) > df['buffer2'].shift(1))).astype(int)
    
    # Sniper logic
    prev1_o = df['open'].shift(1)
    prev1_c = df['close'].shift(1)
    prev2_o = df['open'].shift(2)
    prev2_c = df['close'].shift(2)
    prev3_o = df['open'].shift(3)
    prev3_c = df['close'].shift(3)
    curr_o = df['open']
    curr_c = df['close']
    
    df['sniper_call'] = (
        (prev3_c > prev3_o) & (prev2_c > prev2_o) &
        (prev1_c < prev1_o) & (prev1_c > prev2_o) &
        (prev1_o > prev2_o) & (curr_c > curr_o)
    ).astype(int)
    
    df['sniper_put'] = (
        (prev3_c < prev3_o) & (prev2_c < prev2_o) &
        (prev1_c > prev1_o) & (prev1_c < prev2_o) &
        (prev1_o < prev2_o) & (curr_c < curr_o)
    ).astype(int)

    # 5. Lagged Features (Previous candles)
    for lag in [1, 2, 3]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
        
    # Drop rows with NaN (due to rolling windows)
    df = df.dropna()
    return df

def train_model(data_path="training_data.csv", market_type="otc"):
    """
    Trains a Gradient Boosting model using labeled data.
    market_type: "otc" or "real"
    """
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    
    logger.info("Loading data...")
    df = pd.read_csv(data_path)
    df.dropna(inplace=True)
    
    # Separate features (X) and target (y)
    if 'outcome' not in df.columns:
        logger.error("Data missing 'outcome' column.")
        return

    # Drop non-feature columns
    drop_cols = ['time', 'outcome', 'signal', 'asset', 'from', 'to']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['outcome']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train - Gradient Boosting
    logger.info("Training Gradient Boosting Classifier...")
    # Parameters tuned for stability
    clf = GradientBoostingClassifier(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=4, 
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Model Accuracy: {acc:.2f}")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    # Save
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    save_path = MODEL_PATH_TEMPLATE.format(market_type)
    joblib.dump(clf, save_path)
    logger.info(f"Model saved to {save_path}")
    return clf

def load_model(market_type="otc"):
    """Loads the trained model for a specific market type."""
    path = MODEL_PATH_TEMPLATE.format(market_type)
    
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            logger.error(f"Failed to load {market_type} model: {e}")
    elif market_type == "otc" and os.path.exists(LEGACY_MODEL_PATH):
        # Migration logic: if otc model doesn't exist but legacy one does, load legacy
        try:
            logger.info(f"Loading legacy model for OTC...")
            return joblib.load(LEGACY_MODEL_PATH)
        except Exception as e:
            logger.error(f"Failed to load legacy model: {e}")
    
    return None

def predict_signal(model, features_df):
    """
    Predicts outcome for a single row of features.
    Returns: 1 (Win) or 0 (Loss)
    """
    if model is None:
        return 1 # Fallback: Assume win if no model
        
    try:
        # Align features with model
        if hasattr(model, "feature_names_in_"):
            # Only keep columns that the model knows
            # Add missing columns as 0 (if any, though unlikely if prepare_features is consistent)
            valid_cols = [c for c in model.feature_names_in_ if c in features_df.columns]
            
            if len(valid_cols) < len(model.feature_names_in_):
                missing = set(model.feature_names_in_) - set(features_df.columns)
                logger.warning(f"Missing features for prediction: {missing}")
                # Optional: Add missing as 0 or fail. For now, let's try to proceed with what we have if possible, 
                # but sklearn usually strictly requires all features in order.
                for c in missing:
                    features_df[c] = 0
            
            # Reorder columns to match model
            features_df = features_df[model.feature_names_in_]
            
        prediction = model.predict(features_df)
        return prediction[0]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return 1 # Fallback

if __name__ == "__main__":
    train_model()
