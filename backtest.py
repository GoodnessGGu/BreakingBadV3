import asyncio
import pandas as pd
import numpy as np
import logging
from iqclient import IQOptionAPI
from strategies import calculate_rsi, wma, calculate_adx, calculate_bollinger_bands

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def fetch_historical_data(api, asset, timeframe, count):
    """Fetches historical candles from IQ Option."""
    logger.info(f"Fetching {count} candles for {asset} ({timeframe}s)...")
    
    # Fetch candles (IQ Option API typically returns up to 1000)
    candles = api.get_candle_history(asset, count, timeframe)
    if not candles:
        logger.error("No candles received.")
        return None
    
    df = pd.DataFrame(candles)
    
    # Ensure numeric columns
    cols = ['open', 'close', 'min', 'max', 'volume']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c])
    
    # Convert timestamp to datetime for readability
    if 'from' in df.columns:
        df['time'] = pd.to_datetime(df['from'], unit='s')
         
    return df

def apply_strategy(df):
    """Applies the strategy logic to the DataFrame (Vectorized)."""
    
    # --- 1. Indicators ---
    df['trend_sma'] = df['close'].rolling(window=100).mean()
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['adx'] = calculate_adx(df, 14)
    df['bb_upper'], df['bb_lower'] = calculate_bollinger_bands(df['close'], 20, 2)
    
    df['sma_fast'] = df['close'].rolling(window=1).mean() # Close price
    df['sma_slow'] = df['close'].rolling(window=34).mean()
    df['buffer1'] = df['sma_fast'] - df['sma_slow']
    df['buffer2'] = wma(df['buffer1'], 4)
    
    # --- 2. Market State Filters ---
    is_uptrend = df['close'] > df['trend_sma']
    is_downtrend = df['close'] < df['trend_sma']
    
    safe_call_rsi = (df['rsi'] > 40) & (df['rsi'] < 70)
    safe_put_rsi = (df['rsi'] > 30) & (df['rsi'] < 60)
    strong_trend = df['adx'] > 25
    bb_width = (df['bb_upper'] - df['bb_lower']) / df['close']
    decent_volatility = bb_width > 0.0005
    
    # --- 3. Strategy 1: MA Crossover ---
    # Signal when Buffer1 crosses Buffer2
    buf1 = df['buffer1']
    buf2 = df['buffer2']
    
    ma_call = (buf1 > buf2) & (buf1.shift(1) < buf2.shift(1))
    ma_put = (buf1 < buf2) & (buf1.shift(1) > buf2.shift(1))
    
    # --- 4. Strategy 2: Sniper Pattern ---
    o = df['open']
    c = df['close']
    
    # Shifted values for pattern recognition (o1 = previous candle, o2 = 2 candles ago...)
    o1, c1 = o.shift(1), c.shift(1)
    o2, c2 = o.shift(2), c.shift(2)
    o3, c3 = o.shift(3), c.shift(3)
    
    sniper_call = (
        (o3 < c3) & (o2 < c2) &       # Two Green candles
        (o1 > c1) &                   # One Red candle (Pullback)
        (c1 > o2) & (o1 > o2) &       # Weak pullback
        (o < c)                       # Current Green (Resumption)
    )
    
    sniper_put = (
        (o3 > c3) & (o2 > c2) &       # Two Red candles
        (o1 < c1) &                   # One Green candle (Pullback)
        (c1 < o2) & (o1 < o2) &       # Weak pullback
        (o > c)                       # Current Red (Resumption)
    )
    
    # --- 5. Filter: Avoid Dojis (Indecision) ---
    # Body must be at least 15% of the total candle range
    body_size = (df['close'] - df['open']).abs()
    total_range = df['max'] - df['min']
    valid_candle = body_size > (total_range * 0.15)
    
    # --- 5. Combine Signals ---
    df['signal'] = 0
    
    # Priority: Sniper > MA
    # Note: We use 'loc' to apply signals where filters match
    
    # MA Signals
    df.loc[ma_call & is_uptrend & safe_call_rsi & valid_candle & strong_trend & decent_volatility, 'signal'] = 1   # CALL
    df.loc[ma_put & is_downtrend & safe_put_rsi & valid_candle & strong_trend & decent_volatility, 'signal'] = -1  # PUT
    
    # Sniper Signals (Overwrite MA if present)
    df.loc[sniper_call & is_uptrend & valid_candle & strong_trend & decent_volatility, 'signal'] = 1
    df.loc[sniper_put & is_downtrend & valid_candle & strong_trend & decent_volatility, 'signal'] = -1
    
    return df

def simulate_trades(df, max_gales=2):
    """Simulates trades with Martingale."""
    # If row 'i' has a signal, we enter trade on row 'i+1'
    df['trade_action'] = df['signal'].shift(1)
    df['win'] = False
    df['gale_level'] = -1 # -1 means no trade, 0=Direct, 1=Gale1, etc.
    df['pnl'] = 0.0
    
    # Iterative approach for accurate Martingale simulation
    # (Vectorization is harder with conditional lookahead)
    trades_count = 0
    wins = 0
    
    # Payout approx 87% (0.87), Loss is -100% (-1.0)
    payout_rate = 0.87
    base_amount = 1.0
    multiplier = 2.2
    
    for i in range(len(df)):
        signal = df['trade_action'].iloc[i]
        
        if signal in [1, -1]:
            # Check Direct Win
            current_candle = df.iloc[i]
            is_win = (signal == 1 and current_candle['close'] > current_candle['open']) or \
                     (signal == -1 and current_candle['close'] < current_candle['open'])
            
            if is_win:
                df.at[i, 'win'] = True
                df.at[i, 'gale_level'] = 0
                df.at[i, 'pnl'] = base_amount * payout_rate
            else:
                # Martingale Logic
                current_bet = base_amount
                total_invested = base_amount
                gale_win = False
                for g in range(1, max_gales + 1):
                    if i + g >= len(df): break # End of data
                    
                    current_bet *= multiplier
                    total_invested += current_bet
                    next_candle = df.iloc[i + g]
                    # Martingale usually follows same direction
                    is_gale_win = (signal == 1 and next_candle['close'] > next_candle['open']) or \
                                  (signal == -1 and next_candle['close'] < next_candle['open'])
                    
                    if is_gale_win:
                        df.at[i, 'win'] = True
                        df.at[i, 'gale_level'] = g
                        revenue = current_bet * (1 + payout_rate)
                        df.at[i, 'pnl'] = revenue - total_invested
                        gale_win = True
                        break
                
                if not gale_win:
                    df.at[i, 'win'] = False
                    df.at[i, 'gale_level'] = max_gales + 1 # Mark as full loss (distinct from max gale win)
                    df.at[i, 'pnl'] = -total_invested
    
    return df

async def main():
    api = IQOptionAPI()
    await api._connect()
    
    asset = "EURUSD-OTC"
    timeframe = 60 # 1 minute
    count = 1000   # Number of candles to test
    max_gales = 1  # Reduced to 1 to improve PnL (Gale 2 was unprofitable)
    
    df = await fetch_historical_data(api, asset, timeframe, count)
    if df is None: return
        
    df = apply_strategy(df)
    df = simulate_trades(df, max_gales=max_gales)
    
    # --- Statistics ---
    # Filter only rows where a trade occurred
    trades = df[df['trade_action'].isin([1, -1])].copy()
    
    total_trades = len(trades)
    wins = trades['win'].sum()
    losses = total_trades - wins
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    total_pnl = trades['pnl'].sum()
    
    # Calculate cumulative PnL for the report
    trades['cumulative_pnl'] = trades['pnl'].cumsum()
    
    # Count gales
    gale_counts = trades['gale_level'].value_counts().sort_index()
    
    print(f"\n{'='*40}")
    print(f"📊 BACKTEST REPORT: {asset}")
    print(f"{'='*40}")
    print(f"Timeframe:      {timeframe}s")
    print(f"Candles:        {len(df)}")
    print(f"Total Trades:   {total_trades}")
    print(f"✅ Wins:         {wins}")
    print(f"❌ Losses:       {losses}")
    print(f"🏆 Win Rate:     {win_rate:.2f}%")
    print(f"💰 Est. PnL:     ${total_pnl:.2f}")
    print(f"{'-'*40}")
    print("Martingale Breakdown:")
    for g, count in gale_counts.items():
        if g == 0: label = "Direct Win"
        elif g <= max_gales: label = f"Gale {g} Win"
        else: label = f"Full Loss (Gale {max_gales})"
        print(f"  {label}: {count}")
    print(f"{'='*40}\n")
    
    if total_trades > 0:
        print("Last 5 Trades:")
        print(trades[['time', 'trade_action', 'win', 'gale_level', 'pnl', 'cumulative_pnl']].tail(5))
        
        # Export to Excel
        try:
            filename = f"backtest_{asset}_{timeframe}s.xlsx"
            # Format for cleaner Excel output
            export_df = trades[['time', 'open', 'close', 'trade_action', 'win', 'gale_level', 'pnl', 'cumulative_pnl']].copy()
            export_df['trade_action'] = export_df['trade_action'].apply(lambda x: 'CALL' if x == 1 else 'PUT')
            export_df['result'] = export_df['win'].apply(lambda x: 'WIN' if x else 'LOSS')
            export_df = export_df[['time', 'trade_action', 'open', 'close', 'result', 'gale_level', 'pnl', 'cumulative_pnl']]
            
            export_df.to_excel(filename, index=False)
            print(f"\n💾 Detailed report saved to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save Excel report: {e}")

if __name__ == "__main__":
    asyncio.run(main())