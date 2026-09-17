"""
realtime_bot3.py - RealtimeBot3 Standalone Trader & Multi-Tab Recovery Engine
Supports 4 recovery strategy models and logs to separate Google Sheet tabs:
  1. MODE_GALE1_CAP        -> Tab: "Realtime_Bot_Gale1_Cap"
  2. MODE_DISTRIBUTED      -> Tab: "Realtime_Bot_Distributed"
  3. MODE_FLAT_RECOVERY    -> Tab: "Realtime_Bot_Flat_Recovery"
  4. MODE_AGGRESSIVE_CROSS -> Tab: "Realtime_Bot_Aggressive_Cross"
  5. MODE_BASELINE         -> Tab: "Realtime_Bot_Baseline"
"""

import asyncio
import logging
import os
import sys
import json
import argparse
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.getcwd())

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from iqclient import IQOptionAPI, run_trade
from settings import config
from trade import calculate_dynamic_trade_amount
from trade_database import db
from gsheet_logger import gsheet_logger

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RealtimeBot3")

DEFAULT_ASSETS = ["EURUSD-OTC", "GBPUSD-OTC", "USDJPY-OTC", "EURGBP-OTC", "AUDUSD-OTC"]

TAB_MAPPING = {
    "MODE_GALE1_CAP": "Realtime_Bot_Gale1_Cap",
    "MODE_DISTRIBUTED": "Realtime_Bot_Distributed",
    "MODE_FLAT_RECOVERY": "Realtime_Bot_Flat_Recovery",
    "MODE_AGGRESSIVE_CROSS": "Realtime_Bot_Aggressive_Cross",
    "MODE_BASELINE": "Realtime_Bot_Baseline"
}


class RecoveryEngine:
    """Manages stake sizing and recovery state transitions across trading signals."""
    def __init__(self, mode="MODE_GALE1_CAP", base_stake=1.0, payout_ratio=0.85):
        self.mode = mode
        self.sheet_tab = TAB_MAPPING.get(mode, "Realtime_Bot_Trades")
        self.base_stake = base_stake
        self.payout_ratio = payout_ratio
        
        self.in_recovery = False
        self.recovery_trades_left = 0
        
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.dojis = 0
        self.total_staked = 0.0
        self.total_profit = 0.0
        self.peak_equity = 0.0
        self.max_drawdown = 0.0
        self.current_equity = 0.0

    def get_stake(self, gale_level=0):
        if self.mode == "MODE_GALE1_CAP":
            if gale_level > 1: return 0.0
            return round(self.base_stake * (2.0 ** gale_level), 2)
        elif self.mode == "MODE_DISTRIBUTED":
            stake_multiplier = 2.2 if (self.recovery_trades_left > 0) else 1.0
            return round((self.base_stake * stake_multiplier) * (2.0 ** gale_level), 2)
        elif self.mode == "MODE_FLAT_RECOVERY":
            if self.in_recovery:
                if gale_level > 0: return 0.0
                return round(self.base_stake * 3.5, 2)
            return round(self.base_stake * (2.0 ** gale_level), 2)
        elif self.mode == "MODE_AGGRESSIVE_CROSS":
            effective_base = self.base_stake * 8.5 if self.in_recovery else self.base_stake
            return round(effective_base * (2.0 ** gale_level), 2)
        else:
            return round(self.base_stake * (2.0 ** gale_level), 2)

    def process_result(self, result_str, stake_amount, gale_level, raw_row_profit=None, raw_row_amount=1.0):
        self.total_trades += 1
        self.total_staked += stake_amount
        res_upper = str(result_str).upper()

        if raw_row_profit is not None and raw_row_amount > 0:
            profit_multiplier = stake_amount / raw_row_amount
            profit = round(raw_row_profit * profit_multiplier, 2)
        else:
            if res_upper == 'WIN':
                profit = round(stake_amount * self.payout_ratio, 2)
            elif res_upper == 'LOSS':
                profit = -round(stake_amount, 2)
            else:
                profit = 0.0

        self.total_profit += profit
        self.current_equity += profit

        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        drawdown = self.peak_equity - self.current_equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        if res_upper == 'WIN':
            self.wins += 1
            if self.in_recovery: self.in_recovery = False
            if self.recovery_trades_left > 0: self.recovery_trades_left -= 1
        elif res_upper == 'LOSS':
            self.losses += 1
            if self.mode == "MODE_GALE1_CAP" and gale_level == 1:
                self.in_recovery = True
            elif gale_level >= 2:
                self.in_recovery = True
                self.recovery_trades_left = 3
        else:
            self.dojis += 1

        return profit


def wma(series, period):
    return series.rolling(period).apply(
        lambda x: ((x * np.arange(1, period + 1)).sum()) / np.arange(1, period + 1).sum(),
        raw=True
    )

def analyze_amiq_strategy(candles_data):
    if not candles_data or len(candles_data) < 40:
        return None, None
    df = pd.DataFrame(candles_data)
    df['sma_fast'] = df['close'].rolling(window=1).mean()
    df['sma_slow'] = df['close'].rolling(window=34).mean()
    df['buffer1'] = df['sma_fast'] - df['sma_slow']
    df['buffer2'] = wma(df['buffer1'], 4)

    df['rsi'] = df['close'].diff().apply(lambda x: max(x, 0)).rolling(14).mean() / (
        df['close'].diff().abs().rolling(14).mean() + 1e-9
    ) * 100

    amiq_call = (
        df['buffer1'].iloc[-1] > df['buffer2'].iloc[-1] and
        df['buffer1'].iloc[-2] < df['buffer2'].iloc[-2]
    )

    amiq_put = (
        df['buffer1'].iloc[-1] < df['buffer2'].iloc[-1] and
        df['buffer1'].iloc[-2] > df['buffer2'].iloc[-2]
    )

    signal = None
    if amiq_call: signal = "CALL"
    elif amiq_put: signal = "PUT"

    features = {
        'signal_source': 'realtime_bot3',
        'is_realtime_bot': True,
        'strategy': 'AM_IQ_Simple',
        'rsi': round(float(df['rsi'].iloc[-1]), 2) if 'rsi' in df and not pd.isna(df['rsi'].iloc[-1]) else 50.0,
        'buffer1': round(float(df['buffer1'].iloc[-1]), 6),
        'buffer2': round(float(df['buffer2'].iloc[-1]), 6),
        'close': float(df['close'].iloc[-1])
    }
    return signal, features


async def run_standalone_bot3(assets=None, base_amount=1.0, timeframe=60, scan_interval=5, active_mode="MODE_GALE1_CAP"):
    if not assets: assets = DEFAULT_ASSETS

    active_tab = TAB_MAPPING.get(active_mode, "Realtime_Bot_Gale1_Cap")

    logger.info("=" * 75)
    logger.info("  REALTIME BOT 3 - LIVE TRADER (ACTIVE MODE: %s)", active_mode)
    logger.info("=" * 75)
    logger.info(f"Assets Monitored: {assets}")
    logger.info(f"Base Trade Amount: ${base_amount:.2f}")
    logger.info(f"Active Live Execution Mode: {active_mode} -> Logging to Tab: '{active_tab}'")
    logger.info("Multi-Tab Sheets Sync: Logging all 5 recovery models to separate tabs")
    logger.info("=" * 75)

    email = os.getenv("IQ_EMAIL") or os.getenv("email")
    password = os.getenv("IQ_PASSWORD") or os.getenv("password")

    if not email or not password:
        logger.error("❌ Missing IQ_EMAIL or IQ_PASSWORD in .env file.")
        return

    api = IQOptionAPI(email=email, password=password)
    logger.info("Connecting to IQ Option...")
    await api._connect()

    balance = api.get_current_account_balance()
    logger.info(f"Connected! Balance: ${balance:.2f} ({api.account_mode.upper()})")

    if api.account_mode.lower() != "practice":
        logger.info("Switching to PRACTICE account for safety...")
        api.switch_account("practice")
        await asyncio.sleep(2)
        balance = api.get_current_account_balance()
        logger.info(f"Switched! Practice Balance: ${balance:.2f}")

    recovery_engines = {mode: RecoveryEngine(mode=mode, base_stake=base_amount) for mode in TAB_MAPPING.keys()}
    main_engine = recovery_engines[active_mode]

    max_gales_to_use = 1 if active_mode == "MODE_GALE1_CAP" else config.max_martingale_gales
    cycle = 0

    try:
        while True:
            cycle += 1
            print(f"\n--- RealtimeBot3 Scan Cycle #{cycle} [{datetime.now().strftime('%H:%M:%S')}] [{active_mode}] ---", flush=True)

            for asset in assets:
                try:
                    candles = api.get_candle_history(asset, count=280, timeframe=timeframe)
                    if not candles: continue

                    signal, features = analyze_amiq_strategy(candles)

                    if signal:
                        expiry = 5 if timeframe >= 300 else 1
                        active_stake = main_engine.get_stake(gale_level=0)

                        logger.info(f"⚡ SIGNAL DETECTED: {asset} -> {signal} | Active Stake: ${active_stake:.2f} ({active_mode})")

                        trade_res = await run_trade(
                            api, asset, signal.lower(), expiry, active_stake,
                            max_gales=max_gales_to_use, features=features, ignore_sl_tp=True
                        )
                        
                        prof = float(trade_res.get('profit', 0.0)) if isinstance(trade_res, dict) else 0.0
                        res_str = trade_res.get('result', 'WIN' if prof > 0 else 'LOSS' if prof < 0 else 'EQUAL') if isinstance(trade_res, dict) else ("WIN" if prof > 0 else "LOSS")
                        gales_used = trade_res.get('gale_level', trade_res.get('gales', 0)) if isinstance(trade_res, dict) else 0

                        for mode_id, tab_name in TAB_MAPPING.items():
                            eng = recovery_engines[mode_id]
                            stk = eng.get_stake(gale_level=gales_used)
                            if stk > 0:
                                calc_profit = eng.process_result(res_str, stk, gales_used, prof, active_stake)
                                trade_info = {
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'asset': asset,
                                    'direction': signal,
                                    'amount': stk,
                                    'expiry': expiry * 60,
                                    'result': res_str,
                                    'profit': calc_profit,
                                    'gale_level': gales_used,
                                    'signal_source': f"realtime_bot3 ({mode_id})",
                                    'is_realtime_bot': True,
                                    'rsi': features.get('rsi', 50.0),
                                    'close': features.get('close', 0.0)
                                }
                                try:
                                    gsheet_logger.log_trade(trade_info, worksheet_name=tab_name)
                                except Exception as e:
                                    logger.error(f"Error logging to sheet [{tab_name}]: {e}")

                except Exception as e:
                    logger.error(f"Error analyzing {asset}: {e}")

            await asyncio.sleep(scan_interval)

    except KeyboardInterrupt:
        logger.info("Stopped RealtimeBot3 loop.")


def run_csv_simulation(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at '{csv_path}'")
        return

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df_clean = df.dropna(subset=['Result']).copy()
    df_clean['Timestamp'] = pd.to_datetime(df_clean['Timestamp'], format='mixed')
    df_clean = df_clean.sort_values(by='Timestamp').reset_index(drop=True)

    print("\n" + "=" * 90)
    print(f" REALTIME BOT 3 - RECOVERY ENGINE REPORT ({len(df_clean)} Trades)")
    print("=" * 90)

    summary_results = []

    for mode_id, tab_name in TAB_MAPPING.items():
        engine = RecoveryEngine(mode=mode_id, base_stake=1.0)
        for idx, row in df_clean.iterrows():
            gale = float(row['Gale Level']) if pd.notnull(row['Gale Level']) else 0
            res = str(row['Result']).upper()

            if mode_id == "MODE_GALE1_CAP" and gale > 1: continue

            stake = engine.get_stake(gale_level=gale)
            if stake > 0:
                engine.process_result(
                    result_str=res,
                    stake_amount=stake,
                    gale_level=gale,
                    raw_row_profit=float(row['Profit']),
                    raw_row_amount=float(row['Amount']) if float(row['Amount']) > 0 else 1.0
                )

        decided = engine.wins + engine.losses
        win_rate = (engine.wins / decided * 100) if decided > 0 else 0
        roi = (engine.total_profit / engine.total_staked * 100) if engine.total_staked > 0 else 0

        summary_results.append({
            "Mode": mode_id,
            "Sheet Tab": tab_name,
            "Trades": engine.total_trades,
            "Wins": engine.wins,
            "Losses": engine.losses,
            "Win Rate %": round(win_rate, 2),
            "Total Staked": round(engine.total_staked, 2),
            "Net Profit": round(engine.total_profit, 2),
            "ROI %": round(roi, 2),
            "Max Drawdown": round(engine.max_drawdown, 2)
        })

    results_df = pd.DataFrame(summary_results)
    print("\n" + results_df.to_string(index=False))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealtimeBot3 Standalone Trader & Multi-Tab Recovery Logger")
    parser.add_argument("--backtest", action="store_true", help="Run backtest report on trade dataset CSV")
    parser.add_argument("--csv", type=str, default=r"D:\GUSHTEC\Downloads\Iq_trade_history - Realtime_Bot_Trades.csv", help="Path to trade CSV file")
    parser.add_argument("--mode", type=str, default="MODE_GALE1_CAP", choices=list(TAB_MAPPING.keys()), help="Active recovery mode to execute live")
    args = parser.parse_args()

    if args.backtest:
        run_csv_simulation(args.csv)
    else:
        asyncio.run(run_standalone_bot3(active_mode=args.mode))
