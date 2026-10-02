"""
tests/snd_imbalance_backtest.py
Standalone Quantitative Backtester for Institutional Supply & Demand + Imbalance + Efficient Ranges.

ISOLATED TEST MODULE:
- Does NOT touch or modify any production bot files or Railway deployment.
- Tests multiple assets (Gold, EURUSD, BTC) across multiple configurations:
  1. Entry Style: Imbalance Proximal vs 50% Consequent Encroachment (CE)
  2. Risk-Reward (R:R): 1:2.0, 1:2.5, 1:3.0
  3. Breakeven: At +1.0R vs Pure R:R
  4. Premium/Discount Filtering: Enabled vs Disabled
  5. Displacement Strength: 1.2x ATR vs 1.5x ATR
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Dict, Any, List

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def fetch_asset_data(ticker: str, period: str = "60d", interval: str = "15m") -> pd.DataFrame:
    print(f"📥 Downloading {ticker} ({period}, {interval})...")
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].astype(float)
    
    # Calculate indicators
    df['ATR'] = calculate_atr(df, period=14)
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df.dropna(inplace=True)
    return df

class SNDImbalanceBacktester:
    def __init__(self, df: pd.DataFrame, asset_name: str, config: Dict[str, Any]):
        self.df = df
        self.asset_name = asset_name
        self.cfg = config
        
        # Unpack configuration
        self.entry_style = config.get("entry_style", "CE_50") # "PROXIMAL" or "CE_50"
        self.rr_ratio = config.get("rr_ratio", 2.5)
        self.use_breakeven = config.get("use_breakeven", True)
        self.be_trigger_r = config.get("be_trigger_r", 1.0)
        self.use_premium_discount = config.get("use_premium_discount", True)
        self.disp_mult = config.get("disp_mult", 1.2) # Multiplier of ATR for displacement
        self.sl_buf_mult = config.get("sl_buf_mult", 0.2)
        self.max_zone_hold_bars = config.get("max_zone_hold_bars", 80) # Max bars zone is valid before expiring

    def run(self) -> Dict[str, Any]:
        highs = self.df['High'].values
        lows = self.df['Low'].values
        opens = self.df['Open'].values
        closes = self.df['Close'].values
        atrs = self.df['ATR'].values
        ema100 = self.df['EMA100'].values
        times = self.df.index
        n = len(self.df)

        active_trade = None
        pending_zones = [] # Active unmitigated S&D + Imbalance zones
        trades = []

        for i in range(20, n):
            h, l, o, c = highs[i], lows[i], opens[i], closes[i]
            cur_atr = atrs[i]

            # ----------------------------------------------------
            # 1. Manage Active Open Trade
            # ----------------------------------------------------
            if active_trade is not None:
                side = active_trade["side"]
                entry = active_trade["entry_price"]
                sl = active_trade["sl"]
                tp = active_trade["tp"]
                risk = active_trade["risk"]

                # Check Breakeven ratchet
                if self.use_breakeven and not active_trade["is_be"]:
                    if side == "BUY" and h >= entry + (risk * self.be_trigger_r):
                        active_trade["sl"] = entry
                        active_trade["is_be"] = True
                    elif side == "SELL" and l <= entry - (risk * self.be_trigger_r):
                        active_trade["sl"] = entry
                        active_trade["is_be"] = True

                # Check SL / TP
                hit_tp = (h >= tp) if side == "BUY" else (l <= tp)
                hit_sl = (l <= sl) if side == "BUY" else (h >= sl)

                if hit_tp and hit_sl:
                    # In bar conflict: conservative assume stopped out or check open direction
                    hit_tp = False

                if hit_tp:
                    r_outcome = self.rr_ratio
                    pnl_r = r_outcome
                    trades.append({
                        "entry_time": active_trade["entry_time"],
                        "exit_time": times[i],
                        "side": side,
                        "entry": entry,
                        "exit": tp,
                        "r_result": pnl_r,
                        "outcome": "WIN",
                        "bars_held": i - active_trade["entry_idx"]
                    })
                    active_trade = None
                elif hit_sl:
                    r_outcome = 0.0 if active_trade["is_be"] else -1.0
                    outcome_str = "BE" if active_trade["is_be"] else "LOSS"
                    trades.append({
                        "entry_time": active_trade["entry_time"],
                        "exit_time": times[i],
                        "side": side,
                        "entry": entry,
                        "exit": sl,
                        "r_result": r_outcome,
                        "outcome": outcome_str,
                        "bars_held": i - active_trade["entry_idx"]
                    })
                    active_trade = None

            # ----------------------------------------------------
            # 2. Check Pending Zones for Mitigation & Entry Trigger
            # ----------------------------------------------------
            if active_trade is None and pending_zones:
                valid_zones = []
                for zone in pending_zones:
                    # Expire old zones
                    if i - zone["bar_idx"] > self.max_zone_hold_bars:
                        continue

                    z_side = zone["side"]
                    entry_lvl = zone["entry_level"]
                    sl_lvl = zone["sl"]

                    # Check if price invalidated the zone before entering (traded beyond distal line)
                    if z_side == "BUY" and l < zone["distal"]:
                        continue # Invalidated
                    if z_side == "SELL" and h > zone["distal"]:
                        continue # Invalidated

                    # Check if price reached the entry level (Range Rebalancing / Efficiency Tap)
                    tapped = (l <= entry_lvl <= h)
                    if tapped and active_trade is None:
                        # Enter trade
                        risk = abs(entry_lvl - sl_lvl)
                        if risk > 0:
                            tp_lvl = entry_lvl + (risk * self.rr_ratio) if z_side == "BUY" else entry_lvl - (risk * self.rr_ratio)
                            active_trade = {
                                "side": z_side,
                                "entry_price": entry_lvl,
                                "sl": sl_lvl,
                                "tp": tp_lvl,
                                "risk": risk,
                                "is_be": False,
                                "entry_idx": i,
                                "entry_time": times[i]
                            }
                        # Zone is now mitigated/efficient -> Do NOT reuse (Freshness rule)
                        continue

                    valid_zones.append(zone)
                pending_zones = valid_zones

            # ----------------------------------------------------
            # 3. Detect New S&D Base + Imbalance (3-Candle Formation)
            # ----------------------------------------------------
            # Sequence:
            # Candle i-2: S&D Base origin candle
            # Candle i-1: Displacement candle (creates the imbalance)
            # Candle i  : Confirmation candle establishing the FVG boundary
            
            c_base = i - 2
            c_disp = i - 1
            c_conf = i

            # Measure Displacement Body
            disp_body = abs(closes[c_disp] - opens[c_disp])
            is_strong_disp = disp_body >= (self.disp_mult * atrs[c_disp])

            if is_strong_disp:
                # 50-bar Dealing Range for Premium/Discount calculation
                lookback = 40
                range_high = np.max(highs[max(0, i-lookback):i+1])
                range_low = np.min(lows[max(0, i-lookback):i+1])
                equilibrium = (range_high + range_low) * 0.5

                # ------------------------------------------------
                # A. Bullish Setup: Drop-Base-Rally + BISI Imbalance
                # ------------------------------------------------
                # Displacement was green
                if closes[c_disp] > opens[c_disp]:
                    # Bullish Imbalance: Candle c_conf Low > Candle c_base High
                    gap_size = lows[c_conf] - highs[c_base]
                    if gap_size > (0.1 * atrs[c_conf]):
                        # Imbalance confirmed!
                        fvg_proximal = lows[c_conf]
                        fvg_distal = highs[c_base]
                        fvg_ce = (fvg_proximal + fvg_distal) * 0.5
                        
                        base_low = min(lows[c_base], lows[c_disp])
                        sl_level = base_low - (self.sl_buf_mult * cur_atr)
                        chosen_entry = fvg_ce if self.entry_style == "CE_50" else fvg_proximal

                        # Trend & Discount Filters
                        trend_ok = closes[i] > ema100[i]
                        discount_ok = (not self.use_premium_discount) or (chosen_entry <= equilibrium)

                        if trend_ok and discount_ok:
                            pending_zones.append({
                                "side": "BUY",
                                "bar_idx": i,
                                "entry_level": chosen_entry,
                                "sl": sl_level,
                                "distal": base_low,
                                "fvg_ce": fvg_ce,
                                "created_time": times[i]
                            })

                # ------------------------------------------------
                # B. Bearish Setup: Rally-Base-Drop + SIBI Imbalance
                # ------------------------------------------------
                # Displacement was red
                elif closes[c_disp] < opens[c_disp]:
                    # Bearish Imbalance: Candle c_conf High < Candle c_base Low
                    gap_size = lows[c_base] - highs[c_conf]
                    if gap_size > (0.1 * atrs[c_conf]):
                        # Imbalance confirmed!
                        fvg_proximal = highs[c_conf]
                        fvg_distal = lows[c_base]
                        fvg_ce = (fvg_proximal + fvg_distal) * 0.5

                        base_high = max(highs[c_base], highs[c_disp])
                        sl_level = base_high + (self.sl_buf_mult * cur_atr)
                        chosen_entry = fvg_ce if self.entry_style == "CE_50" else fvg_proximal

                        # Trend & Premium Filters
                        trend_ok = closes[i] < ema100[i]
                        premium_ok = (not self.use_premium_discount) or (chosen_entry >= equilibrium)

                        if trend_ok and premium_ok:
                            pending_zones.append({
                                "side": "SELL",
                                "bar_idx": i,
                                "entry_level": chosen_entry,
                                "sl": sl_level,
                                "distal": base_high,
                                "fvg_ce": fvg_ce,
                                "created_time": times[i]
                            })

        # Calculate performance statistics
        return self._compute_metrics(trades)

    def _compute_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_trades = len(trades)
        if total_trades == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "net_r": 0.0,
                "max_drawdown_r": 0.0,
                "wins": 0,
                "losses": 0,
                "breakevens": 0,
                "avg_r_per_trade": 0.0
            }

        wins = [t for t in trades if t["outcome"] == "WIN"]
        losses = [t for t in trades if t["outcome"] == "LOSS"]
        breakevens = [t for t in trades if t["outcome"] == "BE"]

        win_rate = (len(wins) / total_trades) * 100.0
        gross_win_r = sum(t["r_result"] for t in wins)
        gross_loss_r = abs(sum(t["r_result"] for t in losses))
        net_r = gross_win_r - gross_loss_r
        profit_factor = (gross_win_r / gross_loss_r) if gross_loss_r > 0 else 99.0

        # Equity Curve and Max Drawdown calculation in R-units
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in trades:
            equity += t["r_result"]
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd

        return {
            "total_trades": total_trades,
            "wins": len(wins),
            "losses": len(losses),
            "breakevens": len(breakevens),
            "win_rate": round(win_rate, 1),
            "gross_win_r": round(gross_win_r, 2),
            "gross_loss_r": round(gross_loss_r, 2),
            "net_r": round(net_r, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_r": round(max_dd, 2),
            "avg_r_per_trade": round(net_r / total_trades, 2)
        }

def run_multi_asset_backtest():
    assets = {
        "Gold (XAUUSD)": "GC=F",
        "Euro / USD": "EURUSD=X",
        "Bitcoin (BTCUSD)": "BTC-USD"
    }

    # Fetch data for all assets
    datasets = {}
    for name, ticker in assets.items():
        try:
            datasets[name] = fetch_asset_data(ticker, period="60d", interval="15m")
        except Exception as e:
            print(f"Error downloading {name}: {e}")

    # Define Configurations to Test
    configurations = {
        "Config A (Baseline: 50% CE, 1:2.5 R:R, BE +1R, Strict P/D)": {
            "entry_style": "CE_50",
            "rr_ratio": 2.5,
            "use_breakeven": True,
            "be_trigger_r": 1.0,
            "use_premium_discount": True,
            "disp_mult": 1.2
        },
        "Config B (Aggressive Proximal Entry, 1:2.0 R:R, BE +1R)": {
            "entry_style": "PROXIMAL",
            "rr_ratio": 2.0,
            "use_breakeven": True,
            "be_trigger_r": 1.0,
            "use_premium_discount": True,
            "disp_mult": 1.2
        },
        "Config C (High R:R Runner: 50% CE, 1:3.0 R:R, BE +1R)": {
            "entry_style": "CE_50",
            "rr_ratio": 3.0,
            "use_breakeven": True,
            "be_trigger_r": 1.0,
            "use_premium_discount": True,
            "disp_mult": 1.2
        },
        "Config D (Pure Sniper: 50% CE, 1:2.5 R:R, NO BE, High Displacement 1.5x)": {
            "entry_style": "CE_50",
            "rr_ratio": 2.5,
            "use_breakeven": False,
            "use_premium_discount": True,
            "disp_mult": 1.5
        },
        "Config E (High Volume: Proximal Entry, 1:2.0 R:R, No P/D Filter)": {
            "entry_style": "PROXIMAL",
            "rr_ratio": 2.0,
            "use_breakeven": True,
            "be_trigger_r": 1.0,
            "use_premium_discount": False,
            "disp_mult": 1.2
        }
    }

    print("\n" + "=" * 90)
    print("🚀 INSTITUTIONAL SUPPLY & DEMAND + IMBALANCE + EFFICIENT RANGES BACKTEST RESULTS")
    print("=" * 90)

    summary_rows = []

    for cfg_name, cfg in configurations.items():
        print(f"\n📊 Testing: {cfg_name}")
        for asset_name, df in datasets.items():
            bt = SNDImbalanceBacktester(df, asset_name, cfg)
            res = bt.run()
            res["asset"] = asset_name
            res["config"] = cfg_name
            summary_rows.append(res)
            print(f"   • {asset_name:<18} | Trades: {res['total_trades']:<3} | Win Rate: {res['win_rate']:>5.1f}% | Profit Factor: {res['profit_factor']:>5.2f} | Net R: {res['net_r']:>+6.2f}R | Max DD: {res['max_drawdown_r']:>5.2f}R")

    # Aggregate by Config
    print("\n" + "=" * 90)
    print("🏆 CONFIGURATION AGGREGATE RANKING ACROSS ALL ASSETS")
    print("=" * 90)

    df_res = pd.DataFrame(summary_rows)
    for cfg_name in configurations.keys():
        sub = df_res[df_res['config'] == cfg_name]
        tot_trades = sub['total_trades'].sum()
        avg_wr = sub['win_rate'].mean()
        tot_net_r = sub['net_r'].sum()
        avg_pf = sub['profit_factor'].mean()
        avg_dd = sub['max_drawdown_r'].mean()
        print(f"⭐ {cfg_name}")
        print(f"   Total Trades: {tot_trades} | Avg Win Rate: {avg_wr:.1f}% | Combined Net R: {tot_net_r:+.2f}R | Avg Profit Factor: {avg_pf:.2f} | Avg Max DD: {avg_dd:.2f}R\n")

if __name__ == "__main__":
    run_multi_asset_backtest()
