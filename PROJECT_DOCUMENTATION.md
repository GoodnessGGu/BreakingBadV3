# BreakingBadV3 - System Architecture & Progress Report

## 1. Executive Summary
**BreakingBadV3** is an advanced, AI-augmented automated binary options trading system designed for IQ Option. It integrates multi-strategy technical analysis, multi-stage machine learning filters (XGBoost & Neural Networks), counterfactual off-policy learning for skipped signals, dynamic payout-adjusted risk management, real-time Google Sheets synchronization with automated color-coded retraining logs, and multi-channel Telegram monitoring.

---

## 2. Visual Architecture & Dashboard Overview

![AI Trading Bot Dashboard Illustration](C:\Users\GushEx\.gemini\antigravity-cli\brain\bb7d6c48-dde8-453e-8628-0fdb3514d0ff\bot_architecture_dashboard_1786366754169.jpg)

```
+-----------------------------------------------------------------------------------+
|                                 BREAKINGBAD V3                                    |
+-----------------------------------------------------------------------------------+
                                          |
     +------------------------------------+-----------------------------------+
     |                                    |                                   |
     v                                    v                                   v
+------------------+             +--------------------+              +--------------------+
| Technical Engine |             |  AI & ML Pipeline  |              | Risk Safeguards    |
| - AM_IQ (MA)     |             | - XGBoost Win Model|              | - Dynamic Sizing   |
| - Sniper Pattern |             | - Loss Detector    |              | - Latency Cutoff   |
| - 3-Candle S/R   |             | - Neural Net Gate  |              | - Session/ATR Gate |
+------------------+             +--------------------+              +--------------------+
     |                                    |                                   |
     +------------------------------------+-----------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        COUNTERFACTUAL & LOGGING ENGINE                            |
|  - Real-time Trade Execution (IQ Option WebSocket)                                |
|  - Virtual Counterfactual Logging for Skipped Signals (SQLite & CSV)             |
|  - Multi-Tab Google Sheets Sync (Trades, Realtime_Bot_Trades, Bot2_Trades)        |
+-----------------------------------------------------------------------------------+
```

---

## 3. Core Architectural Components

### A. Technical Signal Generation (`strategies.py`)
1. **AM_IQ Strategy**: Fast SMA (1) vs Slow SMA (34) buffer crossed with 4-period WMA.
2. **Sniper Pattern**: 4-candle price action reversal detection.
3. **3-Candle Support/Resistance**: Price proximity to dynamic S/R levels aligned with EMA 200 trend.

### B. Multi-Stage AI & Neural Network Gate Pipeline
1. **Gate 1 - Loss Detector**: Evaluates market feature vector against trained XGBoost model. Blocks signals with predicted loss probability $P(\text{Loss}) > 0.65$.
2. **Gate 2 - Win Predictor**: Validates directional probability before entry approval.
3. **Gate 3 - Adaptive NN Threshold**: Dynamically adjusts acceptance threshold based on recent 10-trade session win rate and suppresses entries during erratic ATR volatility spikes ($> 1.8\times \text{Average}$).

### C. Counterfactual Off-Policy AI Learning Engine
- **Tracking Decisions Made & Skipped**: Evaluates both executed real trades and rejected/skipped signals.
- **Virtual Trade Post-Expiry Evaluation**: When a filter blocks a signal, a background task tracks price movement post-expiry to record whether the signal `WOULD HAVE WON` or `WOULD HAVE LOST`.
- **Database & Dataset Exporter**: Virtual outcomes are saved into SQLite `virtual_trades` table and exported to `live_trade_feedback.csv` for continuous AI model re-training.
- **Filter Accuracy Metrics**: Tracks live Saved Losses vs Missed Wins to compute real-time filter efficiency statistics.

### D. Dynamic Bet Sizing & Latency Protection (`trade.py`)
- **Payout-Adjusted Bet Sizing**: Calculates recovery bet stake $S = \frac{\text{Loss}}{\text{Payout}}$ to optimize risk/reward.
- **Latency Safeguard**: Enforces a strict 2.5-second entry delay cutoff (`target_time`). Trades delayed beyond 2.5s are rejected as `LATENCY_REJECTED`.

### E. Google Sheets Integration & Multi-Tab Routing (`gsheet_logger.py`)

![Google Sheets Color-Coded Live Retraining Log](C:\Users\GushEx\.gemini\antigravity-cli\brain\bb7d6c48-dde8-453e-8628-0fdb3514d0ff\google_sheets_live_log_1786366775609.jpg)

- **Modern Authentication**: Built using `google-auth` Service Account credentials.
- **20 AI Retraining Features**: Records timestamp, asset, direction, amount, expiry, result, PnL, gale level, source, RSI, ADX, BB_Width, ATR, EMA_Trend, Orderbook_Ratio, Win_Rate_Gate, Loss_Prob, NN_Prob, Entry_Latency, and Close_Price.
- **Automatic Row Coloring**:
  - **WIN / VIRTUAL_WIN**: Soft Green (`#D4EDDA`)
  - **LOSS / VIRTUAL_LOSS**: Soft Red (`#F8D7DA`)
  - **DRAW / TIE**: Soft Yellow (`#FFF3CD`)
- **Multi-Tab Routing**:
  - Channel / Telegram signals ➔ **`Trades`** tab
  - `realtime_bot.py` ➔ **`Realtime_Bot_Trades`** tab
  - `realtime_bot2.py` ➔ **`Realtime_Bot2_Trades`** tab

---

## 4. Key Accomplishments & Progress Summary

| Feature / Upgrade | Description | Status |
| :--- | :--- | :--- |
| **Counterfactual Learning** | Virtual trade logging for skipped signals to train AI on rejected setups | ✅ Implemented & Verified |
| **Dynamic Bet Sizing** | Payout-adjusted stake calculation & balance percentage modes | ✅ Implemented & Verified |
| **Latency Safeguard** | 2.5-second max entry delay enforcement | ✅ Implemented & Verified |
| **Standalone Test Runners** | `realtime_bot.py` & `realtime_bot2.py` (No Telegram required) | ✅ Implemented & Verified |
| **Simple AM_IQ Bot** | `realtime_bot2.py` testing pure MA crossover without modifying existing files | ✅ Implemented & Verified |
| **Google Sheets Reconnection** | Upgraded to modern `google-auth` & authenticated service account | ✅ Connected & Verified |
| **Multi-Tab GSheets Sync** | Automated routing to `Trades`, `Realtime_Bot_Trades`, and `Realtime_Bot2_Trades` | ✅ Connected & Verified |
| **Color-Coded Rows** | Soft Green for Wins, Soft Red for Losses on Google Sheets | ✅ Connected & Verified |
| **20 Retraining Columns** | Expanded feature vector recording for offline AI model updates | ✅ Connected & Verified |

---

## 5. Entry Points & Execution Guide

### 1. Standalone Real-Time Trader (Full AI + Technicals)
Runs continuous scans, executes practice trades, logs counterfactual virtual trades, and syncs to Google Sheets tab `Realtime_Bot_Trades`:
```bash
python realtime_bot.py
```

### 2. Standalone Simple AM_IQ Strategy Tester
Runs pure AM_IQ crossover strategy and logs to Google Sheets tab `Realtime_Bot2_Trades`:
```bash
python realtime_bot2.py
```

### 3. Smart Trail Signals Bot (Higher Expiries & Gale 1 Capping)
Translates TradingView's 'Smart Trail Signals NO CONDITIONS' indicator with dynamic ATR volatility trailing stops, 3-minute expiries, breakout exhaustion filter, and 15-minute loss cooldown. Logs to Google Sheets tab `Smart_Trail_Trades`:
```bash
python smart_trail_bot.py --expiry 3 --max-gales 1
```

### 4. Telegram Control Bot
Launches Telegram bot for remote command control and signal monitoring:
```bash
python telegram_bot.py
```

### 5. Test Google Sheets Connection
```bash
python test_gsheet_conn.py
```
