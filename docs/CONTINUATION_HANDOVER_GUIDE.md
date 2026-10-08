# BreakingBad V3 — Session Continuation & Project Handover Guide

> **Generated on:** October 8, 2026  
> **Repository:** `https://github.com/GoodnessGGu/BreakingBadV3`  
> **Target Audience:** Project developer and any AI assistant continuing this project on a new PC.

---

## 1. Executive Summary & Current Status

BreakingBad V3 is an institutional-grade, multi-strategy algorithmic trading bot and Telegram signal copier deployed live on **Railway**. It is currently executing in **Prop Firm Mode** on an active demo/training balance ($5,000 capital) to validate rule compliance, profit expectancy, and automated risk controls before launching a funded prop challenge.

### Key Metrics At Handover (October 8, 2026)
* **Initial Capital:** `$5,000.00`
* **Current Account Balance:** **`$4,989.66`** (Drawdown: `-0.21%` — virtually untouched)
* **Prop Firm Mode Realized PnL (Oct 4–7):** **`+$23.75`** (59 trades)
* **Clean Technical Strategy PnL (Excluding Patched Loop Bug):** **`+$129.05` (+2.58% in 3 days, Profit Factor: 4.66)**
* **US100 (NAS100) Performance:** **4/4 Wins (100% Win Rate) | `+$102.15` Profit**
* **Google Sheet Records:** 108 total synchronized trades with row coloring and PnL breakdown.
* **Production Status:** Live on Railway (`4d777625-4677-4e79-9ac8-117acd2c44e4` — `SUCCESS`).

---

## 2. Infrastructure & Hosting

### Railway Production Environment
* **Project ID:** `603331ca-ba20-4352-a398-2f7fa2c4f86a`
* **Service Name:** `BreakingBadV3`
* **CLI Command:**
  ```powershell
  # Check status & logs
  railway status
  railway logs -n 50

  # Deploy new commit
  railway up -d -m "deploy description"
  ```

### Active Broker & Account
* **Broker:** IQ Option Forex / CFD / Digital via `forex_mcp_client.py`
* **Active Prop Balance ID:** `1237481096`
* **Assets Active:**
  * **US100 (NAS100 CFD):** Asset ID `1471` (`mcfd.1471`)
  * **Gold (XAU/USD CFD):** Asset ID `74` (`mcfd.74`)
  * **Currencies & Cryptos:** EUR/USD, GBP/USD, BTC, etc.

### Google Sheets Real-Time Journal
* **Spreadsheet ID:** `1vYkKqMXbqubtcf_Srim2kwtIwvC6EjI3dlYD6mGrPy4`
* **Worksheet:** `Forex_Margin_Trades`
* **Logger:** [`gsheet_logger.py`](file:///C:/Users/GushEx/Documents/IQOPTIONS%20BOT/BreakingBadV3/gsheet_logger.py)
* **Fields:** Timestamp, Asset, Side, Lots, Entry Price, Stop Loss, Take Profit, Exit Price, PnL, Pips, Risk/Reward, Exit Reason, Position ID, Balance/Equity.

---

## 3. Autonomous Strategy Engines & Architecture

The bot runs via [`run_unified_bot.py`](file:///C:/Users/GushEx/Documents/IQOPTIONS%20BOT/BreakingBadV3/run_unified_bot.py) which orchestrates the following modular engines:

1. **[`TrendlineStrategyEngine`](file:///C:/Users/GushEx/Documents/IQOPTIONS%20BOT/BreakingBadV3/strategies/trendline_engine.py)**:
   * **Setup:** 15-Minute 3rd Touch Trendline Bounce filtered by 50 EMA trend direction.
   * **Assets:** US100 (`asset_id: 1471`) and Gold (`asset_id: 74`).
   * **Execution:** Dual-Ticket partial profit system:
     * *Leg 1 (50% lot):* Banks profit at `+1.0R`.
     * *Leg 2 (50% lot):* Shifts Stop Loss to Breakeven when TP1 hits, targets `+2.5R` runner.
   * **Safety Controls:** 45-minute post-loss cooldown, 3-loss circuit breaker lockout.
   * **Trade Logging:** Integrated dual-ticket logging to Google Sheets with position ID deduplication.

2. **[`ICTEngine`](file:///C:/Users/GushEx/Documents/IQOPTIONS%20BOT/BreakingBadV3/strategies/ict_engine.py)**:
   * **Setup:** Institutional Silver Bullet, Fair Value Gap (FVG), and Change in State of Delivery (CISD) liquidity sweeps.
   * **Assets:** XAUUSD, XAGUSD, Crypto, Forex majors.

3. **[`SNDStrategyEngine`](file:///C:/Users/GushEx/Documents/IQOPTIONS%20BOT/BreakingBadV3/strategies/snd_engine.py)**:
   * **Setup:** Multi-timeframe Supply & Demand order blocks with volume confirmation and imbalance retests.

4. **[`CRTStrategyEngine`](file:///C:/Users/GushEx/Documents/IQOPTIONS%20BOT/BreakingBadV3/strategies/crt_engine.py)**:
   * **Setup:** Candle Range Theory daily and session high/low manipulation runs.

5. **[`NewsStraddleEngine`](file:///C:/Users/GushEx/Documents/IQOPTIONS%20BOT/BreakingBadV3/strategies/news_straddle_engine.py)**:
   * **Setup:** High-impact economic news breakout straddles (NFP, CPI, FOMC). Arms pending dual bracket stops 2 minutes before release, auto-cancels unhit side.

6. **Telegram Signal Copiers (`copiers/`)**:
   * Copiers active for Callisto, Kingmahn Tribe, GSociety, Gold Pips, and Polycarp.
   * Telethon parses incoming signals, validates TP/SL formatting, and executes via MCP.

7. **Risk Manager ([`bot/prop_firm_manager.py`](file:///C:/Users/GushEx/Documents/IQOPTIONS%20BOT/BreakingBadV3/bot/prop_firm_manager.py))**:
   * Enforces max 0.5%–1.0% risk per trade.
   * Hard stop at 4.0% daily drawdown.
   * Hard stop at 8.0% total account drawdown.

8. **Safety Net Logger ([`bot/telegram_controller.py`](file:///C:/Users/GushEx/Documents/IQOPTIONS%20BOT/BreakingBadV3/bot/telegram_controller.py))**:
   * Hooks into `handle_position_closed`. If any broker trade closes that was not captured by an engine (e.g. manual closure, broker margin call, or unmapped ticket), the controller automatically logs it to `Forex_Margin_Trades`.

---

## 4. Key Investigations & Bug Fixes Resolved

1. **The Gold Rapid Re-entry Bug (Fixed)**:
   * *Issue:* Prior to Oct 7, during a sharp Gold drop, `trendline_engine.py` re-entered immediately after every stop-out, taking 32 micro-losses (-$105.30 total).
   * *Fix:* Enforced a strict 45-minute cooldown timer per symbol after any loss, plus a 3-consecutive-loss hard circuit breaker.
2. **Missing Google Sheets Logging on US100 & Other Engines (Fixed)**:
   * *Issue:* `trendline_engine.py`, `crt_engine.py`, and `snd_engine.py` lacked Google Sheets dispatch calls.
   * *Fix:* Fully wired dual-ticket Google Sheets logging across all engines and backfilled all 36 historical unlogged trades.
3. **Prop Firm Gatekeeper Hook (Fixed)**:
   * Enforced sizing validation directly at the broker interface so no runaway position size can be executed.

---

## 5. How to Set Up and Continue on a New PC

### Step 1: Clone the Repository
```bash
git clone https://github.com/GoodnessGGu/BreakingBadV3.git
cd BreakingBadV3
```

### Step 2: Set Up Python Environment
```bash
# Recommended Python version: 3.10 or 3.11
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### Step 3: Required Credential Files
Ensure the following files are copied or configured on the new PC:
1. **`credentials.json`** / **`utils/service_account.json`**: Google Cloud Service Account credentials with read/write access to Google Sheets.
2. **`.env`**: Contains Telegram Bot Token, Telegram API ID/Hash, Railway tokens, and broker credentials.
3. **`*.session`**: Telethon session files (`user_desktop_session.session`) for Telegram listener authentication.

### Step 4: Verification & Diagnostic Commands
Run these diagnostic scripts from the root directory:
```bash
# 1. Check live trade performance & Google Sheet sync
python scratch/analyze_performance.py

# 2. Check currently open broker positions
python scratch/check_open_pos.py

# 3. Audit broker history vs Google Sheet
python scratch/sync_and_audit_all_trades.py
```

### Step 5: Interacting with Railway
```bash
# Login to Railway CLI
railway login

# Link the project
railway link 603331ca-ba20-4352-a398-2f7fa2c4f86a

# View live logs
railway logs
```

---

## 6. Next Steps & Recommended Roadmap

1. **Continue Observation in Prop Firm Mode:** Let the bot run on the current training balance on Railway. Monitor the newly added 45-min cooldown and universal Google Sheet logging.
2. **Funded Prop Firm Challenge Transition:** Once 1-2 weeks of clean, green performance are confirmed, connect live credentials from the chosen prop firm (FundedNext, FTMO, or TradeLocker-based firms).
3. **Multi-Timeframe CRT & SND Expansion:** Activate additional liquid pairs (EURUSD, GBPUSD) once backtests confirm win-rates > 60%.

---

*Refer to [`docs/CHAT_HISTORY_EXPORT.md`](file:///C:/Users/GushEx/Documents/IQOPTIONS%20BOT/BreakingBadV3/docs/CHAT_HISTORY_EXPORT.md) for the complete turn-by-turn conversation log and debugging transcript.*
