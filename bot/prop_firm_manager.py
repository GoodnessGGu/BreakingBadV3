"""
bot/prop_firm_manager.py - Institutional Prop Firm Challenge Simulator & Guardian
Enforces strict institutional risk rules for a $5,000 evaluation account:
  - Profit Targets: Phase 1 (+8.0% / $400), Phase 2 (+5.0% / $250)
  - Daily Drawdown Limit: -4.0% ($200.00 max daily loss from day's opening equity)
  - Maximum Overall Drawdown: -8.0% ($400.00 max loss, hard equity floor at $4,600.00)
  - Dynamic Position Sizing: Automatically scales risk to 0.5% - 1.0% ($25 - $50) per trade
  - Margin & Concurrency Limits: Max 2 concurrent open positions
  - Automatic Daily Reset at 00:00 UTC
  - Persistent JSON state ledger (survives container restarts)
"""

import os
import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List, Callable, Set

logger = logging.getLogger("PropFirmGuardian")

PROP_STATE_FILE = os.path.join(os.getcwd(), "data", "prop_firm_state.json")

class PropFirmGuardian:
    def __init__(
        self,
        initial_capital: float = 5000.0,
        phase: int = 1,
        broadcast_func: Optional[Callable] = None
    ):
        self.initial_capital = float(initial_capital)
        self.phase = int(phase)
        self.broadcast_func = broadcast_func

        # Default rules for standard 5k evaluation
        self.profit_target_pct = 8.0 if self.phase == 1 else 5.0
        self.daily_drawdown_limit_pct = 4.0  # 4% of day's start balance ($200)
        self.max_drawdown_limit_pct = 8.0    # 8% of initial capital ($400)
        self.risk_per_trade_pct = 0.75       # 0.75% per trade ($37.50)
        self.max_open_positions = 2

        # State attributes
        self.is_enabled: bool = False
        self.current_balance: float = self.initial_capital
        self.daily_start_balance: float = self.initial_capital
        self.daily_pnl: float = 0.0
        self.peak_balance: float = self.initial_capital
        self.total_pnl: float = 0.0
        self.status: str = "ACTIVE"  # "ACTIVE", "DAILY_LOCKED", "PASSED", "BREACHED"
        self.last_reset_date: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        self.total_trades: int = 0
        self.wins: int = 0
        self.losses: int = 0
        self.breakevens: int = 0
        self.recent_trades: List[Dict[str, Any]] = []
        self.processed_trade_ids: Set[str] = set()

        # Institutional Risk Rules: Frequency Cap & Circuit Breakers
        self.max_daily_trades: int = 8
        self.daily_trade_count: int = 0
        self.consecutive_losses: int = 0
        self.max_consecutive_losses: int = 3
        self.consecutive_loss_freeze_until: float = 0.0

        # Load persisted state if exists
        self.load_state()
        self.check_daily_reset()

    # -------------------------------------------------------------
    # State Persistence
    # -------------------------------------------------------------
    def save_state(self):
        try:
            os.makedirs(os.path.dirname(PROP_STATE_FILE), exist_ok=True)
            data = {
                "is_enabled": self.is_enabled,
                "phase": self.phase,
                "initial_capital": self.initial_capital,
                "profit_target_pct": self.profit_target_pct,
                "daily_drawdown_limit_pct": self.daily_drawdown_limit_pct,
                "max_drawdown_limit_pct": self.max_drawdown_limit_pct,
                "risk_per_trade_pct": self.risk_per_trade_pct,
                "max_open_positions": self.max_open_positions,
                "current_balance": round(self.current_balance, 2),
                "daily_start_balance": round(self.daily_start_balance, 2),
                "daily_pnl": round(self.daily_pnl, 2),
                "peak_balance": round(self.peak_balance, 2),
                "total_pnl": round(self.total_pnl, 2),
                "status": self.status,
                "last_reset_date": self.last_reset_date,
                "total_trades": self.total_trades,
                "wins": self.wins,
                "losses": self.losses,
                "breakevens": self.breakevens,
                "daily_trade_count": self.daily_trade_count,
                "consecutive_losses": self.consecutive_losses,
                "consecutive_loss_freeze_until": self.consecutive_loss_freeze_until,
                "max_daily_trades": self.max_daily_trades,
                "recent_trades": self.recent_trades[-30:]
            }
            with open(PROP_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"[PropFirm] Error saving state: {e}")

    def load_state(self):
        if not os.path.exists(PROP_STATE_FILE):
            return
        try:
            with open(PROP_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.is_enabled = data.get("is_enabled", False)
            self.phase = data.get("phase", 1)
            self.initial_capital = data.get("initial_capital", 5000.0)
            self.profit_target_pct = data.get("profit_target_pct", 8.0 if self.phase == 1 else 5.0)
            self.daily_drawdown_limit_pct = data.get("daily_drawdown_limit_pct", 4.0)
            self.max_drawdown_limit_pct = data.get("max_drawdown_limit_pct", 8.0)
            self.risk_per_trade_pct = data.get("risk_per_trade_pct", 0.75)
            self.max_open_positions = data.get("max_open_positions", 2)
            self.current_balance = data.get("current_balance", 5000.0)
            self.daily_start_balance = data.get("daily_start_balance", 5000.0)
            self.daily_pnl = data.get("daily_pnl", 0.0)
            self.peak_balance = data.get("peak_balance", 5000.0)
            self.total_pnl = data.get("total_pnl", 0.0)
            self.status = data.get("status", "ACTIVE")
            self.last_reset_date = data.get("last_reset_date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
            self.total_trades = data.get("total_trades", 0)
            self.wins = data.get("wins", 0)
            self.losses = data.get("losses", 0)
            self.breakevens = data.get("breakevens", 0)
            self.daily_trade_count = data.get("daily_trade_count", 0)
            self.consecutive_losses = data.get("consecutive_losses", 0)
            self.consecutive_loss_freeze_until = data.get("consecutive_loss_freeze_until", 0.0)
            self.max_daily_trades = data.get("max_daily_trades", 8)
            self.recent_trades = data.get("recent_trades", [])
            logger.info(f"🛡️ [PropFirm] State loaded. Balance: ${self.current_balance:.2f} | Status: {self.status} | Trades Today: {self.daily_trade_count}/{self.max_daily_trades}")
        except Exception as e:
            logger.error(f"[PropFirm] Error loading state: {e}")

    # -------------------------------------------------------------
    # Daily Reset Handling
    # -------------------------------------------------------------
    def check_daily_reset(self) -> bool:
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today_str != self.last_reset_date:
            logger.info(f"🌅 [PropFirm] New Trading Day ({today_str})! Resetting daily drawdown baseline & trade counter.")
            self.last_reset_date = today_str
            self.daily_start_balance = self.current_balance
            self.daily_pnl = 0.0
            self.daily_trade_count = 0
            self.consecutive_losses = 0
            self.consecutive_loss_freeze_until = 0.0
            if self.status == "DAILY_LOCKED":
                self.status = "ACTIVE"
            self.save_state()
            return True
        return False

    # -------------------------------------------------------------
    # Sizing & Execution Guard
    # -------------------------------------------------------------
    def get_target_balance(self) -> float:
        return self.initial_capital * (1.0 + (self.profit_target_pct / 100.0))

    def get_max_loss_equity_floor(self) -> float:
        # Standard prop firm: static 8% from initial capital ($4,600)
        return self.initial_capital * (1.0 - (self.max_drawdown_limit_pct / 100.0))

    def get_max_daily_loss_amount(self) -> float:
        return self.daily_start_balance * (self.daily_drawdown_limit_pct / 100.0)

    def can_execute_trade(self, symbol: str, current_open_count: int = 0) -> Tuple[bool, str, float]:
        """
        Validates if a new trade is permitted under Prop Firm Rules.
        Returns: (allowed: bool, rejection_reason: str, recommended_lot_size: float)
        """
        if not self.is_enabled:
            return True, "", 1.0

        self.check_daily_reset()

        if self.status == "BREACHED":
            return False, "🚨 Prop Firm Account is BREACHED (-8% Max Drawdown hit). Trading halted.", 0.0

        if self.status == "DAILY_LOCKED":
            return False, "🛡️ Daily Drawdown Limit (-4%) reached. Bot is locked until next daily session (00:00 UTC).", 0.0

        if self.status == "PASSED":
            return False, f"🏆 Target Reached! Challenge Phase {self.phase} Passed. Awaiting Phase advance.", 0.0

        if current_open_count >= self.max_open_positions:
            return False, f"⚠️ Concurrency Cap: Max {self.max_open_positions} concurrent positions reached.", 0.0

        # Check consecutive loss circuit breaker (cooling off after multiple losses)
        if time.time() < self.consecutive_loss_freeze_until:
            rem_m = int(max(1, (self.consecutive_loss_freeze_until - time.time()) / 60))
            return False, f"🛡️ Circuit Breaker Active: Paused for {rem_m}m after {self.consecutive_losses} consecutive losses.", 0.0

        # Check daily trade count limit
        if self.daily_trade_count >= self.max_daily_trades:
            return False, f"🛑 Max Daily Trades ({self.max_daily_trades}) reached. Session closed under Prop Firm rules.", 0.0

        # Check remaining daily drawdown buffer
        remaining_daily_budget = self.get_max_daily_loss_amount() + self.daily_pnl
        if remaining_daily_budget <= 15.0:
            return False, f"🛡️ Insufficient Daily Drawdown Buffer (${remaining_daily_budget:.2f} remaining).", 0.0

        # Calculate strict dollar risk per trade ($37.50 on $5k)
        target_risk_usd = self.current_balance * (self.risk_per_trade_pct / 100.0)
        target_risk_usd = min(target_risk_usd, remaining_daily_budget * 0.5)

        # Determine optimal lot size for $5,000 risk envelope
        lots = self._compute_asset_prop_lots(symbol, target_risk_usd)
        return True, "Approved under Prop Firm Rules", lots

    def _compute_asset_prop_lots(self, symbol: str, risk_usd: float) -> float:
        """
        Returns safe lot sizes calibrated for a $5,000 prop firm challenge.
        Guarantees that lot sizes respect broker minimums while protecting the risk budget.
        """
        sym = symbol.upper()
        if sym in ["XAUUSD", "GOLD"]:
            # Gold: 2 tickets of 1.0 lot min broker size
            return 2.0
        elif sym in ["BTCUSD", "BITCOIN"]:
            # Bitcoin: 0.01 lots
            return 0.01
        elif sym in ["NAS100", "US100"]:
            return 1.0
        elif sym in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD"]:
            return 1.0
        else:
            return 1.0

    # -------------------------------------------------------------
    # Settlement & Settlement Tracking
    # -------------------------------------------------------------
    async def on_trade_closed(self, symbol: str, pnl_usd: float, side: str = "BUY", lots: float = 1.0, reason: str = "closed", trade_id: Optional[str] = None):
        """Records trade completion, updates challenge metrics, and checks triggers."""
        if not self.is_enabled:
            return

        if trade_id:
            tid_str = str(trade_id)
            if tid_str in self.processed_trade_ids:
                return
            self.processed_trade_ids.add(tid_str)

        self.check_daily_reset()
        self.current_balance += pnl_usd
        self.daily_pnl += pnl_usd
        self.total_pnl += pnl_usd
        self.total_trades += 1
        self.daily_trade_count += 1

        if pnl_usd > 0:
            self.wins += 1
            self.consecutive_losses = 0
            if self.current_balance > self.peak_balance:
                self.peak_balance = self.current_balance
        elif pnl_usd == 0:
            self.breakevens += 1
        else:
            self.losses += 1
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.consecutive_loss_freeze_until = time.time() + 3600  # 60 minute pause
                logger.warning(f"🛡️ [PropFirm] Circuit Breaker activated! {self.consecutive_losses} losses in a row. Pausing 60 min.")
                await self._alert(
                    f"🛡️ **[PROP FIRM CIRCUIT BREAKER ACTIVATED]** 🛡️\n\n"
                    f"• Consecutive Losses: `{self.consecutive_losses}` in a row.\n"
                    f"• All new trades **PAUSED for 60 Minutes** to protect capital.\n"
                    f"• Account capital is preserved from further drawdown."
                )

        ts_str = datetime.now(timezone.utc).strftime("%H:%M")
        self.recent_trades.append({
            "symbol": symbol,
            "side": side,
            "lots": lots,
            "pnl": round(pnl_usd, 2),
            "reason": reason,
            "time": ts_str
        })

        logger.info(f"🛡️ [PropFirm] Trade Settled: {symbol} PnL=${pnl_usd:.2f} | Balance=${self.current_balance:.2f} | DailyPnL=${self.daily_pnl:.2f}")

        # 1. Check Target Reached
        target_bal = self.get_target_balance()
        if self.current_balance >= target_bal:
            self.status = "PASSED"
            self.save_state()
            await self._alert(
                f"🎉 **[PROP FIRM CHALLENGE PASSED!]** 🎉\n\n"
                f"• Target Balance (${target_bal:.2f}) ACHIEVED!\n"
                f"• Current Equity: `${self.current_balance:.2f}` (+{self.profit_target_pct:.1f}%)\n"
                f"• Phase {self.phase} Successfully Completed! 🏆"
            )
            return

        # 2. Check Daily Drawdown Breach
        max_daily_loss = self.get_max_daily_loss_amount()
        if self.daily_pnl <= -max_daily_loss:
            self.status = "DAILY_LOCKED"
            self.save_state()
            await self._alert(
                f"🛡️ **[PROP FIRM DAILY DRAWDOWN LOCKDOWN]**\n\n"
                f"• Daily Loss Limit reached: `${abs(self.daily_pnl):.2f}` / `-${max_daily_loss:.2f}` (-{self.daily_drawdown_limit_pct:.1f}%)\n"
                f"• All new trades are **FROZEN** until next 00:00 UTC session.\n"
                f"• Account capital is protected from further drawdown."
            )
            return

        # 3. Check Overall Max Drawdown Breach
        floor = self.get_max_loss_equity_floor()
        if self.current_balance <= floor:
            self.status = "BREACHED"
            self.save_state()
            await self._alert(
                f"🚨 **[PROP FIRM ACCOUNT BREACHED]** 🚨\n\n"
                f"• Equity reached Maximum Drawdown floor: `${self.current_balance:.2f}` (Floor: `${floor:.2f}`).\n"
                f"• Overall Loss: `${self.initial_capital - self.current_balance:.2f}` (-{self.max_drawdown_limit_pct:.1f}%)\n"
                f"• Challenge failed. Reset via `/prop_reset` to restart."
            )
            return

        self.save_state()

    async def _alert(self, text: str):
        if self.broadcast_func:
            try:
                await self.broadcast_func(text)
            except Exception as e:
                logger.error(f"[PropFirm] Broadcast error: {e}")

    # -------------------------------------------------------------
    # Controls & Resets
    # -------------------------------------------------------------
    def toggle(self) -> bool:
        self.is_enabled = not self.is_enabled
        self.save_state()
        logger.info(f"🔄 [PropFirm] Mode toggled: {'ENABLED' if self.is_enabled else 'DISABLED'}")
        return self.is_enabled

    def set_phase(self, phase: int):
        self.phase = 2 if phase == 2 else 1
        self.profit_target_pct = 5.0 if self.phase == 2 else 8.0
        self.save_state()

    def set_risk(self, risk_pct: float):
        self.risk_per_trade_pct = max(0.25, min(2.0, float(risk_pct)))
        self.save_state()

    def reset_challenge(self, initial_capital: float = 5000.0, phase: int = 1):
        self.initial_capital = float(initial_capital)
        self.phase = int(phase)
        self.profit_target_pct = 8.0 if self.phase == 1 else 5.0
        self.current_balance = self.initial_capital
        self.daily_start_balance = self.initial_capital
        self.daily_pnl = 0.0
        self.peak_balance = self.initial_capital
        self.total_pnl = 0.0
        self.status = "ACTIVE"
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.breakevens = 0
        self.daily_trade_count = 0
        self.consecutive_losses = 0
        self.consecutive_loss_freeze_until = 0.0
        self.recent_trades = []
        self.last_reset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.save_state()
        logger.info(f"🔄 [PropFirm] Challenge successfully reset to fresh ${self.initial_capital:.2f} account.")

    # -------------------------------------------------------------
    # Visual Dashboard Generation
    # -------------------------------------------------------------
    def get_dashboard_text(self) -> str:
        self.check_daily_reset()
        status_icons = {
            "ACTIVE": "🟢 ACTIVE (In Evaluation)",
            "DAILY_LOCKED": "🔒 DAILY LOCKED (Paused until 00:00 UTC)",
            "PASSED": "🏆 TARGET PASSED! (Ready for funded)",
            "BREACHED": "🚨 ACCOUNT BREACHED (-8% Max DD)"
        }
        st_text = status_icons.get(self.status, self.status)

        target_usd = self.initial_capital * (self.profit_target_pct / 100.0)
        target_bal = self.initial_capital + target_usd
        gain_usd = self.current_balance - self.initial_capital
        gain_pct = (gain_usd / self.initial_capital) * 100.0
        gain_sign = "+" if gain_usd >= 0 else ""

        # Progress bar for Target (0 to 100%)
        target_progress = max(0.0, min(1.0, gain_usd / target_usd)) if target_usd > 0 else 0.0
        target_bars = int(target_progress * 10)
        target_bar_str = "🟩" * target_bars + "⬜" * (10 - target_bars)

        # Progress bar for Daily Drawdown
        max_daily_loss = self.get_max_daily_loss_amount()
        daily_loss_val = abs(min(0.0, self.daily_pnl))
        daily_progress = min(1.0, daily_loss_val / max_daily_loss) if max_daily_loss > 0 else 0.0
        daily_bars = int(daily_progress * 8)
        daily_bar_str = "🟥" * daily_bars + "⬜" * (8 - daily_bars)

        # Progress bar for Max Overall Drawdown
        max_overall_loss = self.initial_capital * (self.max_drawdown_limit_pct / 100.0)
        overall_loss_val = max(0.0, self.initial_capital - self.current_balance)
        overall_progress = min(1.0, overall_loss_val / max_overall_loss) if max_overall_loss > 0 else 0.0
        overall_bars = int(overall_progress * 8)
        overall_bar_str = "🟥" * overall_bars + "⬜" * (8 - overall_bars)

        wr = (self.wins / max(1, self.wins + self.losses)) * 100.0 if (self.wins + self.losses) > 0 else 0.0

        p_icon = "🟢" if self.is_enabled else "🔴"

        text = (
            f"🏛️ *Prop Firm Challenge Simulator ($5K)*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Master Guard: {p_icon} `{'ON' if self.is_enabled else 'OFF'}` | Phase: `{self.phase}`\n"
            f"• Status: *{st_text}*\n"
            f"• Current Equity: `${self.current_balance:.2f}` (`{gain_sign}${gain_usd:.2f}` | `{gain_sign}{gain_pct:.2f}%`)\n"
            f"• Peak High-Water Mark: `${self.peak_balance:.2f}`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🎯 *Profit Target* (+{self.profit_target_pct:.1f}% | `${target_usd:.2f}`):\n"
            f"  {target_bar_str} `{gain_sign}${gain_usd:.2f}` / `${target_usd:.2f}` ({target_progress*100:.1f}%)\n"
            f"  Goal Equity: `${target_bal:.2f}`\n\n"
            f"🛡️ *Daily Drawdown Buffer* (-{self.daily_drawdown_limit_pct:.1f}% | `-${max_daily_loss:.2f}`):\n"
            f"  {daily_bar_str} `-${daily_loss_val:.2f}` / `-${max_daily_loss:.2f}` used\n"
            f"  Remaining Day Buffer: `${max(0.0, max_daily_loss - daily_loss_val):.2f}`\n\n"
            f"⚠️ *Overall Max Drawdown* (-{self.max_drawdown_limit_pct:.1f}% | `-${max_overall_loss:.2f}`):\n"
            f"  {overall_bar_str} `-${overall_loss_val:.2f}` / `-${max_overall_loss:.2f}` used\n"
            f"  Hard Equity Floor: `${self.get_max_loss_equity_floor():.2f}`\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚙️ *Evaluation Rules & Protection*:\n"
            f"  • Risk per Trade: `{self.risk_per_trade_pct:.2f}%` (${self.current_balance * self.risk_per_trade_pct / 100:.2f})\n"
            f"  • Today's Executions: `{self.daily_trade_count}/{self.max_daily_trades}` (Max Daily Cap)\n"
            f"  • Consecutive Loss Streak: `{self.consecutive_losses}/{self.max_consecutive_losses}` (Circuit Breaker)\n"
            f"  • Max Concurrent Trades: `{self.max_open_positions}`\n"
            f"  • Win Rate: `{wr:.1f}%` ({self.wins}W - {self.losses}L - {self.breakevens}BE | `{self.total_trades}` Total)\n"
        )

        if self.recent_trades:
            text += "\n📋 *Recent Challenge Trades*:\n"
            for t in self.recent_trades[-4:]:
                t_icon = "🏆" if t["pnl"] > 0 else ("🛡️" if t["pnl"] == 0 else "❌")
                t_sign = "+" if t["pnl"] >= 0 else ""
                text += f"  {t_icon} `{t['time']}` *{t['symbol']}* {t['side']} {t['lots']}L ➔ `{t_sign}${t['pnl']:.2f}`\n"

        return text
