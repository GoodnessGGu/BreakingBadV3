"""
run_unified_bot.py - Master Launcher for BreakingBad V3 Unified Bot

Launches concurrently in ONE unified process:
  1. Telegram Bot Control Center (@WalterAWbot)
  2. Multi-Channel Telethon Listener:
     - CallistoFx Live (Dual-zone BUY/SELL with candle confirmation)
     - Gold Pips Hunter (Direct XAU/Gold signals + Breakeven + CloseAll)
     - Polycarp VIP Room (5m Blitz Options execution on IQ Option MCP)
  3. Multi-Instrument Autonomous ICT / SMC Strategy Engine (15M sweep + FVG + Breakeven)

Completely eliminates SQLite session file locking and process conflicts.
"""

import os
import sys
import time
import signal
import logging
import asyncio
import argparse
from dotenv import load_dotenv

sys.path.append(os.getcwd())
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
load_dotenv()

from clients.forex_mcp_client import IQForexMCPClient
from clients.blitz_mcp_client import IQBlitzMCPClient
from copiers.callisto_copier import CallistoCopier
from copiers.gold_pips_copier import GoldPipsCopier
from copiers.gsociety_copier import GSocietyCopier
from copiers.polycarp_copier import PolycarpCopier
from copiers.channel_manager import ChannelManager
from strategies.ict_engine import ICTStrategyEngine
from bot.telegram_controller import TelegramTradingBot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MasterLauncher")

async def main():
    parser = argparse.ArgumentParser(description="BreakingBad V3 Master Bot Launcher")
    parser.add_argument("--account", default=os.getenv("ACCOUNT_TYPE", "training"), choices=["training", "regular"])
    parser.add_argument("--lots", type=float, default=float(os.getenv("DEFAULT_LOTS", "1.0")))
    parser.add_argument("--leverage", type=int, default=int(os.getenv("DEFAULT_LEVERAGE", "100")))
    parser.add_argument("--blitz-stake", type=float, default=float(os.getenv("BLITZ_STAKE", "2.0")))
    parser.add_argument("--enable-polycarp", action="store_true", default=os.getenv("ENABLE_POLYCARP", "false").lower() == "true")
    parser.add_argument("--ict-symbol", default=os.getenv("ICT_SYMBOL", "XAUUSD"))
    parser.add_argument("--lookback-mins", type=int, default=int(os.getenv("LOOKBACK_MINS", "240")))
    args = parser.parse_args()

    token = os.getenv("TELEGRAM_TOKEN")
    admin_id = os.getenv("ADMIN_ID") or 6420777416

    if not token:
        logger.error("❌ Missing TELEGRAM_TOKEN in .env")
        return

    logger.info("=" * 65)
    logger.info("🚀 INITIALIZING BREAKINGBAD V3 TRADING ECOSYSTEM")
    logger.info("=" * 65)

    # 1. Initialize MCP Clients
    forex_mcp = IQForexMCPClient(base_url="https://marginal-cfd.mcp.iqoption.com")
    blitz_mcp = IQBlitzMCPClient(base_url="https://blitz-options.mcp.iqoption.com")

    logger.info("Connecting to IQ Option Marginal CFD MCP...")
    if not forex_mcp.initialize():
        logger.error("Failed to initialize Forex/CFD MCP Client. Check IQ_AI_TOKEN.")
        return

    logger.info("Connecting to IQ Option Blitz Options MCP...")
    if not blitz_mcp.initialize():
        logger.warning("Failed to initialize Blitz MCP Client. Polycarp copier will be disabled.")

    # 2. Query Balances
    fx_bal = forex_mcp.get_training_balance() if args.account == "training" else forex_mcp.get_real_balance()
    fx_bid = fx_bal.get("balance_id") if fx_bal else None
    logger.info(f"Connected Forex/CFD Balance ID: {fx_bid} | Equity: ${fx_bal.get('equity', 0.0):.2f}")

    blitz_bal = blitz_mcp.get_training_balance() if args.account == "training" else blitz_mcp.get_real_balance()
    blitz_bid = blitz_bal.get("balance_id") if blitz_bal else None
    logger.info(f"Connected Blitz Balance ID: {blitz_bid} | Amount: ${blitz_bal.get('amount', 0.0):.2f}")

    # 3. Create Copiers
    callisto = CallistoCopier(
        mcp_client=forex_mcp,
        lots=args.lots,
        leverage=args.leverage
    )
    callisto.set_balance(fx_bid, args.account)

    gold_pips = GoldPipsCopier(
        mcp_client=forex_mcp,
        lots=args.lots,
        leverage=args.leverage
    )
    gold_pips.set_balance(fx_bid, args.account)

    gsociety = GSocietyCopier(
        mcp_client=forex_mcp,
        lots=args.lots,
        leverage=args.leverage
    )
    gsociety.set_balance(fx_bid, args.account)

    polycarp = PolycarpCopier(
        blitz_mcp=blitz_mcp,
        channel_id=-1002551711564,
        stake_amount=args.blitz_stake,
        enabled=args.enable_polycarp
    )
    polycarp.set_balance(blitz_bid, args.account)

    # 4. Register in Channel Manager
    channel_mgr = ChannelManager(
        session_name="user_desktop_session",
        lookback_mins=args.lookback_mins
    )
    channel_mgr.register_copier(callisto)
    channel_mgr.register_copier(gold_pips)
    channel_mgr.register_copier(gsociety)
    channel_mgr.register_copier(polycarp)

    # 5. Create ICT Strategy Engine
    ict_engine = ICTStrategyEngine(
        mcp_client=forex_mcp,
        symbol=args.ict_symbol,
        account_type=args.account,
        lots=args.lots,
        leverage=args.leverage,
        rr_ratio=2.0
    )
    ict_engine.set_balance(fx_bid, args.account)

    # 6. Create Telegram Bot Controller
    tg_bot = TelegramTradingBot(
        token=token,
        admin_id=admin_id,
        channel_mgr=channel_mgr,
        ict_engine=ict_engine,
        forex_mcp=forex_mcp,
        blitz_mcp=blitz_mcp,
        lots=args.lots,
        leverage=args.leverage,
        blitz_stake=args.blitz_stake
    )
    tg_bot.account_type = args.account

    # 7. Start Concurrent Execution
    logger.info("=" * 65)
    logger.info("🟢 ALL MODULES INITIALIZED — STARTING MASTER EVENT LOOP")
    logger.info("=" * 65)

    try:
        await asyncio.gather(
            tg_bot.start(),
            channel_mgr.start(),
            ict_engine.run_loop()
        )
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Shutting down master bot...")
    finally:
        await tg_bot.stop()
        await channel_mgr.stop()
        logger.info("All components stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete.")
