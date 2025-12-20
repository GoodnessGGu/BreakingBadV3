# first_main.py
import sys
import logging
import asyncio
import re
from datetime import datetime, date
from collections import defaultdict
from iqclient import IQOptionAPI  # you renamed it, so we keep IQOptionAPI

# Configure console to support emojis
sys.stdout.reconfigure(encoding="utf-8")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)

# Example raw signals (replace or load from file)
SIGNALS_TEXT = """
🟢 01:38 - EURUSD-OTC CALL M1
🟢 01:38 - AUDUSD-OTC CALL M1
🔴 01:40 - EURJPY-OTC PUT M2
🟢 01:42 - NZDUSD-OTC CALL M1
""".strip()


def parse_signals(text: str):
    """
    Parse raw signal text into structured list of dicts.
    """
    signals = []
    pattern = re.compile(
        r"(\d{1,2}:\d{2})\s*-\s*([A-Z0-9\-/]+)\s+(CALL|PUT)\s+M(\d+)",
        re.IGNORECASE
    )

    for line in text.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        time_str, asset, direction, expiry = m.groups()
        hh, mm = map(int, time_str.split(":"))
        scheduled_dt = datetime.combine(date.today(), datetime.min.time()).replace(
            hour=hh, minute=mm, second=0
        )
        signals.append({
            "time": scheduled_dt,
            "asset": asset,
            "direction": direction.lower(),
            "expiry": int(expiry),
            "line": line.strip(),
        })
    return sorted(signals, key=lambda x: x["time"])


def run_trade(api, asset, direction, expiry, amount, max_gales=2):
    """
     Place trade with digital → binary fallback, and run martingale loop.
     This runs in a separate thread when launched from asyncio.
     """
    current_amount = amount
    for gale in range(max_gales + 1):
        # Try digital trade
        success, order_id = api.execute_digital_option_trade(
            asset, current_amount, direction, expiry=expiry
        )

        # If digital fails, skip this trade
        if not success:
            logger.error(f"❌ Failed to place trade on {asset} (Digital only)")
            return

        logger.info(
            f"🎯 Placed trade: {asset} {direction.upper()} ${current_amount} (Expiry {expiry}m)"
        )
        pnl_ok, pnl = api.get_trade_outcome(order_id, expiry=expiry)
        balance = api.get_current_account_balance()

        if pnl_ok and pnl > 0:
            if gale == 0:
                logger.info(
                    f"✅ Direct WIN on {asset} | Profit: ${pnl:.2f} | Balance: ${balance:.2f}"
                )
            else:
                logger.info(
                    f"🔥 WIN after Gale {gale} on {asset} | Profit: ${pnl:.2f} | Balance: ${balance:.2f}"
                )
            return
        else:
            logger.warning(
                f"⚠️ LOSS on {asset} | Attempt {gale} | PnL: {pnl} | Balance: ${balance:.2f}"
            )
            current_amount *= 2  # Martingale

    logger.error(f"💀 Lost all attempts (Direct + {max_gales} Gales) on {asset}")


async def main():
    print("\n📡 Initializing API and Establishing Connection")
    api = IQOptionAPI()
    api._connect()
    logger.info("Connected ✅")
    logger.info(f"Starting balance: ${api.get_current_account_balance()}")

    signals = parse_signals(SIGNALS_TEXT)
    if not signals:
        logger.info("No valid signals found.")
        return

    # Group signals by scheduled time
    grouped = defaultdict(list)
    for sig in signals:
        grouped[sig["time"]].append(sig)

    # Process groups in chronological order
    for sched_time in sorted(grouped.keys()):
        now = datetime.now()
        delay = (sched_time - now).total_seconds()
        if delay > 0:
            logger.info(
                f"\n⏳ Waiting {int(delay)}s until {sched_time.strftime('%H:%M')} "
                f"for {len(grouped[sched_time])} signal(s)..."
            )
            await asyncio.sleep(delay)

        logger.info(
            f"\n🚀 Executing {len(grouped[sched_time])} signal(s) scheduled at {sched_time.strftime('%H:%M')}"
        )
        for sig in grouped[sched_time]:
            logger.info(f"📊 Signal: {sig['line']}")

        # Fire all trades immediately in parallel (each in its own thread)
        await asyncio.gather(
            *(
                asyncio.to_thread(
                    run_trade, api, sig["asset"], sig["direction"], sig["expiry"], 1
                )
                for sig in grouped[sched_time]
            )
        )

    logger.info("\n✅ All signals processed.")
    logger.info(f"Final balance: ${api.get_current_account_balance()}")


if __name__ == "__main__":
    asyncio.run(main())