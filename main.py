import asyncio
import logging
import sys
from iqclient import IQOptionAPI, run_trade
from signal_parser import load_signals, parse_signals
from datetime import datetime
from collections import defaultdict
from settings import DEFAULT_TRADE_AMOUNT, MAX_MARTINGALE_GALES, MARTINGALE_MULTIPLIER, PAUSED





async def process_signals(api, raw_text: str):
    signals = parse_signals(raw_text)
    if not signals:
        logger.info("⚠️ No valid signals found.")
        return

    grouped = defaultdict(list)
    for sig in signals:
        grouped[sig["time"]].append(sig)

    for sched_time in sorted(grouped.keys()):
        now = datetime.now()
        delay = (sched_time - now).total_seconds()
        if delay > 0:
            logger.info(f"⏳ Waiting {int(delay)}s until {sched_time.strftime('%H:%M')} for {len(grouped[sched_time])} signal(s)...")
            await asyncio.sleep(delay)

        logger.info(f"🚀 Executing {len(grouped[sched_time])} signal(s) at {sched_time.strftime('%H:%M')}")
        await asyncio.gather(*[
            run_trade(api, s["asset"], s["direction"], s["expiry"], DEFAULT_TRADE_AMOUNT)
            for s in grouped[sched_time]
        ])


async def main():
    logger.info("📡 Connecting to IQ Option API...")
    api = IQOptionAPI()
    await api._connect()
    logger.info(f"✅ Connected | Balance: ${api.get_current_account_balance():.2f}")

    raw_signals = load_signals("signals.txt")
    await process_signals(api, raw_signals)


if __name__ == "__main__":
    asyncio.run(main())
