import asyncio
from bot.news_engine import EconomicNewsEngine

async def main():
    ne = EconomicNewsEngine()
    ok = await ne.fetch_calendar(force=True)
    print(f"Calendar fetch success: {ok} | Total events: {len(ne.events)}")
    high_events = [e for e in ne.events if e['impact'] == 'High']
    print(f"High-impact events count: {len(high_events)}")
    for e in high_events:
        print(f"{e['dt_utc']} UTC | {e['country']} | {e['title']}")

if __name__ == "__main__":
    asyncio.run(main())
