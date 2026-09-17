import sys, asyncio, os, re
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\GushEx\Documents\IQOPTIONS BOT\BreakingBadV3")
from dotenv import load_dotenv
load_dotenv()
from telethon import TelegramClient

API_ID   = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")
CALLISTO = -1002848189989
DEBUG_IDS = [34110, 34293]

async def main():
    client = TelegramClient("user_desktop_session", API_ID, API_HASH)
    await client.connect()
    entity = await client.get_entity(CALLISTO)
    for msg_id in DEBUG_IDS:
        msgs = await client.get_messages(entity, ids=msg_id)
        m = msgs if not isinstance(msgs, list) else msgs[0]
        print(f"=== MSG #{m.id} ===")
        print(f"m.text repr:\n{repr(m.text)}")
        print(f"\nm.message repr:\n{repr(m.message)}")
        upper = m.text.upper() if m.text else ""
        pattern = r"(?:NEW\s+)?(BUY|SELL)\s+ZONE[:\s]+([0-9]+(?:\.[0-9]+)?)\s*[-\u2013]\s*([0-9]+(?:\.[0-9]+)?)"
        matches = list(re.finditer(pattern, upper))
        print(f"\nRegex matches ({len(matches)}):")
        for mm in matches:
            print(f"  group(0)={mm.group(0)} group(1)={mm.group(1)} group(2)={mm.group(2)} group(3)={mm.group(3)}")
        print()
    await client.disconnect()

asyncio.run(main())
