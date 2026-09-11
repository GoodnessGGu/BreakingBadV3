import sys
import asyncio
from telethon import TelegramClient
from telethon.tl.types import Channel

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import os
from dotenv import load_dotenv

load_dotenv()
API_ID = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")
SESSION_NAME = "user_desktop_session"
GOLD_CHANNEL_ID = -1003679078163

async def main():
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    await client.connect()
    
    print(f"Fetching latest messages from 'Gold Pips Hunter' ({GOLD_CHANNEL_ID})...\n")
    entity = await client.get_entity(GOLD_CHANNEL_ID)
    print(f"Channel Title: {entity.title}\n")
    
    messages = await client.get_messages(entity, limit=10)
    for m in reversed(messages):
        if m.text:
            print("=" * 60)
            print(f"[{m.date}] Message ID: {m.id}")
            print(m.text)
            print("-" * 60)
            
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
