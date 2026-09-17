import os
import asyncio
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.tl.types import Channel, Chat

load_dotenv()

api_id = os.getenv("TELEGRAM_API_ID")
api_hash = os.getenv("TELEGRAM_API_HASH")

async def main():
    print(f"Connecting to Telegram with session 'bot_session'...")
    client = TelegramClient("bot_session", api_id, api_hash)
    await client.connect()
    
    if not await client.is_user_authorized():
        print("❌ Session is not authorized! Phone verification required.")
        return

    me = await client.get_me()
    print(f"✅ Connected as: {me.first_name} (@{me.username}) [ID: {me.id}]")

    print("\n=== SEARCHING DIALOGS FOR 'Gold' OR 'Callisto' ===")
    found = []
    all_channels = []
    
    async for dialog in client.iter_dialogs():
        entity = dialog.entity
        name = dialog.name
        is_channel = isinstance(entity, Channel)
        
        if is_channel or isinstance(entity, Chat):
            all_channels.append({
                "id": entity.id,
                "title": name,
                "username": getattr(entity, "username", None),
                "is_channel": is_channel,
                "is_group": getattr(entity, "megagroup", False)
            })
            
            # Check match
            lower_name = name.lower()
            if any(k in lower_name for k in ["gold", "pips", "hunter", "callisto", "fx", "signal"]):
                found.append({
                    "id": entity.id,
                    "title": name,
                    "username": getattr(entity, "username", None)
                })

    print(f"\n--- MATCHING CHANNELS ({len(found)}) ---")
    for f in found:
        print(f"Title: {f['title']}")
        print(f"Channel ID: -100{f['id']}")
        print(f"Raw ID: {f['id']}")
        print(f"Username: @{f['username']}" if f['username'] else "Username: None (Private)")
        print("-" * 40)

    print(f"\nTotal channels/chats in account: {len(all_channels)}")
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
