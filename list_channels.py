import asyncio
import os
from telethon import TelegramClient
from dotenv import load_dotenv

# Load env variables
load_dotenv()

API_ID = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")

if not API_ID or not API_HASH:
    print("❌ Error: TELEGRAM_API_ID or TELEGRAM_API_HASH not found in .env")
    exit(1)

async def main():
    print("Connecting to Telegram...")
    # 'anon' is the session name. It will use/create anon.session in the current folder.
    # If prompted, you will need to enter your phone number and code in the terminal.
    async with TelegramClient('anon', int(API_ID), API_HASH) as client:
        print("\nYour Channels & Groups:\n" + "="*60)
        print(f"{'ID':<15} | {'Name'}")
        print("-" * 60)
        
        async for dialog in client.iter_dialogs():
            # Filter for Channels and Supergroups
            if dialog.is_channel or dialog.is_group:
                print(f"{dialog.id:<15} | {dialog.title}")
        
        print("="*60 + "\nCopy the ID of the channel you want to monitor into your .env file.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nError: {e}")
