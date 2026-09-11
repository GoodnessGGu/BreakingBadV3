"""
telegram_auth.py - Interactive Telegram Account Login & Channel Grabber

Allows authenticating your Telegram user account and extracting all joined channel IDs.
"""

import os
import sys
import json
import asyncio
import argparse
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.tl.types import Channel, Chat
from telethon.errors import (
    SessionPasswordNeededError,
    PhoneCodeInvalidError,
    PhoneCodeExpiredError
)

# Force UTF-8 stdout
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

API_ID = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")
SESSION_NAME = "user_desktop_session"
STATE_FILE = "tele_auth_state.json"

if not API_ID or not API_HASH:
    print("❌ TELEGRAM_API_ID or TELEGRAM_API_HASH missing in .env")
    sys.exit(1)

async def send_login_code(phone: str):
    clean_phone = phone.strip().replace(" ", "").replace("-", "")
    print(f"Connecting to Telegram for phone: {clean_phone}...")
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    await client.connect()

    if await client.is_user_authorized():
        me = await client.get_me()
        print(f"✅ Already logged in as {me.first_name} (@{me.username}) [ID: {me.id}]!")
        await client.disconnect()
        return

    try:
        sent = await client.send_code_request(clean_phone)
        with open(STATE_FILE, "w") as f:
            json.dump({
                "phone": clean_phone,
                "phone_code_hash": sent.phone_code_hash
            }, f)
        print(f"📩 Code requested successfully!")
        print(f"Check your Telegram app for the login code.")
    except Exception as e:
        print(f"❌ Failed to send code: {e}")
    finally:
        await client.disconnect()

async def verify_code(code: str):
    clean_code = code.strip().replace(" ", "").replace("-", "")
    if not os.path.exists(STATE_FILE):
        print("❌ No active login state found. Please request code first.")
        return

    with open(STATE_FILE, "r") as f:
        state = json.load(f)

    phone = state.get("phone")
    phone_code_hash = state.get("phone_code_hash")

    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    await client.connect()

    try:
        await client.sign_in(phone=phone, code=clean_code, phone_code_hash=phone_code_hash)
        me = await client.get_me()
        print(f"🎉 SUCCESS! Logged in as: {me.first_name} {me.last_name or ''} (@{me.username}) [ID: {me.id}]")
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
    except SessionPasswordNeededError:
        print("🔐 2-Step Verification (2FA) is enabled on this account.")
        print("Please provide your 2FA password to complete login.")
    except PhoneCodeInvalidError:
        print("❌ Invalid verification code. Please check and try again.")
    except PhoneCodeExpiredError:
        print("❌ Verification code expired. Please request a new code.")
    except Exception as e:
        print(f"❌ Error during verification: {e}")
    finally:
        await client.disconnect()

async def submit_password(password: str):
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    await client.connect()
    try:
        await client.sign_in(password=password.strip())
        me = await client.get_me()
        print(f"🎉 SUCCESS! 2FA verified. Logged in as: {me.first_name} (@{me.username}) [ID: {me.id}]")
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
    except Exception as e:
        print(f"❌ Error verifying password: {e}")
    finally:
        await client.disconnect()

async def list_all_channels():
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    await client.connect()

    if not await client.is_user_authorized():
        print("❌ Not authorized. Please log in first.")
        await client.disconnect()
        return

    me = await client.get_me()
    print(f"🔍 Fetching all channels for {me.first_name} (@{me.username})...\n")

    channels = []
    async for dialog in client.iter_dialogs():
        entity = dialog.entity
        if isinstance(entity, Channel) or isinstance(entity, Chat):
            is_broadcast = getattr(entity, "broadcast", False)
            is_megagroup = getattr(entity, "megagroup", False)
            username = getattr(entity, "username", None)
            
            # Telegram channel IDs use the -100 prefix for bots/APIs
            full_channel_id = f"-100{entity.id}" if not str(entity.id).startswith("-100") else str(entity.id)
            
            channels.append({
                "title": dialog.name,
                "channel_id": full_channel_id,
                "raw_id": entity.id,
                "username": f"@{username}" if username else None,
                "type": "Channel" if is_broadcast else ("Supergroup" if is_megagroup else "Group"),
                "unread_count": dialog.unread_count
            })

    # Save full list to JSON
    with open("my_telegram_channels.json", "w", encoding="utf-8") as f:
        json.dump(channels, f, indent=2, ensure_ascii=False)

    print(f"✅ FOUND {len(channels)} CHANNELS / GROUPS!\n")
    print("=" * 70)
    for c in channels:
        u_str = f" | {c['username']}" if c['username'] else " (Private)"
        print(f"📢 {c['title']}{u_str}")
        print(f"   ID:   {c['channel_id']}")
        print(f"   Type: {c['type']}")
        print("-" * 70)

    print(f"\n📁 Full list saved to 'my_telegram_channels.json'")
    await client.disconnect()

def main():
    parser = argparse.ArgumentParser(description="Telegram Account Login & Channel Grabber")
    parser.add_argument("--send-code", help="Phone number with country code (e.g. +1234567890)")
    parser.add_argument("--verify", help="Verification code received from Telegram")
    parser.add_argument("--password", help="2FA cloud password if enabled")
    parser.add_argument("--list-channels", action="store_true", help="List all joined channels and IDs")

    args = parser.parse_args()

    if args.send_code:
        asyncio.run(send_login_code(args.send_code))
    elif args.verify:
        asyncio.run(verify_code(args.verify))
    elif args.password:
        asyncio.run(submit_password(args.password))
    elif args.list_channels:
        asyncio.run(list_all_channels())
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
