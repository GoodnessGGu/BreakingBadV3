"""
export_session_string.py - Helper to export current Telethon session to StringSession
Use this string in Railway / Cloud environment variables as TELEGRAM_STRING_SESSION.
"""

import os
from dotenv import load_dotenv
from telethon.sync import TelegramClient
from telethon.sessions import StringSession

load_dotenv()

api_id = os.getenv("TELEGRAM_API_ID")
api_hash = os.getenv("TELEGRAM_API_HASH")

if not api_id or not api_hash:
    print("❌ Error: Missing TELEGRAM_API_ID or TELEGRAM_API_HASH in .env")
    exit(1)

session_file = "user_desktop_session"
if not os.path.exists(f"{session_file}.session"):
    print(f"❌ Error: Session file '{session_file}.session' not found.")
    exit(1)

print(f"Opening '{session_file}.session'...")
with TelegramClient(session_file, int(api_id), api_hash) as client:
    if not client.is_user_authorized():
        print("❌ Session is not authorized.")
        exit(1)
    
    me = client.get_me()
    session_str = StringSession.save(client.session)
    print("=" * 60)
    print(f"Authorized as: {me.first_name} (@{me.username}) ID: {me.id}")
    print("=" * 60)
    print("TELEGRAM_STRING_SESSION:")
    print(session_str)
    print("=" * 60)
