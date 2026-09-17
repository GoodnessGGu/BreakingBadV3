"""
channel_id_grabber.py - Real-Time Telegram Channel ID Grabber

Uses your bot @WalterAWbot to automatically catch and display Channel IDs.
Simply forward ANY post/signal from 'Gold Pips Hunter' or 'CallistoFx' to @WalterAWbot on Telegram!
"""

import os
import sys
import time
import requests
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")

if not TOKEN:
    print("[ERROR] TELEGRAM_TOKEN not found in .env")
    exit(1)

def get_channel_info():
    print("=" * 60)
    print("[BOT] TELEGRAM CHANNEL ID GRABBER ACTIVE!")
    print("-> Open Telegram on your phone or PC.")
    print("-> Search for your bot: @WalterAWbot")
    print("-> Click 'Start', then FORWARD any message from 'Gold Pips Hunter' into @WalterAWbot.")
    print("=" * 60)
    print("Waiting for forwarded message... (Press Ctrl+C to stop)\n")

    offset = None
    # Flush existing updates first
    try:
        r = requests.get(f"https://api.telegram.org/bot{TOKEN}/getUpdates?offset=-1", timeout=10).json()
        if r.get("ok") and r.get("result"):
            offset = r["result"][-1]["update_id"] + 1
    except Exception as e:
        print(f"Init error: {e}")

    while True:
        try:
            url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
            params = {"timeout": 20}
            if offset:
                params["offset"] = offset

            resp = requests.get(url, params=params, timeout=25).json()
            if not resp.get("ok"):
                time.sleep(2)
                continue

            for update in resp.get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message") or update.get("channel_post")
                if not msg:
                    continue

                chat_id = msg.get("chat", {}).get("id")
                
                # Check if this was forwarded from a channel
                forward_chat = msg.get("forward_from_chat")
                if forward_chat:
                    f_id = forward_chat.get("id")
                    f_title = forward_chat.get("title", "Unknown")
                    f_username = forward_chat.get("username")
                    f_type = forward_chat.get("type", "channel")
                    
                    print("\n" + "=" * 50)
                    print(f"[SUCCESS] CHANNEL DETECTED!")
                    print(f"Channel Title:    {f_title}")
                    print(f"Channel ID:       {f_id}")
                    print(f"Channel Username: @{f_username}" if f_username else "Channel Username: None (Private Channel)")
                    print(f"Channel Type:     {f_type}")
                    print("=" * 50 + "\n")
                    
                    # Reply back on Telegram
                    reply_text = (
                        f"✅ *Channel Identified!*\n\n"
                        f"📢 *Title:* `{f_title}`\n"
                        f"🆔 *Channel ID:* `{f_id}`\n"
                    )
                    if f_username:
                        reply_text += f"🔗 *Username:* @{f_username}\n"
                    reply_text += f"\nYou can now paste this ID into your bot config!"
                    
                    try:
                        requests.post(
                            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                            json={"chat_id": chat_id, "text": reply_text, "parse_mode": "Markdown"}
                        )
                    except Exception:
                        pass
                        
                else:
                    text = msg.get("text", "")
                    sender = msg.get("from", {}).get("first_name", "User")
                    print(f"[MSG] Received from {sender}: '{text}'")
                    reply_text = (
                        "👋 Hello! To get a channel's ID, please **FORWARD** a message/signal from that channel to me here!"
                    )

                    try:
                        requests.post(
                            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                            json={"chat_id": chat_id, "text": reply_text, "parse_mode": "Markdown"}
                        )
                    except Exception:
                        pass

        except requests.RequestException:
            time.sleep(2)
        except KeyboardInterrupt:
            print("\nStopped.")
            break

if __name__ == "__main__":
    get_channel_info()
