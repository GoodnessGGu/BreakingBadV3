import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

with open("my_telegram_channels.json", "r", encoding="utf-8") as f:
    channels = json.load(f)

print(f"Total channels/groups loaded: {len(channels)}")

keywords = ["gold", "pip", "hunter", "callisto", "xau", "signal", "trade", "fx", "amar"]

seen_ids = set()
matches = []

for c in channels:
    title = c.get("title", "").lower()
    uname = (c.get("username") or "").lower()
    for kw in keywords:
        if kw in title or kw in uname:
            if c["channel_id"] not in seen_ids:
                seen_ids.add(c["channel_id"])
                matches.append(c)
            break

print(f"\nFound {len(matches)} relevant trading channels/groups:")
for m in matches:
    print(f"Title: {m['title']}")
    print(f"   Channel ID: {m['channel_id']}")
    print(f"   Username:   {m['username']}")
    print(f"   Type:       {m['type']}")
    print("-" * 50)
