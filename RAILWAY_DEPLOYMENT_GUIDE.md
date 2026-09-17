# 🚀 BreakingBad V3 — Railway Cloud Deployment Guide

This guide details how to deploy the **BreakingBad V3 Unified Bot** to [Railway.app](https://railway.app) for 24/7 continuous uptime.

---

## 1. Prerequisites Prepared in the Repository
The following production files have been configured and pushed to GitHub:
- `Dockerfile`: Minimal Python 3.11 image with dependencies cached.
- `requirements.txt`: Lightweight production dependencies without heavy ML overhead.
- `railway.json`: Declares Dockerfile builder and auto-restart policy on failures.
- `Procfile`: Worker declaration for Railway.
- `.dockerignore`: Excludes local session files, datasets, and caches.
- `export_session_string.py`: Helper script that generated your cloud-ready `TELEGRAM_STRING_SESSION`.

---

## 2. Step-by-Step Railway Deployment

### Step 1: Push Repository to GitHub
Ensure all latest commits are pushed to your GitHub repository:
```bash
git push origin main
```

### Step 2: Create a New Project on Railway
1. Go to [Railway.app](https://railway.app) and log into your dashboard.
2. Click **New Project** -> **Deploy from GitHub repo**.
3. Select `GoodnessGGu/BreakingBadV3` (or your repository).
4. Railway will detect the `Dockerfile` and `railway.json` automatically.

### Step 3: Add Environment Variables in Railway
In your Railway project service settings, navigate to the **Variables** tab and click **New Variable** (or **RAW Editor**):

```env
TELEGRAM_TOKEN=8354182082:AAE-LNeZ4VPPRHwd-YLwDo-L-TVHao4queQ
ADMIN_ID=6420777416
TELEGRAM_API_ID=28666718
TELEGRAM_API_HASH=3cfb2e693126f59ba79b908709320e40
TELEGRAM_STRING_SESSION=1BJWap1wBu5nFoNul5fySfJ4FQTOwIkPYjfA8v1WS3_MXCNRaI-3YKtsAYqpkQYV_0JwCQ4KuvrOcErvBEp0H5mHBZ8VtL4j5xkZHjZU3V2j1zz-Xl1WDldY7q25W_Is--Tt1QIthlkFZBuWAtsmerQZFjN5FRMiBLM-IH-aiKLLisfm2n8qOjE4pPY2WKR2f6s0zLK7fBUVS4ohuFqRQRNKBVGP3aoEhkcbRrY9DoGV0rtAQENPNH4A0jIqWtJnahUcF_MvR8O5hGHogW7X4PlICLKDB-phl0c1uhWw4kJ0Ks3uooL98LVfMQbIQ5OSvYcJq8A4glIY2vFcRYH9s5B_mDq1LVmI=
IQ_AI_TOKEN=YOUR_IQ_OPTION_AI_TOKEN
ACCOUNT_TYPE=training
DEFAULT_LOTS=1.0
DEFAULT_LEVERAGE=100
BLITZ_STAKE=2.0
ICT_SYMBOL=XAUUSD
LOOKBACK_MINS=10
```

> [!IMPORTANT]
> **Why `TELEGRAM_STRING_SESSION`?**
> Standard `.session` files are local binary SQLite databases that cannot prompt for phone confirmation codes inside headless containers. The `TELEGRAM_STRING_SESSION` variable allows Telethon to authenticate headlessly on Railway without requiring interactive verification!

### Step 4: Deploy & Verify
1. Click **Deploy**.
2. Go to the **Deployments** tab and click on the active deployment to view the build and runtime logs.
3. Look for the startup sequence:
   ```text
   ✅ IQ Option MCP session initialized!
   ✅ IQ Option Blitz MCP session initialized!
   Connecting unified Telethon listener using StringSession from environment...
   ✅ Connected as: 3Gees (@gushex)
   🤖 Telegram Bot UI active & listening for user commands!
   🚀 ChannelManager running!
   ```
4. Open Telegram, open **@WalterAWbot**, and type `/status` or `/menu` to confirm the cloud instance is active!
