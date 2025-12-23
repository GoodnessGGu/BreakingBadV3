# Deployment Guide for IQ Option Bot 🚀

This bot is **stateful**. It remembers things:
1.  **Telegram Session** (`anon.session`): So you don't have to log in every time.
2.  **AI Brain** (`training_data.csv`): So it gets smarter over time.
3.  **ML Model** (`models/trade_model.pkl`): The trained brain.

Because of this, **you cannot use simple "Container" platforms** like Heroku or Render Free Tier easily, because they "wipe" the hard drive every day (Ephemeral Storage).

---

## ✅ Best Free Options (VPS)

These options give you a full "Virtual Computer" that runs 24/7 and keeps your files safe.

### 1. AWS Free Tier (EC2) 🥇
*   **What**: Amazon gives you a `t2.micro` or `t3.micro` server for **12 months free**.
*   **Pros**: Reliable, Standard Linux (Ubuntu), Persistent Storage.
*   **Cons**: Requires Credit Card for sign-up. Free period ends after 1 year.
*   **How**:
    1.  Sign up for AWS.
    2.  Launch **EC2 Instance** -> Select **Ubuntu 22.04 LTS**.
    3.  Select `t2.micro` (Free Tier usage eligible).
    4.  Download the `.pem` key file.
    5.  Connect via SSH: `ssh -i key.pem ubuntu@<IP-ADDRESS>`

### 2. Oracle Cloud "Always Free" 🥈
*   **What**: Oracle offers "Always Free" ARM-based instances (Ampere).
*   **Pros**: **Free Forever** (not just 1 year). Very powerful (up to 4 CPUs, 24GB RAM).
*   **Cons**: Signup verification is strict; sometimes hard to get an account.
*   **How**: Sign up for Oracle Cloud Free Tier and create a **VM.Standard.A1.Flex** instance.

### 3. Google Cloud Free Tier 🥉
*   **What**: `e2-micro` instance.
*   **Pros**: Always free in specific US regions.
*   **Cons**: Very weak CPU (shared core). Might lag with ML training.

### 4. Your Old Laptop / Raspberry Pi 🏠
*   **What**: Run it on an old computer at home.
*   **Pros**: $0 Cost. Maximum privacy.
*   **Cons**: Electricity/Internet outages stop the bot.
*   **How**: Install Python 3.10+, clone code, run `python telegram_bot.py`.

---

## 🛠️ Setup Guide (Ubuntu VPS)

Once you have your VPS (AWS/Oracle/Google), run these commands:

```bash
# 1. Update System
sudo apt update && sudo apt upgrade -y

# 2. Install Python & Tools
sudo apt install python3-pip python3-venv git screen -y

# 3. Clone Your Code (Or drag-and-drop using FileZilla/SCP)
# (Assuming you uploaded it to a folder named 'bot')
cd bot

# 4. Install Dependencies
pip3 install -r requirements.txt

# 5. Initialize AI Brain (Important!)
# This will fetch fresh data and train the model immediately.
python3 collect_data.py

# 5. Run the Bot (Use 'screen' to keep it running after you disconnect)
screen -S trading_bot
python3 telegram_bot.py
```

To detach from the screen (keep it running in background): Press `Ctrl+A` then `D`.
To re-attach later: `screen -r trading_bot`

---

## ⚠️ Avoid These (Unless you know what you're doing)
*   **Heroku Free**: No longer exists (Paid only).
*   **Render Free**: Wipes persistence. Bot will forget login loops/training data daily.
*   **PythonAnywhere Free**: Only allows HTTP requests to a "Whitelist". **Will block IQ Option Websockets**.
