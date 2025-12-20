# Channel Monitoring Setup Guide

## Quick Start

### 1. Get Telegram API Credentials

1. Visit https://my.telegram.org/auth
2. Log in with your phone number
3. Go to "API development tools"
4. Create an application
5. Copy your `API_ID` and `API_HASH`

### 2. Get Your Channel ID

**Method 1: Using a Bot**
- Forward a message from your channel to [@userinfobot](https://t.me/userinfobot)
- Or use [@getidsbot](https://t.me/getidsbot)
- The ID will be a negative number (e.g., `-1001234567890`)

**Method 2: Using Telegram Desktop**
- Right-click the channel
- Copy the invite link
- The channel ID is in the link

### 3. Update .env File

Edit your `.env` file and add these lines:

```env
# Replace with your actual values
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=abcdef1234567890abcdef1234567890
SIGNAL_CHANNEL_ID=-1001234567890
```

### 4. Install New Dependency

```bash
pip install telethon==1.34.0
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### 5. Start Your Bot

```bash
python telegram_bot.py
```

### 6. Enable Channel Monitoring

In your Telegram bot, send:
```
/channels
```

Or tap the "📡 Channels" button.

**First time setup:** Telethon may ask you to verify your phone number. Check the console output and follow the prompts.

---

## Signal Format

Your channel should post signals in this format:

```
🔔 NEW SIGNAL!

🎫 Trade: 🇦🇺 AUD/JPY 🇯🇵 (OTC)
⏳ Timer: 5 minutes
➡️ Entry: 12:36 PM
📈 Direction: BUY 🟩
```

**Supported Directions:**
- `BUY` → Converted to `CALL`
- `SELL` → Converted to `PUT`

**Time Format:**
- 12-hour format with AM/PM (e.g., `12:36 PM`, `02:15 AM`)

---

## What Happens When a Signal is Posted

1. **Signal Detected** → You receive notification
2. **Waiting Period** → Bot waits until entry time (if needed)
3. **Trade Entered** → You receive notification with trade details
4. **Trade Execution** → Martingale strategy applied if configured
5. **Results** → You receive win/loss notification

---

## Commands

- `/channels` - Toggle channel monitoring on/off
- `/pause` - Pause all trading (including channel signals)
- `/resume` - Resume trading
- `/status` - Check bot status and monitoring state

---

## Troubleshooting

**"Channel Monitoring Not Configured"**
→ Add `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, and `SIGNAL_CHANNEL_ID` to `.env`

**"Failed to start monitoring"**
→ Check that your API credentials are correct
→ Ensure the channel ID is correct (negative number)
→ Verify you have access to the channel

**Signal not detected**
→ Verify the signal format matches exactly
→ Check that monitoring is active (`/channels` to verify)
→ Ensure the channel ID is correct

**"Phone number verification required"**
→ Check the console output when starting monitoring
→ Enter your phone number and verification code when prompted
→ This only happens on first setup

---

## Notes

- Channel monitoring runs in the background
- The bot will continue monitoring until you stop it with `/channels`
- If bot is paused (`/pause`), signals will be detected but not traded
- Session file (`bot_session.session`) stores authentication - don't delete it
- All trades respect your configured settings (amount, martingale, etc.)
