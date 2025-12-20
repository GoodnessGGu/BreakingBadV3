# Timezone Configuration Guide

## Overview

The bot now supports configurable timezones to ensure all trades execute at the correct time, regardless of where your server is located or what timezone your signal provider uses.

---

## Quick Setup

### 1. Set Your Timezone

Edit your `.env` file and set the `TIMEZONE` variable:

```env
# For Brazil (UTC-3)
TIMEZONE=America/Sao_Paulo

# For US Eastern
TIMEZONE=America/New_York

# For UK
TIMEZONE=Europe/London

# For UTC (default)
TIMEZONE=UTC
```

### 2. Restart the Bot

After changing the timezone, restart the bot:
```bash
python telegram_bot.py
```

---

## Using the `/timezone` Command

### View Current Timezone

Send `/timezone` to see your current timezone and the current time:

```
🌍 Current Timezone

Timezone: America/Sao_Paulo
Current Time: 2025-12-13 00:07:24 -03

Common Timezones:
• UTC - Coordinated Universal Time
• America/New_York - US Eastern
• America/Sao_Paulo - Brazil (UTC-3)
• Europe/London - UK
• Asia/Tokyo - Japan

To change: /timezone <timezone_name>
```

### Validate a Timezone

Send `/timezone <timezone_name>` to check if a timezone is valid:

```
/timezone America/Sao_Paulo
```

Response:
```
✅ Timezone Update

To set timezone to America/Sao_Paulo, update your .env file:

TIMEZONE=America/Sao_Paulo

Then restart the bot with /shutdown and start again.
```

---

## Common Timezones

| Region | Timezone | UTC Offset |
|--------|----------|------------|
| **Americas** |
| US Eastern | `America/New_York` | UTC-5/-4 |
| US Pacific | `America/Los_Angeles` | UTC-8/-7 |
| Brazil | `America/Sao_Paulo` | UTC-3 |
| Argentina | `America/Argentina/Buenos_Aires` | UTC-3 |
| **Europe** |
| UK | `Europe/London` | UTC+0/+1 |
| Germany | `Europe/Berlin` | UTC+1/+2 |
| **Asia** |
| Japan | `Asia/Tokyo` | UTC+9 |
| India | `Asia/Kolkata` | UTC+5:30 |
| Singapore | `Asia/Singapore` | UTC+8 |
| **Other** |
| UTC | `UTC` | UTC+0 |

---

## How It Works

### Signal Parsing

When a signal is received with an entry time like "12:36 PM":
- ✅ The time is interpreted in **your configured timezone**
- ✅ The bot waits until that exact time in your timezone
- ✅ The trade executes at the correct moment

### Example

**Scenario:**
- Your timezone: `America/Sao_Paulo` (UTC-3)
- Signal entry time: `12:36 PM`
- Current time: `12:30 PM -03`

**What happens:**
1. Bot parses "12:36 PM" as 12:36 PM Brazil time
2. Bot waits 6 minutes
3. At exactly 12:36 PM -03, the trade executes

### Database Timestamps

All timestamps in the database include timezone information:
```
2025-12-13T00:07:24-03:00
```

This ensures:
- ✅ Accurate trade history
- ✅ Correct time filtering for `/history`, `/charts`, `/export`
- ✅ Proper timezone conversion if you move servers

---

## Status Display

The `/status` command now shows your configured timezone:

```
🔌 Connection: ✅ Connected
💼 Account Type: PRACTICE
💰 Balance: $10000.00
🕒 Uptime: 0h 2m
🌍 Timezone: America/Sao_Paulo

⚙️ Settings:
💵 Amount: $10 | 🔄 Gales: 2
⏸️ Paused: False | 🚫 Suppress: True
```

---

## Troubleshooting

### Invalid Timezone Error

If you see:
```
⚠️ Unknown timezone 'XYZ', falling back to UTC
```

**Solution:**
1. Check the timezone name is correct
2. Use `/timezone <name>` to validate
3. See full list: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones

### Trades Executing at Wrong Time

**Check:**
1. Verify timezone with `/timezone`
2. Compare current time shown vs your actual time
3. Update `.env` if needed
4. Restart bot

### Signal Channel Uses Different Timezone

**Example:** Your signal channel posts times in UTC, but you're in Brazil (UTC-3)

**Solution:**
Set your timezone to match the signal channel:
```env
TIMEZONE=UTC
```

Or convert manually when you see signals.

---

## Technical Details

### Timezone-Aware Functions

The bot now uses timezone-aware datetime throughout:

**Before:**
```python
now = datetime.now()  # Naive, no timezone
```

**After:**
```python
from timezone_utils import now
now = now()  # Timezone-aware
```

### Affected Components

✅ **Signal Parsing** - Entry times interpreted in configured timezone  
✅ **Trade Execution** - Scheduled at correct timezone  
✅ **Database** - All timestamps include timezone  
✅ **Charts** - Time axes show correct timezone  
✅ **Exports** - Timestamps formatted with timezone  
✅ **Logs** - All log times in configured timezone  

---

## Best Practices

1. **Match Signal Source**: Set timezone to match your signal provider
2. **Check Status**: Use `/status` to verify timezone after restart
3. **Test First**: Send a test signal to verify timing is correct
4. **Document**: Note your timezone in your trading journal

---

## Summary

✅ **Configured via `.env`**: Easy to change  
✅ **Validated via `/timezone`**: Check before changing  
✅ **Displayed in `/status`**: Always visible  
✅ **Used everywhere**: Consistent across all features  
✅ **Database-safe**: Timestamps include timezone info  

Your bot is now timezone-aware! 🌍
