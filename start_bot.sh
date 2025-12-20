#!/bin/bash
# start_bot.sh - Linux/Mac shell script to start the bot with supervisor

echo "========================================"
echo "Starting Trading Bot with Supervisor"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "ERROR: Python 3 is not installed"
    exit 1
fi

echo "Starting bot supervisor..."
echo "Press Ctrl+C to stop the bot"
echo ""

python3 bot_supervisor.py
