@echo off
REM start_bot.bat - Windows batch script to start the bot with supervisor

echo ========================================
echo Starting Trading Bot with Supervisor
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Starting bot supervisor...
echo Press Ctrl+C to stop the bot
echo.

python bot_supervisor.py

pause
