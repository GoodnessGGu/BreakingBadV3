@echo off
title BreakingBad V3 - Unified Master Trading Bot
chcp 65001 >nul
echo =======================================================
echo   BreakingBad V3 - Unified Master Trading Bot
echo   Telegram Controller + Callisto + Gold Pips + Polycarp + ICT
echo =======================================================

:loop
python run_unified_bot.py --account training --lots 1.0 --leverage 100 --blitz-stake 2.0 --ict-symbol XAUUSD --lookback-mins 10
echo [%date% %time%] Process exited or crashed. Restarting in 10 seconds...
timeout /t 10 /nobreak
goto loop
