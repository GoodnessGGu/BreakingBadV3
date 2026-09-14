@echo off
title BreakingBadV3 - Autonomous ICT Gold Engine
cd /d "%~dp0"
:loop
echo [%date% %time%] Starting Autonomous ICT Gold Engine...
python gold_ict_engine.py --account training --lots 1.0 --leverage 100 --rr 2.0
echo [%date% %time%] Engine exited (code %errorlevel%). Restarting in 10 seconds...
timeout /t 10 /nobreak
goto loop
