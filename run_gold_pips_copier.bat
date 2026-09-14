@echo off
title Gold Pips Hunter Copier
cd /d "%~dp0"
:loop
echo [%date% %time%] Starting Gold Pips Hunter copier...
python gold_pips_copier.py --account training --lots 1.0 --leverage 100 --tp-target 1
echo [%date% %time%] Copier exited (code %errorlevel%). Restarting in 10 seconds...
timeout /t 10 /nobreak
goto loop
