@echo off
title CallistoFx Zone Copier
cd /d "%~dp0"
:loop
echo [%date% %time%] Starting CallistoFx Zone copier...
python callisto_zone_copier.py --account training --lots 1.0 --leverage 100
echo [%date% %time%] Copier exited (code %errorlevel%). Restarting in 10 seconds...
timeout /t 10 /nobreak
goto loop
