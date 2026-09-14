@echo off
title BreakingBadV3 - All Copiers Launcher
cd /d "%~dp0"
echo ================================================
echo  BreakingBadV3 - Starting ALL Copiers
echo ================================================
echo.
echo Launching Gold Pips Hunter copier...
start "Gold Pips Hunter" cmd /k run_gold_pips_copier.bat
timeout /t 3 /nobreak >nul
echo Launching CallistoFx Zone copier...
start "CallistoFx Zones" cmd /k run_callisto_zone_copier.bat
echo.
echo Both copiers are running in separate windows.
echo You can close this window - the copier windows will keep running.
echo If a copier crashes, it will auto-restart in 10 seconds.
echo.
pause
