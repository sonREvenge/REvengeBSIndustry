@echo off
cd /d "%~dp0"
:restart_loop
echo [%date% %time%] Starting Brawl Industry...
python main.py
set EXITCODE=%errorlevel%
echo [%date% %time%] Process exited (code %EXITCODE%). Restarting in 10s...
timeout /t 10 /nobreak >nul
goto restart_loop
