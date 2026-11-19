@echo off
REM Keeps Stan's Discord bot running. It restarts 30 seconds after a crash, and stops for good
REM if a setting, the bot token or Discord's Message Content Intent needs fixing (see data\discord.log).
cd /d "%~dp0"
if not exist data mkdir data
set "PY="
if exist .venv\Scripts\python.exe set "PY=.venv\Scripts\python.exe"
if not defined PY if exist data\python_path.txt set /p PY=<data\python_path.txt
if not defined PY (where py >nul 2>&1 && set "PY=py -3")
if not defined PY set "PY=python"
:loop
echo ==== %date% %time% starting Stan's Discord bot ==== >> data\discord.log
%PY% -m agent.discord_bot.bot >> data\discord.log 2>&1
set "CODE=%errorlevel%"
echo Bot exited with code %CODE% >> data\discord.log
if "%CODE%"=="2" (
  echo Not restarting: fix the problem above, then run this file again. >> data\discord.log
  exit /b 2
)
ping -n 31 127.0.0.1 >nul
goto loop
