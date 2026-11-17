@echo off
REM The 0DTE bot's data collector: saves QQQ option chains every 5 minutes until the close.
cd /d "%~dp0"
if not exist data mkdir data
echo ==== %date% %time% ==== >> data\collect.log
set "PY="
if exist .venv\Scripts\python.exe set "PY=.venv\Scripts\python.exe"
if not defined PY if exist data\python_path.txt set /p PY=<data\python_path.txt
if not defined PY (where py >nul 2>&1 && set "PY=py -3")
if not defined PY set "PY=python"
%PY% -m agent.stan.zerodte.collect watch >> data\collect.log 2>&1
set "CODE=%errorlevel%"
echo Exit code: %CODE% >> data\collect.log
%PY% -m agent.discord_bot.notify collect %CODE% >> data\collect.log 2>&1
