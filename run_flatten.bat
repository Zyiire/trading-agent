@echo off
REM At 3:55 PM Stan cancels the box bracket's open legs, closes the box position and records the trade's R.
cd /d "%~dp0"
if not exist data mkdir data
echo ==== %date% %time% ==== >> data\flatten.log
set "PY="
if exist .venv\Scripts\python.exe set "PY=.venv\Scripts\python.exe"
if not defined PY if exist data\python_path.txt set /p PY=<data\python_path.txt
if not defined PY (where py >nul 2>&1 && set "PY=py -3")
if not defined PY set "PY=python"
echo Using: %PY% >> data\flatten.log
%PY% -c "import sys; print(sys.executable)" >> data\flatten.log 2>&1
%PY% -m agent.stan.box.live flatten >> data\flatten.log 2>&1
set "CODE=%errorlevel%"
echo Exit code: %CODE% >> data\flatten.log
%PY% -m agent.discord_bot.notify flatten %CODE% >> data\flatten.log 2>&1
