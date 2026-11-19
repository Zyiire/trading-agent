@echo off
REM Stan trades the paper account shortly after the open (fractional day orders fill right away).
cd /d "%~dp0"
if not exist data mkdir data
echo ==== %date% %time% ==== >> data\open.log
set "PY="
if exist .venv\Scripts\python.exe set "PY=.venv\Scripts\python.exe"
if not defined PY if exist data\python_path.txt set /p PY=<data\python_path.txt
if not defined PY (where py >nul 2>&1 && set "PY=py -3")
if not defined PY set "PY=python"
echo Using: %PY% >> data\open.log
%PY% -c "import sys; print(sys.executable)" >> data\open.log 2>&1
%PY% -m agent.stan.paper >> data\open.log 2>&1
set "CODE=%errorlevel%"
echo Exit code: %CODE% >> data\open.log
%PY% -m agent.discord_bot.notify open %CODE% >> data\open.log 2>&1
