@echo off
REM Stan watches QQQ from 9:31 to 10:31 for the box breakout and sends one bracket order (paper account).
cd /d "%~dp0"
if not exist data mkdir data
echo ==== %date% %time% ==== >> data\box.log
set "PY="
if exist .venv\Scripts\python.exe set "PY=.venv\Scripts\python.exe"
if not defined PY if exist data\python_path.txt set /p PY=<data\python_path.txt
if not defined PY (where py >nul 2>&1 && set "PY=py -3")
if not defined PY set "PY=python"
echo Using: %PY% >> data\box.log
%PY% -c "import sys; print(sys.executable)" >> data\box.log 2>&1
%PY% -m agent.stan.box.live watch >> data\box.log 2>&1
set "CODE=%errorlevel%"
echo Exit code: %CODE% >> data\box.log
%PY% -m agent.discord_bot.notify box %CODE% >> data\box.log 2>&1
