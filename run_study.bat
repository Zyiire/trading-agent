@echo off
REM Stan's nightly school term: studies market history after the close and writes new notes.
REM It never places a trade. It can change which strategy Stan ranks best, which the 9:35 AM run trades.
cd /d "%~dp0"
if not exist data mkdir data
echo ==== %date% %time% ==== >> data\study.log
set "PY="
if exist .venv\Scripts\python.exe set "PY=.venv\Scripts\python.exe"
if not defined PY if exist data\python_path.txt set /p PY=<data\python_path.txt
if not defined PY (where py >nul 2>&1 && set "PY=py -3")
if not defined PY set "PY=python"
echo Using: %PY% >> data\study.log
%PY% -m agent.stan.train --pace 0 >> data\study.log 2>&1
echo Exit code: %errorlevel% >> data\study.log
