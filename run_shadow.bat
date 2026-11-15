@echo off
REM Stan's evening run: shadow trading and practice, then paper trading on the
REM Alpaca PAPER account. Schedule this for 4:45pm on weekdays with Windows Task
REM Scheduler (see README). Live trading is locked in the code.
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
if not exist data mkdir data
python -m agent.stan.shadow >> data\shadow.log 2>&1
python -m agent.stan.paper >> data\paper.log 2>&1
