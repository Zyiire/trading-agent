@echo off
REM Stan's daily shadow trading run. Schedule this for 4:45pm on weekdays
REM with Windows Task Scheduler (see README). It never places real orders.
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
if not exist data mkdir data
python -m agent.stan.shadow >> data\shadow.log 2>&1
