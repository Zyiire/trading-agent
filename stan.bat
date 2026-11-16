@echo off
REM Double-click to open Stan in your browser. Add --demo to look around with made-up prices.
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
if not exist data mkdir data
python -m agent.web %*
pause
