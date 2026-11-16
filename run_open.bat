@echo off
REM Stan trades the paper account shortly after the open (fractional day orders fill right away).
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
if not exist data mkdir data
echo ==== %date% %time% ==== >> data\open.log
python -m agent.stan.paper >> data\open.log 2>&1
