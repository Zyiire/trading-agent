@echo off
REM One-time setup: Stan trades at 9:35 AM every Monday to Friday, the 0DTE data collector
REM runs from 9:25 AM to the close, and Stan studies a new school term at 6 PM, until you remove the tasks.
REM Run this from the same terminal you use for Stan, so it remembers that Python.
REM The computer needs to be on and awake at 9:35 AM. To pause, use the Trading switch in the web app.
REM To remove the schedule completely:
REM   schtasks /Delete /TN "Stan open run" /F
REM   schtasks /Delete /TN "Stan 0DTE data" /F
REM   schtasks /Delete /TN "Stan nightly study" /F
cd /d "%~dp0"
if not exist data mkdir data
for /f "delims=" %%p in ('where python 2^>nul') do (
  echo %%p | findstr /i "WindowsApps" >nul || (if not exist data\python_path.txt echo "%%p"> data\python_path.txt)
)
if exist data\python_path.txt (echo Stan will use Python at: & type data\python_path.txt) else (echo Could not find Python outside the Microsoft Store shortcut. Stan will try the py launcher.)
schtasks /Create /TN "Stan open run" /TR "\"%~dp0run_open.bat\"" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 09:35 /F
if errorlevel 1 (echo Could not create the task. Try right-clicking this file and choosing Run as administrator.) else (echo Done. Stan will trade at 9:35 AM every Monday to Friday. Pause him with the Trading switch in the web app. Log: data\open.log)
schtasks /Create /TN "Stan 0DTE data" /TR "\"%~dp0run_collect.bat\"" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 09:25 /F
if errorlevel 1 (echo Could not create the data task.) else (echo Done. The 0DTE data collector runs 9:25 AM to the close every weekday. Log: data\collect.log)
schtasks /Create /TN "Stan nightly study" /TR "\"%~dp0run_study.bat\"" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 18:00 /F
if errorlevel 1 (echo Could not create the study task.) else (echo Done. Stan studies a new term at 6 PM every weekday. Log: data\study.log)
pause
