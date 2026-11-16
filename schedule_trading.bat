@echo off
REM One-time setup: Stan trades at 9:35 AM every Monday to Friday, until you remove the task.
REM Run this from the same terminal you use for Stan, so it remembers that Python.
REM The computer needs to be on and awake at 9:35 AM. To pause, use the Trading switch in the web app.
REM To remove the schedule completely:
REM   schtasks /Delete /TN "Stan open run" /F
cd /d "%~dp0"
if not exist data mkdir data
for /f "delims=" %%p in ('where python 2^>nul') do (
  echo %%p | findstr /i "WindowsApps" >nul || (if not exist data\python_path.txt echo "%%p"> data\python_path.txt)
)
if exist data\python_path.txt (echo Stan will use Python at: & type data\python_path.txt) else (echo Could not find Python outside the Microsoft Store shortcut. Stan will try the py launcher.)
schtasks /Create /TN "Stan open run" /TR "\"%~dp0run_open.bat\"" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 09:35 /F
if errorlevel 1 (echo Could not create the task. Try right-clicking this file and choosing Run as administrator.) else (echo Done. Stan will trade at 9:35 AM every Monday to Friday. Pause him with the Trading switch in the web app. Log: data\open.log)
pause
