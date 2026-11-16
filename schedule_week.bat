@echo off
REM One-time setup: Stan trades at 9:35 AM every weekday through Friday Nov 20, 2026.
REM The computer needs to be on and awake at 9:35 AM. Remove early with:
REM   schtasks /Delete /TN "Stan open run" /F
cd /d "%~dp0"
schtasks /Create /TN "Stan open run" /TR "\"%~dp0run_open.bat\"" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 09:35 /ED 11/20/2026 /F
if errorlevel 1 (echo Could not create the task. Try right-clicking this file and choosing Run as administrator.) else (echo Done. Stan will trade at 9:35 AM on weekdays through Friday Nov 20. Log: data\open.log)
pause
