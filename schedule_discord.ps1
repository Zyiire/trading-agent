# Registers the "Stan Discord" task: starts run_discord.bat, hidden, whenever Zy logs in.
# No time limit (Windows otherwise stops tasks after 72 hours), keeps running on battery, one copy at a time.
$ErrorActionPreference = 'Stop'
$bat = Join-Path $PSScriptRoot 'run_discord.bat'
$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument "-NoProfile -WindowStyle Hidden -Command `"& '$bat'`"" -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -StartWhenAvailable
Register-ScheduledTask -TaskName 'Stan Discord' -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName 'Stan Discord'
Write-Output 'Done. Stan Discord starts when you log in and is starting now. Log: data\discord.log'
