# Register (or re-register) the nightly lesson-mining run with Task Scheduler.
#
#   powershell -NoProfile -File scripts\register-nightly.ps1            # daily at 04:30
#   powershell -NoProfile -File scripts\register-nightly.ps1 -At 03:00
#   Unregister-ScheduledTask -TaskName 'agent-knowledge nightly' -Confirm:$false   # remove it
#
# Runs as the current user, only while logged on (no stored password), with
# pythonw so no console window appears. The run itself declines to start when
# the GPU is busy or a model server is already up (see akn/nightly.py), and
# logs to state\nightly-<date>.log.
param([string]$At = '04:30')
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$py = (Get-Command python -ErrorAction Stop).Source
$pyw = Join-Path (Split-Path -Parent $py) 'pythonw.exe'
if (-not (Test-Path $pyw)) { throw "pythonw.exe not found next to $py" }

$action = New-ScheduledTaskAction -Execute $pyw -Argument '-m akn nightly' -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 8) -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName 'agent-knowledge nightly' -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Force `
    -Description ('Mine new Claude Code transcript lines for lessons with the local LLM (' + $root + '). Writes reports\latest.md.') | Out-Null

Get-ScheduledTask -TaskName 'agent-knowledge nightly' | Select-Object TaskName, State,
    @{n = 'Next run'; e = { (Get-ScheduledTaskInfo $_).NextRunTime } } | Format-List
