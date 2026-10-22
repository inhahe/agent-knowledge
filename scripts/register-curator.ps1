# Register (or re-register) the morning curator with Task Scheduler.
#
#   powershell -NoProfile -File scripts\register-curator.ps1            # daily at 06:00
#   powershell -NoProfile -File scripts\register-curator.ps1 -At 07:00
#   Unregister-ScheduledTask -TaskName 'agent-knowledge curator' -Confirm:$false   # remove it
#
# Runs `python -m akn curate --run` as the current user, only while logged on, with
# pythonw so no console window appears. The run waits (up to 3 h) for the nightly
# mining run to finish. A headless Opus session (akn/curate.py, curator/PROMPT.md) then
# rewrites the one section it owns, "Lessons agents learned (maintained automatically)",
# at the end of the global and workspace CLAUDE.md files, within a size budget. It logs
# to state\curator-<date>.log; what changed and why is in curated\CHANGES.md.
# Disable-ScheduledTask -TaskName 'agent-knowledge curator' pauses it.
param([string]$At = '06:00')
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$py = (Get-Command python -ErrorAction Stop).Source
$pyw = Join-Path (Split-Path -Parent $py) 'pythonw.exe'
if (-not (Test-Path $pyw)) { throw "pythonw.exe not found next to $py" }

$action = New-ScheduledTaskAction -Execute $pyw -Argument '-m akn curate --run' -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 5) -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName 'agent-knowledge curator' -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Force `
    -Description ('Curates the newest lessons report into the "Lessons agents learned (maintained automatically)" section of the global and workspace CLAUDE.md files (' + $root + '; see CURATED-RULES.md and curated\CHANGES.md).') | Out-Null

Get-ScheduledTask -TaskName 'agent-knowledge curator' | Select-Object TaskName, State,
    @{n = 'Next run'; e = { (Get-ScheduledTaskInfo $_).NextRunTime } } | Format-List
