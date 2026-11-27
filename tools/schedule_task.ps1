<#
.SYNOPSIS
  Register a ONE-SHOT Windows Task Scheduler task that runs a PowerShell script at a given time
  (e.g. launch Kaggle runs right after the weekly quota reset). Only on an explicit user request to launch.

.DESCRIPTION
  Shared by all projects (first used 2026-11-27, MMCultureQA X15/X5 launch at 07:30 Saturday).
  Pattern: write a small launch script that calls tools\launch.ps1 once per run with a PINNED commit and wraps
  everything in Start-Transcript (log next to the script), then register it here. The task wakes the PC from
  sleep (needs wake timers enabled), runs late if the PC was off (StartWhenAvailable), and is limited to 1 h.
  Claude still has to check the result after the start time (log, QUEUED, model loaded) — session crons only
  live while the Claude session is open.

.EXAMPLE
  & C:\temp_job\ai-experiment-control-center\tools\schedule_task.ps1 -TaskName myproj-launch-2026-12-05 `
    -At '2026-12-05T07:30:00' -Script C:\path\to\repo\.git\sched_launch.ps1
  # Inspect / remove:
  Get-ScheduledTask -TaskName myproj-launch-2026-12-05 | Get-ScheduledTaskInfo
  Unregister-ScheduledTask -TaskName myproj-launch-2026-12-05 -Confirm:$false
#>
param(
  [Parameter(Mandatory = $true)][string]$TaskName,
  [Parameter(Mandatory = $true)][datetime]$At,
  [Parameter(Mandatory = $true)][string]$Script,
  [string]$Description = 'One-shot scheduled launch (registered by Claude on user request)'
)
if (-not (Test-Path $Script)) { throw "Script not found: $Script" }
$ecc = Split-Path -Parent $PSScriptRoot
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ('-NoProfile -ExecutionPolicy Bypass -File "{0}"' -f $Script) -WorkingDirectory $ecc
$trigger = New-ScheduledTaskTrigger -Once -At $At
$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 1)
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Description $Description -Force | Out-Null
Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo | Format-List TaskName, NextRunTime, LastTaskResult
