<#
.SYNOPSIS
  Launch one ECC Kaggle run the way every successful launch was done (plan -> show summary -> execute).

.DESCRIPTION
  Shared by all projects (moved from MMCultureQA's .git/ecc_launch.ps1, 2026-11-27). Always passes
  --commit, --requirements (if the repo has a requirements file) and --monitor-mode script. A launch without
  --requirements runs on Kaggle's default libraries and can die at model load (2026-11-26 incident).
  Must run in a real terminal tab (needs the ECC account DB in %LOCALAPPDATA%).

  If -Cmd contains ; | $ or nested quotes, do NOT use this script: create `launch plan` by hand (inner double
  quotes written as \"...\"), read the Command line in the plan, then `launch execute --plan-id` (CLAUDE.md 8.3).

.EXAMPLE
  & C:\temp_job\ai-experiment-control-center\tools\launch.ps1 -Account kaggle-2 -Repo POG42069/my-repo `
    -Branch exp/x -Commit <full-sha> -Project my-wandb-project -RunName x15-run -TimeoutHours 12 `
    -Cmd 'bash scripts/kaggle_run.sh configs/x15.yaml'

.EXAMPLE
  # Plan only (no GPU quota used): prints the summary and the plan id, does not execute.
  & ...\tools\launch.ps1 ... -PlanOnly
#>
param(
  [Parameter(Mandatory = $true)][string]$Account,
  [Parameter(Mandatory = $true)][string]$Repo,
  [Parameter(Mandatory = $true)][string]$Branch,
  [Parameter(Mandatory = $true)][string]$Commit,
  [Parameter(Mandatory = $true)][string]$Project,
  [Parameter(Mandatory = $true)][string]$RunName,
  [Parameter(Mandatory = $true)][string]$Cmd,
  [double]$TimeoutHours = 2,
  [string]$Accelerator = 'NvidiaTeslaT4',
  [string]$Requirements = 'requirements.txt',
  [switch]$NoRequirements,
  [string]$MonitorMode = 'script',
  [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
if ($Cmd -match '[;|$]') {
  throw "Cmd contains ; | or `$ - create the plan by hand (see the help of this script / CLAUDE.md 8.3)."
}
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = '.\.venv\Scripts\python.exe'
$planArgs = @('-m', 'ecc.cli', 'launch', 'plan', '--account', $Account, '--repo', $Repo, '--branch', $Branch,
  '--commit', $Commit, '--command', $Cmd, '--project', $Project, '--run-name', $RunName,
  '--accelerator', $Accelerator, '--timeout-hours', ([string]$TimeoutHours), '--monitor-mode', $MonitorMode)
if (-not $NoRequirements) { $planArgs += @('--requirements', $Requirements) }
$plan = (& $py @planArgs | Out-String | ConvertFrom-Json)
if (-not $plan.plan_id) { throw "launch plan failed" }
Write-Host $plan.summary
Write-Host ("Requirements : {0}" -f $(if ($NoRequirements) { 'NONE (-NoRequirements)' } else { $Requirements }))
Write-Host ("PLAN_ID {0}" -f $plan.plan_id)
if ($PlanOnly) { Write-Host 'PLAN_ONLY - not executed'; return }
$res = (& $py -m ecc.cli launch execute --plan-id $plan.plan_id | Out-String | ConvertFrom-Json)
Write-Host ("LAUNCHED {0} ok={1} version={2} at {3} commit={4} url={5}" -f $Account, $res.ok, $res.version,
  (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $res.commit_sha, $res.url)
if (-not $res.ok) { Write-Host $res.message }
