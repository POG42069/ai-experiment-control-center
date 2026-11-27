<#
.SYNOPSIS
  Event-driven monitoring helper: polls Kaggle kernel status + W&B run state/metrics and appends one line per
  fact to a log file, which a Claude Monitor reads by snapshot (open-read-close, never tail -f).

.DESCRIPTION
  Shared by all projects (moved from MMCultureQA's .git/ecc_watch.ps1, 2026-11-27). Run it in its own
  terminal tab; close the tab when the runs finish. Log line formats:
    HH:mm:ss KAGGLE <account> <STATUS>
    HH:mm:ss WANDB <run-name> <STATE> eval_epoch=.. f1=.. step=.. stage=..
  -Runs are W&B run names (= ECC_RUN_NAME / the --run-name used at launch).

.EXAMPLE
  & C:\temp_job\ai-experiment-control-center\tools\watch.ps1 -Accounts kaggle-2,kaggle-3 `
    -Runs x15-run,x5-run -Project my-wandb-project -Log C:\path\to\repo\.git\ecc_watch.log
#>
param(
  [Parameter(Mandatory = $true)][string[]]$Accounts,
  [string[]]$Runs = @(),
  [Parameter(Mandatory = $true)][string]$Project,
  [Parameter(Mandatory = $true)][string]$Log,
  [int]$IntervalSec = 180,
  [string]$MetricKey = 'eval/bertscore_f1',
  [switch]$Once
)
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = '.\.venv\Scripts\python.exe'
function Write-WatchLine([string]$Text) {
  for ($i = 0; $i -lt 5; $i++) {
    try { Add-Content -Path $Log -Value $Text -ErrorAction Stop; return } catch { Start-Sleep -Milliseconds 500 }
  }
}
while ($true) {
  $ts = Get-Date -Format 'HH:mm:ss'
  foreach ($a in $Accounts) {
    try {
      $j = (& $py -m ecc.cli launch status --account $a | Out-String | ConvertFrom-Json)
      Write-WatchLine ("{0} KAGGLE {1} {2}" -f $ts, $a, $j.status)
    } catch { Write-WatchLine ("{0} KAGGLE {1} STATUS_ERROR" -f $ts, $a) }
  }
  if ($Runs.Count -gt 0) {
    try {
      $runsJson = (& $py -m ecc.cli list-runs --project $Project | Out-String | ConvertFrom-Json)
      foreach ($r in $runsJson) {
        if ($Runs -contains $r.name) {
          $f1 = $r.latest.$MetricKey; $ep = $r.latest.'eval/epoch'; $step = $r.latest.'train/global_step'
          $stage = $r.latest.'infer/stage'
          Write-WatchLine ("{0} WANDB {1} {2} eval_epoch={3} f1={4} step={5} stage={6}" -f $ts, $r.name, $r.status, $ep, $f1, $step, $stage)
        }
      }
    } catch { Write-WatchLine ("{0} WANDB LIST_ERROR" -f $ts) }
  }
  if ($Once) { break }
  Start-Sleep -Seconds $IntervalSec
}
