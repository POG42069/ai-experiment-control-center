[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$RequirementsFile = Join-Path $RepoRoot "requirements.txt"
$MonitorDir = Join-Path $RepoRoot "monitor"
$ExampleConfig = Join-Path $RepoRoot "ecc_config.example.toml"
$LocalConfig = Join-Path $RepoRoot "ecc_config.toml"

Write-Host "[ECC Setup] Repository root: $RepoRoot"

# 1. Create virtual environment if missing
if (-not (Test-Path $VenvPython)) {
    Write-Host "[ECC Setup] Creating virtual environment at $VenvDir ..."
    $pyLauncher = Get-Command "py" -ErrorAction SilentlyContinue
    $pythonCmd = Get-Command "python" -ErrorAction SilentlyContinue

    if ($null -ne $pyLauncher) {
        & py -3 -m venv $VenvDir
    }
    elseif ($null -ne $pythonCmd) {
        & python -m venv $VenvDir
    }
    else {
        Write-Error "Python 3 was not found on PATH (tried 'py -3' and 'python'). Install Python 3.10+ and re-run .\setup.ps1."
        exit 1
    }

    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VenvPython)) {
        Write-Error "Failed to create virtual environment at $VenvDir."
        exit 1
    }
}
else {
    Write-Host "[ECC Setup] Virtual environment already exists: $VenvDir"
}

# 2. Install pinned dashboard & backend dependencies
if (-not (Test-Path $RequirementsFile)) {
    Write-Error "Missing requirements.txt at $RequirementsFile"
    exit 1
}

Write-Host "[ECC Setup] Installing dependencies from requirements.txt ..."
& $VenvPython -m pip install -r $RequirementsFile
if ($LASTEXITCODE -ne 0) {
    Write-Error "pip install -r requirements.txt failed with exit code $LASTEXITCODE."
    exit 1
}

# 3. Install monitor package (editable) idempotently
$monitorAlreadyEditable = $false
if (-not $Force) {
    $checkCode = @'
import importlib.metadata, json, os, sys
from pathlib import Path
from urllib.parse import unquote, urlparse

try:
    dist = importlib.metadata.distribution("experiment-monitor")
    raw = dist.read_text("direct_url.json")
    if raw:
        info = json.loads(raw)
        url = info.get("url", "")
        is_editable = bool(info.get("dir_info", {}).get("editable", False))
        parsed = urlparse(url)
        loc = unquote(parsed.path)
        if os.name == "nt" and loc.startswith("/") and len(loc) > 2 and loc[2] == ":":
            loc = loc[1:]
        expected = Path(sys.argv[1]).resolve()
        actual = Path(loc).resolve()
        if is_editable and actual == expected:
            sys.exit(0)
except Exception:
    pass
sys.exit(1)
'@
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $checkCode | & $VenvPython - $MonitorDir 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $monitorAlreadyEditable = $true
        }
    }
    finally {
        $ErrorActionPreference = $prevEap
    }
}

if ($monitorAlreadyEditable) {
    Write-Host "[ECC Setup] Package 'experiment-monitor' is already installed in editable mode from $MonitorDir"
}
else {
    Write-Host "[ECC Setup] Installing 'experiment-monitor' in editable mode from $MonitorDir ..."
    & $VenvPython -m pip install -e $MonitorDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "pip install -e monitor failed with exit code $LASTEXITCODE."
        exit 1
    }
}

# 4. Copy ecc_config.example.toml -> ecc_config.toml only if absent
if (-not (Test-Path $LocalConfig)) {
    if (Test-Path $ExampleConfig) {
        Copy-Item -Path $ExampleConfig -Destination $LocalConfig
        Write-Host "[ECC Setup] Created $LocalConfig from ecc_config.example.toml (no secrets)."
    }
    else {
        Write-Warning "ecc_config.example.toml not found; skipping config copy."
    }
}
else {
    Write-Host "[ECC Setup] Config already exists (preserved): $LocalConfig"
}

Write-Host ""
Write-Host "[ECC Setup] Done."
Write-Host "  - Launch dashboard : .\run_dashboard.ps1"
Write-Host "  - Launch demo mode : .\run_dashboard.ps1 -Demo"
Write-Host "  - Add accounts     : .\.venv\Scripts\python.exe scripts\add_account.py --help"
Write-Host "  - Run tests        : .\.venv\Scripts\python.exe -m pytest -q"
exit 0
