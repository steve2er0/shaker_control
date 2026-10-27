[CmdletBinding()]
param([switch]$Monitor, [switch]$CheckOnly)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$RepoRoot = Split-Path -Parent $PSScriptRoot
$EnvironmentPython = Join-Path $RepoRoot '.venv-py313\Scripts\python.exe'
$EntryPoint = if ($Monitor) { 'accel_monitor.py' } else { 'app.py' }
$AppName = if ($Monitor) { 'monitor' } else { 'controller' }

try {
    if (-not (Test-Path -LiteralPath $EnvironmentPython)) {
        throw 'Run setup-windows.cmd first. The .venv-py313 Windows environment is missing.'
    }
    $LogDirectory = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'ShakerControl\Logs'
    New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
    $LogFile = Join-Path $LogDirectory ("{0}-{1}-{2}.log" -f $AppName, (Get-Date -Format 'yyyyMMdd-HHmmss'), $PID)
    Write-Host "Application log: $LogFile"
    $PythonArguments = @('-u', (Join-Path $RepoRoot $EntryPoint))
    if ($CheckOnly) {
        $Module = if ($Monitor) { 'accel_monitor' } else { 'app' }
        $PythonArguments = @('-u', '-c', "import importlib, sys; assert sys.version_info[:2] == (3, 13); importlib.import_module('$Module'); print('$AppName startup imports OK')")
    }
    Push-Location -LiteralPath $RepoRoot
    try {
        # Windows PowerShell 5.1 treats native stderr as ErrorRecords. Qt can
        # write harmless diagnostics there, so use the process exit code.
        $ErrorActionPreference = 'Continue'
        & $EnvironmentPython @PythonArguments 2>&1 | Tee-Object -FilePath $LogFile
        $ApplicationExitCode = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
    } finally {
        Pop-Location
    }
    if ($ApplicationExitCode -ne 0) { throw "Application exited with code $ApplicationExitCode. See $LogFile" }
    exit 0
} catch {
    Write-Host "Could not run Shaker Control: $_" -ForegroundColor Red
    exit 1
}
