[CmdletBinding()]
param(
    [switch]$Development,
    [switch]$Hardware,
    [string]$PythonExecutable
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$RepoRoot = Split-Path -Parent $PSScriptRoot
$EnvironmentDirectory = Join-Path $RepoRoot '.venv-py313'
$EnvironmentPython = Join-Path $EnvironmentDirectory 'Scripts\python.exe'
$CheckPython = "import struct, sys; assert sys.version_info[:2] == (3, 13) and struct.calcsize('P') == 8, 'Use 64-bit CPython 3.13'; print(sys.executable)"

try {
    if (-not (Test-Path -LiteralPath $EnvironmentPython)) {
        if (Test-Path -LiteralPath $EnvironmentDirectory) {
            throw 'The .venv-py313 folder exists without a Windows interpreter. Move it aside, then run setup again.'
        }
        if (-not $PythonExecutable) {
            if (Get-Command py -ErrorAction SilentlyContinue) {
                $PythonExecutable = & py -3.13 -c $CheckPython
                if ($LASTEXITCODE -ne 0) { throw 'Install 64-bit Python 3.13 from python.org, then run setup again.' }
            } else {
                $PythonExecutable = (Get-Command python -ErrorAction Stop).Source
            }
        }
        & $PythonExecutable -c $CheckPython
        if ($LASTEXITCODE -ne 0) { throw 'The selected interpreter must be 64-bit CPython 3.13.' }
        & $PythonExecutable -m venv $EnvironmentDirectory
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
    }

    & $EnvironmentPython -c $CheckPython
    if ($LASTEXITCODE -ne 0) { throw 'The existing .venv-py313 environment must use 64-bit CPython 3.13.' }
    $Requirements = if ($Development) { 'requirements-dev.txt' } else { 'requirements.txt' }
    & $EnvironmentPython -m pip install --only-binary=:all: -r (Join-Path $RepoRoot $Requirements)
    if ($LASTEXITCODE -ne 0) { throw 'Runtime dependency installation failed.' }
    if ($Hardware) {
        & $EnvironmentPython -m pip install --only-binary=:all: -r (Join-Path $RepoRoot 'requirements-hardware.txt')
        if ($LASTEXITCODE -ne 0) { throw 'Optional NI Python binding installation failed.' }
    }
    & $EnvironmentPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Installed dependencies are inconsistent.' }
    Push-Location -LiteralPath $RepoRoot
    try {
        & $EnvironmentPython -c "import app, accel_monitor; print('Desktop startup imports OK')"
        if ($LASTEXITCODE -ne 0) { throw 'Desktop imports failed. Review the Python/DLL error above before launching.' }
    } finally {
        Pop-Location
    }
    Write-Host 'Setup complete. Open start-controller.cmd or start-monitor.cmd.'
    if ($Hardware) { Write-Host 'NI hardware also requires the compatible NI-DAQmx driver installed separately.' }
    exit 0
} catch {
    Write-Host "Setup failed: $_" -ForegroundColor Red
    exit 1
}
