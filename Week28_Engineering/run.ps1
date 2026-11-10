$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$localPython = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$runtime = Join-Path $PSScriptRoot '.runtime'
if (Test-Path -LiteralPath $venvPython) {
    & $venvPython (Join-Path $PSScriptRoot 'app.py') @args
} elseif ((Test-Path -LiteralPath $localPython) -and (Test-Path -LiteralPath $runtime)) {
    $env:PYTHONPATH = $runtime
    & $localPython (Join-Path $PSScriptRoot 'app.py') @args
} else {
    & python (Join-Path $PSScriptRoot 'app.py') @args
}
exit $LASTEXITCODE
