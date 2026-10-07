param([string]$InstallRoot)
$ErrorActionPreference = 'Stop'
$configPath = Join-Path $PSScriptRoot '..\backend\local-config.json'
if (-not $InstallRoot) {
    if (-not (Test-Path -LiteralPath $configPath)) { throw 'Run scripts\setup-local-ai.ps1 first.' }
    $InstallRoot = (Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json).root
}
$env:FOURVIEW_AI_ROOT = [System.IO.Path]::GetFullPath($InstallRoot)
$env:HF_HOME = Join-Path $InstallRoot 'cache\huggingface'
$env:U2NET_HOME = Join-Path $InstallRoot 'cache\rembg'
$env:TEMP = Join-Path $InstallRoot 'temp'
$env:TMP = $env:TEMP
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$pythonExe = Join-Path $InstallRoot 'runtime\venv\Scripts\python.exe'
Set-Location -LiteralPath (Join-Path $PSScriptRoot '..')
& $pythonExe -m uvicorn backend.server:app --host 127.0.0.1 --port 8008
