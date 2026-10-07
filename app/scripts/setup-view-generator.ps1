param([string]$InstallRoot = 'D:\FourViewAI')
$ErrorActionPreference = 'Stop'
$InstallRoot = [System.IO.Path]::GetFullPath($InstallRoot)
$env:FOURVIEW_AI_ROOT = $InstallRoot
$env:HF_HOME = Join-Path $InstallRoot 'cache\huggingface'
$env:UV_CACHE_DIR = Join-Path $InstallRoot 'cache\uv'
$env:TEMP = Join-Path $InstallRoot 'temp'
$env:TMP = $env:TEMP
$python = Join-Path $InstallRoot 'runtime\venv\Scripts\python.exe'
$uv = Join-Path $InstallRoot 'runtime\uv\uv.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Install the local AI runtime first.' }
& $uv pip install --python $python --target (Join-Path $InstallRoot 'vendor\Wonder3D-deps') --no-deps 'diffusers==0.19.3'
if ($LASTEXITCODE -ne 0) { throw 'View runtime installation failed.' }
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    & $python -u -m backend.setup_views
    if ($LASTEXITCODE -ne 0) { throw 'View model download failed. Run this installer again to continue.' }
} finally { Pop-Location }
