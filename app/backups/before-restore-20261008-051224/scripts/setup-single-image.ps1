param([string]$InstallRoot = 'D:\FourViewAI')
$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$pythonExe = Join-Path $InstallRoot 'runtime\venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Install the local AI runtime first.' }
$env:PYTHONUTF8 = '1'
& $pythonExe -u (Join-Path $projectRoot 'backend\setup_single_image.py') --root $InstallRoot
if ($LASTEXITCODE -ne 0) { throw 'Single-image model installation failed; run again to resume downloads.' }
