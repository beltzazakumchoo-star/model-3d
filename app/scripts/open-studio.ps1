param([string]$InstallRoot = 'D:\FourViewAI', [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$project = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$env:FOURVIEW_AI_ROOT = [System.IO.Path]::GetFullPath($InstallRoot)
$env:HF_HOME = Join-Path $InstallRoot 'cache\huggingface'
$env:U2NET_HOME = Join-Path $InstallRoot 'cache\rembg'
$env:TEMP = Join-Path $InstallRoot 'temp'
$env:TMP = $env:TEMP
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$pythonExe = Join-Path $InstallRoot 'runtime\venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Run setup-local-ai.ps1 first.' }
& (Join-Path $PSScriptRoot 'start-local-vision.ps1') -InstallRoot $InstallRoot
$running = $false
try { $null = Invoke-RestMethod 'http://127.0.0.1:8008/api/health' -TimeoutSec 3; $running = $true } catch {}
if (-not $running) {
    $process = Start-Process -FilePath $pythonExe -ArgumentList '-X','faulthandler','-u','-m','uvicorn','backend.server:app','--host','127.0.0.1','--port','8008' -WorkingDirectory $project -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $InstallRoot 'logs\server-out.log') -RedirectStandardError (Join-Path $InstallRoot 'logs\server-error.log')
    $process.Id | Set-Content -LiteralPath (Join-Path $InstallRoot 'runtime\server.pid')
    @{pid=$process.Id; started=$process.StartTime.ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $InstallRoot 'runtime\server.json') -Encoding utf8
    for ($attempt=0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Seconds 1
        if ($process.HasExited) { throw ('Local AI exited. See ' + (Join-Path $InstallRoot 'logs\server-error.log')) }
        try { $null = Invoke-RestMethod 'http://127.0.0.1:8008/api/health' -TimeoutSec 2; $running = $true; break } catch {}
    }
}
if (-not $running) { throw 'Local AI did not start within 30 seconds. See logs on the selected drive.' }
if (-not $NoBrowser) { Start-Process 'http://127.0.0.1:8008' }
