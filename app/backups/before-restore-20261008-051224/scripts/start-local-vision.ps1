param([string]$InstallRoot = 'D:\FourViewAI')
$ErrorActionPreference = 'Stop'
$env:OLLAMA_HOST = '127.0.0.1:11435'
$env:OLLAMA_MODELS = Join-Path $InstallRoot 'models\local-vision'
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_MAX_LOADED_MODELS = '1'
$env:OLLAMA_NUM_PARALLEL = '1'
try { $null = Invoke-RestMethod 'http://127.0.0.1:11435/api/version' -TimeoutSec 2; return } catch {}
$exe = Join-Path $InstallRoot 'runtime\local-vision\ollama.exe'
if (-not (Test-Path -LiteralPath $exe)) { return }
$process = Start-Process -FilePath $exe -ArgumentList 'serve' -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $InstallRoot 'logs\local-vision-out.log') -RedirectStandardError (Join-Path $InstallRoot 'logs\local-vision-error.log')
$process.Id | Set-Content -LiteralPath (Join-Path $InstallRoot 'runtime\local-vision\server.pid')
