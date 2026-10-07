param([string]$InstallRoot = 'D:\FourViewAI')
$ErrorActionPreference = 'Stop'
$InstallRoot = [System.IO.Path]::GetFullPath($InstallRoot)
$recordPath = Join-Path $InstallRoot 'runtime\server.json'
if (-not (Test-Path -LiteralPath $recordPath)) { throw 'No launcher-owned AI process was recorded.' }
$record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
$process = Get-Process -Id ([int]$record.pid) -ErrorAction SilentlyContinue
if (-not $process) { Write-Host 'Local AI is already stopped.'; exit }
$expected = ([datetime]$record.started).ToUniversalTime()
$runtime = (Join-Path $InstallRoot 'runtime') + '\'
if ($process.StartTime.ToUniversalTime() -ne $expected -or -not $process.Path.StartsWith($runtime, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Process identity changed; no process was stopped.'
}
# Windows venv uses a launcher and interpreter child. Stop this owned tree.
& taskkill.exe /PID $process.Id /T /F
if ($LASTEXITCODE -ne 0) { throw 'The AI process could not be stopped.' }
