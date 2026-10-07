param([string]$InstallRoot = 'D:\FourViewAI')
# Undo update-backend.ps1: it replaced newer local app\backend and app\scripts
# files with older GitHub copies. project\ was not touched and is kept in sync
# with app\, so copy it back, then re-apply only the paint crash fixes.
$ErrorActionPreference = 'Stop'
$InstallRoot = [System.IO.Path]::GetFullPath($InstallRoot)
$app = Join-Path $InstallRoot 'app'
$project = Join-Path $InstallRoot 'project'
$python = Join-Path $InstallRoot 'runtime\venv\Scripts\python.exe'
$patcher = Join-Path $PSScriptRoot 'apply_paint_fix.py'
foreach ($path in (Join-Path $project 'backend\server.py'), $python, $patcher) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Not found: $path" }
}

if (-not (Select-String -LiteralPath (Join-Path $project 'backend\server.py') -Pattern 'qwen' -Quiet)) {
    Write-Host 'project\backend\server.py does not contain the Qwen3-VL code. Python files that mention Qwen:'
    Get-ChildItem -LiteralPath $app, $project -Recurse -Filter '*.py' -File -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch '\\node_modules\\' } |
        Select-String -Pattern 'qwen' -List |
        ForEach-Object { '{0}   {1}' -f (Get-Item $_.Path).LastWriteTime, $_.Path }
    throw 'Nothing was changed. Please send a screenshot of this output.'
}

$stop = Join-Path $app 'scripts\stop-studio.ps1'
try { & $stop -InstallRoot $InstallRoot } catch { Write-Host $_.Exception.Message }
Start-Sleep -Seconds 2

$backup = Join-Path $app ('backups\before-restore-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Force -Path (Join-Path $backup 'backend'), (Join-Path $backup 'scripts') | Out-Null
Copy-Item -Path (Join-Path $app 'backend\*.py') -Destination (Join-Path $backup 'backend')
Copy-Item -Path (Join-Path $app 'scripts\*.ps1') -Destination (Join-Path $backup 'scripts')

$restored = 0
foreach ($folder in 'backend', 'scripts') {
    $pattern = if ($folder -eq 'backend') { '*.py' } else { '*.ps1' }
    foreach ($file in Get-ChildItem -Path (Join-Path $project "$folder\$pattern") -File) {
        $target = Join-Path $app "$folder\$($file.Name)"
        # Restore every backend file; for scripts only those app\ already has.
        if ($folder -eq 'backend' -or (Test-Path -LiteralPath $target)) {
            Copy-Item -LiteralPath $file.FullName -Destination $target -Force
            $restored++
        }
    }
}
Write-Host "Restored $restored files from project\. Previous app files saved in $backup"

& $python -I $patcher (Join-Path $app 'backend') (Join-Path $project 'backend')
if ($LASTEXITCODE -ne 0) { throw 'Applying the paint fix failed; the restored files are unchanged otherwise.' }
foreach ($folder in (Join-Path $app 'backend'), (Join-Path $project 'backend')) {
    Get-ChildItem -LiteralPath $folder -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force
}
Write-Host 'Starting FourView Studio...'
& (Join-Path $app 'scripts\open-studio.ps1') -InstallRoot $InstallRoot
