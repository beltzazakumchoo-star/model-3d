param(
    [string]$InstallRoot = 'D:\FourViewAI',
    [string]$Branch = 'main',
    [string]$Repository = 'beltzazakumchoo-star/model-3d'
)
# Replace app\backend\*.py and app\scripts\*.ps1 with the GitHub branch, then
# restart Local AI so the running server loads the new code.
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$InstallRoot = [System.IO.Path]::GetFullPath($InstallRoot)
$app = Join-Path $InstallRoot 'app'
if (-not (Test-Path -LiteralPath (Join-Path $app 'backend\server.py'))) { throw "Not found: $app\backend\server.py" }

$staging = Join-Path $env:TEMP ('fourview-update-' + [guid]::NewGuid())
$downloads = @()
foreach ($folder in 'backend', 'scripts') {
    $listing = Invoke-RestMethod "https://api.github.com/repos/$Repository/contents/app/$folder`?ref=$Branch" -Headers @{'User-Agent'='FourView'}
    foreach ($item in $listing) {
        if ($item.type -eq 'file' -and $item.name -match '\.(py|ps1|txt)$') {
            $target = Join-Path $staging "$folder\$($item.name)"
            New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
            Invoke-WebRequest $item.download_url -OutFile $target -UseBasicParsing
            $downloads += [pscustomobject]@{Source=$target; Target=(Join-Path $app "$folder\$($item.name)")}
        }
    }
}
if (-not ($downloads | Where-Object { $_.Target -like '*backend\server.py' })) { throw 'Download incomplete; no files were changed.' }

$stop = Join-Path $app 'scripts\stop-studio.ps1'
try { & $stop -InstallRoot $InstallRoot } catch { Write-Host $_.Exception.Message }
Start-Sleep -Seconds 2
foreach ($file in $downloads) { Copy-Item -LiteralPath $file.Source -Destination $file.Target -Force }
Get-ChildItem -LiteralPath (Join-Path $app 'backend') -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force
Remove-Item -LiteralPath $staging -Recurse -Force
Write-Host "Updated $($downloads.Count) files from $Branch. Restarting FourView Studio..."
& (Join-Path $app 'scripts\open-studio.ps1') -InstallRoot $InstallRoot
