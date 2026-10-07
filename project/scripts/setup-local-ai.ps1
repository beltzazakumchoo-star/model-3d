param([string]$InstallRoot = 'D:\FourViewAI')
$ErrorActionPreference = 'Stop'
$InstallRoot = [System.IO.Path]::GetFullPath($InstallRoot)
$drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($InstallRoot))
if ($drive.AvailableFreeSpace -lt 20GB) { throw 'AI setup requires at least 20 GB free on the selected drive.' }
foreach ($folder in @('app','runtime','models','cache','outputs','logs','temp','vendor')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $InstallRoot $folder) | Out-Null
}
$env:TEMP = Join-Path $InstallRoot 'temp'
$env:TMP = $env:TEMP
$env:UV_CACHE_DIR = Join-Path $InstallRoot 'cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $InstallRoot 'runtime\python'
$env:HF_HOME = Join-Path $InstallRoot 'cache\huggingface'
$env:U2NET_HOME = Join-Path $InstallRoot 'cache\rembg'
$env:FOURVIEW_AI_ROOT = $InstallRoot
$env:HF_HUB_DISABLE_XET = '1'
$env:HF_HUB_DOWNLOAD_TIMEOUT = '120'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$uvExe = Join-Path $InstallRoot 'runtime\uv\uv.exe'
if (-not (Test-Path -LiteralPath $uvExe)) {
    Write-Host '[1/6] Downloading uv runtime manager...'
    $zip = Join-Path $InstallRoot 'temp\uv.zip'
    Invoke-WebRequest -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile $zip
    Expand-Archive -LiteralPath $zip -DestinationPath (Join-Path $InstallRoot 'runtime\uv') -Force
}
Write-Host '[2/6] Preparing isolated Python 3.11...'
& $uvExe python install 3.11 --no-bin --no-registry
if ($LASTEXITCODE -ne 0) { throw 'Python installation failed.' }
$venv = Join-Path $InstallRoot 'runtime\venv'
if (-not (Test-Path -LiteralPath (Join-Path $venv 'Scripts\python.exe'))) {
    & $uvExe venv --python 3.11 $venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
$pythonExe = Join-Path $venv 'Scripts\python.exe'
Write-Host '[3/6] Installing CUDA 12.8 PyTorch for RTX 50 series...'
& $uvExe pip install --python $pythonExe 'torch==2.7.1' 'torchvision==0.22.1' --index-url 'https://download.pytorch.org/whl/cu128'
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
Write-Host '[4/6] Installing local generation service dependencies...'
& $uvExe pip install --python $pythonExe -r (Join-Path $PSScriptRoot '..\backend\requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Service dependency installation failed.' }
$repo = Join-Path $InstallRoot 'vendor\Hunyuan3D-2'
if (-not (Test-Path -LiteralPath (Join-Path $repo 'hy3dgen'))) {
    & git clone --depth 1 'https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git' $repo
    if ($LASTEXITCODE -ne 0) { throw 'Hunyuan3D source download failed.' }
}
& $uvExe pip install --python $pythonExe --no-deps --no-build-isolation -e $repo
if ($LASTEXITCODE -ne 0) { throw 'Hunyuan3D package installation failed.' }
Write-Host '[5/6] Downloading multiview weights and background-removal model...'
& $pythonExe (Join-Path $PSScriptRoot '..\backend\download_models.py')
if ($LASTEXITCODE -ne 0) { throw 'Model download failed. Re-run setup to resume.' }
Write-Host '[6/6] Checking GPU and model imports...'
& $pythonExe -c "import torch; from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0)); print((torch.ones(8, device='cuda') * 2).sum().item())"
if ($LASTEXITCODE -ne 0) { throw 'GPU compatibility check failed.' }
$config = @{root=$InstallRoot; python=$pythonExe; model='Hunyuan3D-2mv'} | ConvertTo-Json
$config | Set-Content -LiteralPath (Join-Path $PSScriptRoot '..\backend\local-config.json') -Encoding utf8
$project = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$installedApp = Join-Path $InstallRoot 'app'
foreach ($folder in @('backend','dist','scripts')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $installedApp $folder) | Out-Null
}
Get-ChildItem -LiteralPath (Join-Path $project 'backend') -File | Copy-Item -Destination (Join-Path $installedApp 'backend') -Force
Get-ChildItem -LiteralPath (Join-Path $project 'dist') | Copy-Item -Destination (Join-Path $installedApp 'dist') -Recurse -Force
Copy-Item -LiteralPath (Join-Path $project 'scripts\open-studio.ps1') -Destination (Join-Path $installedApp 'scripts') -Force
Copy-Item -LiteralPath (Join-Path $project 'scripts\stop-studio.ps1') -Destination (Join-Path $installedApp 'scripts') -Force
Copy-Item -LiteralPath (Join-Path $project 'scripts\setup-single-image.ps1') -Destination (Join-Path $installedApp 'scripts') -Force
$launcher = '@echo off' + "`r`n" + 'powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0app\scripts\open-studio.ps1" -InstallRoot "%~dp0."' + "`r`n" + 'if errorlevel 1 pause' + "`r`n"
$launcher | Set-Content -LiteralPath (Join-Path $InstallRoot 'Open FourView Studio.cmd') -Encoding ascii
$stopper = '@echo off' + "`r`n" + 'powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0app\scripts\stop-studio.ps1" -InstallRoot "%~dp0."' + "`r`n" + 'if errorlevel 1 pause' + "`r`n"
$stopper | Set-Content -LiteralPath (Join-Path $InstallRoot 'Stop Local AI.cmd') -Encoding ascii
Copy-Item -LiteralPath (Join-Path $project 'README.md') -Destination (Join-Path $InstallRoot 'README.md') -Force
Write-Host "Ready. Start with: powershell -ExecutionPolicy Bypass -File scripts\start-local-ai.ps1"

