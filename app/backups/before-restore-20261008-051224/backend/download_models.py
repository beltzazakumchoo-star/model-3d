"""Download only the fp16 multiview model; all caches live on the selected drive."""
import os
from pathlib import Path

root = Path(os.environ.get('FOURVIEW_AI_ROOT', 'D:/FourViewAI'))
os.environ['HF_HOME'] = str(root / 'cache' / 'huggingface')
os.environ['U2NET_HOME'] = str(root / 'cache' / 'rembg')
os.environ['HF_XET_CACHE'] = str(root / 'cache' / 'huggingface' / 'xet')
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import snapshot_download
from rembg import new_session
from checkpoint_download import REVISION, download_checkpoint

snapshot_download(
    repo_id='tencent/Hunyuan3D-2mv',
    revision=REVISION,
    local_dir=str(root / 'models' / 'Hunyuan3D-2mv'),
    allow_patterns=['hunyuan3d-dit-v2-mv/config.yaml', 'LICENSE', 'NOTICE'],
)
download_checkpoint(root / 'models' / 'Hunyuan3D-2mv')
new_session('u2net', providers=['CPUExecutionProvider'])
print('Multiview checkpoint and background removal model are ready.')
