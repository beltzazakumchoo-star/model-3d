"""Install the pinned Wonder3D view generator without changing the shape runtime."""
import os
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
root = Path(os.environ.get('FOURVIEW_AI_ROOT', 'D:/FourViewAI'))
os.environ['HF_HOME'] = str(root / 'cache/huggingface')
from huggingface_hub import snapshot_download

for repo, revision, destination, patterns in [
    ('flamehaze1115/wonder3d-pipeline', '3d356e9db3d3ad838b656210c66c9d8fd6f1936a',
     root / 'vendor/Wonder3D-pipeline', ['pipeline.py', 'mvdiffusion/models/*.py', 'README.md']),
    ('flamehaze1115/wonder3d-v1.0', 'd6d2efc033a06a74d3761268de7295c97e6935d2',
     root / 'models/Wonder3D', ['*.json', '**/*.json', '**/*.bin', 'README.md']),
]:
    print('Installing ' + repo, flush=True)
    snapshot_download(repo, revision=revision, local_dir=str(destination),
        allow_patterns=patterns, max_workers=3)
    (destination / 'PINNED_REVISION.txt').write_text(repo + '\n' + revision, encoding='utf-8')
print('View generator downloaded.', flush=True)
