"""Download pinned official single-image shape and paint weights, with checksums."""
import importlib.util
import json
from pathlib import Path
import requests

REPO = 'tencent/Hunyuan3D-2'
REVISION = '9cd649ba6913f7a852e3286bad86bfa9a2d83dcf'


def install(root):
    target = root / 'models/Hunyuan3D-2'
    response = requests.get(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true', timeout=60)
    response.raise_for_status()
    manifest = response.json()
    selected = []
    for file in manifest['siblings']:
        name = file['rfilename']
        wanted = name in ('LICENSE', 'NOTICE', 'README.md',
            'hunyuan3d-dit-v2-0/config.yaml', 'hunyuan3d-dit-v2-0/model.fp16.safetensors')
        wanted |= name.startswith('hunyuan3d-paint-v2-0/') and (
            name.endswith(('.json', '.txt', '.safetensors')) or name.endswith('text_encoder/pytorch_model.bin'))
        if wanted:
            selected.append(file)
    target.mkdir(parents=True, exist_ok=True)
    for file in selected:
        name = file['rfilename']
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
        print(name, flush=True)
        if 'lfs' in file:
            # Reuse the resumable, range-validated downloader in the app.
            spec = importlib.util.spec_from_file_location('weight_download', root / 'project/backend/checkpoint_download.py')
            download = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(download)
            download.FILE, download.URL = name, url
            download.SIZE, download.SHA256 = file['size'], file['lfs']['sha256']
            download.REVISION = REVISION
            download.download_checkpoint(target)
        else:
            data = requests.get(url, timeout=60)
            data.raise_for_status()
            destination.write_bytes(data.content)
    (target / 'installation.json').write_text(json.dumps({
        'repo': REPO, 'revision': REVISION, 'files': selected}, indent=2), encoding='utf-8')
    print('Official shape and paint weights installed and verified.', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('D:/FourViewAI'))
    install(parser.parse_args().root)
