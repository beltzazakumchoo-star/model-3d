"""Lightweight checks for the verified, offline TRELLIS v1 installation."""
import json
from pathlib import Path

SNAPSHOT = 'ab6a010207b4ceab566c5d8927d90a6f69c5f976'


def readiness(root):
    root = Path(root)
    cache = root / 'trellis1/cache'
    snapshot = cache / 'huggingface/hub/models--jetx--TRELLIS-image-large/snapshots' / SNAPSHOT
    result = {'trellis1_ready': False, 'trellis1_reason': 'TRELLIS ยังติดตั้งไม่ครบ'}
    try:
        spec = json.loads((snapshot / 'pipeline.json').read_text(encoding='utf-8'))['args']
        paths = [root / 'runtime/trellis1/verified.json',
                 cache / 'torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth',
                 cache / 'torch/hub/facebookresearch_dinov2_main/hubconf.py',
                 root / 'vendor/trellis1/bundle/code/trellis/pipelines/trellis_image_to_3d.py']
        for name, value in spec['models'].items():
            if name == 'slat_decoder_rf':
                continue
            paths.extend([snapshot / (value + suffix) for suffix in ('.json', '.safetensors')])
        if not all(p.is_file() and p.stat().st_size > 0 for p in paths):
            return result
        import torch
        if not torch.cuda.is_available():
            result['trellis1_reason'] = 'TRELLIS ต้องใช้ CUDA GPU'
            return result
        verified = json.loads(paths[0].read_text(encoding='utf-8'))
        if verified.get('snapshot') != SNAPSHOT or not verified.get('inference_verified'):
            return result
        result.update(trellis1_ready=True, trellis1_reason='', trellis1_version='v1-dinov2')
    except (OSError, ValueError, KeyError):
        pass
    return result
