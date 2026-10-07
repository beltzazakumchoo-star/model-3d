"""Readiness for the isolated TRELLIS runtime; never auto-download gated weights."""
import json
from pathlib import Path

DINO_URL = 'https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m'
TIERS = {'draft': 'draft', 'balanced': 'standard', 'detail': 'high'}


def readiness(root):
    root = Path(root)
    runtime = root / 'runtime/trellis2'
    models = root / 'trellis2/models'
    reasons = []
    if not (runtime / 'runtime-verified.json').is_file():
        reasons.append('กำลังเตรียม runtime TRELLIS.2')
    manifest = root / 'trellis2/weights-source.json'
    try:
        files = json.loads(manifest.read_text(encoding='utf-8')).get('files', {})
        weights_ready = bool(files) and all((models / name).is_file() and
            (models / name).stat().st_size == size for name, size in files.items())
    except (OSError, ValueError, TypeError):
        weights_ready = False
    if not weights_ready:
        reasons.append('กำลังดาวน์โหลดโมเดล TRELLIS.2')
    dino = models / 'facebook/dinov3-vitl16-pretrain-lvd1689m'
    dino_ready = all((dino / name).is_file() for name in
                     ('model.safetensors', 'config.json', 'preprocessor_config.json'))
    if not dino_ready:
        reasons.append('ต้องขอสิทธิ์ DINOv3 จาก Meta และดาวน์โหลดด้วยบัญชีที่ได้รับสิทธิ์')
    return {'trellis_ready': not reasons, 'trellis_reason': ' · '.join(reasons),
            'trellis_dino_ready': dino_ready, 'trellis_access_url': DINO_URL}


def write_progress(folder, stage, fraction):
    fraction = max(0, min(1, fraction))
    file = Path(folder) / 'worker-progress.json'
    temporary = file.with_suffix('.tmp')
    temporary.write_text(json.dumps({'stage': 'TRELLIS.2 · ' + stage,
                                    'steps': 100, 'step': round(fraction * 100)}), encoding='utf-8')
    temporary.replace(file)
