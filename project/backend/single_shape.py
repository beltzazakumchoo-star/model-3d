"""Isolated single-image Hunyuan3D-DiT inference, leaving no resident GPU model."""
import json
import os
from pathlib import Path


def report(folder, stage, step=0, steps=0):
    path = folder / 'worker-progress.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'stage': stage, 'step': step, 'steps': steps}, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)
    print(stage, flush=True)


def generate(folder, steps):
    import torch
    from PIL import Image
    from backend.model_loader import load_multiview
    root = Path(os.environ.get('FOURVIEW_AI_ROOT', 'D:/FourViewAI'))
    report(folder, 'กำลังโหลด Hunyuan3D สำหรับภาพเดียว…')
    engine = load_multiview(root / 'models/Hunyuan3D-2', 'hunyuan3d-dit-v2-0')
    def progress(step, _time, _value):
        report(folder, f'สร้างรูปทรงจากภาพเดียว · {step+1}/{steps}', step+1, steps)
    saved = json.loads((folder / 'job.json').read_text(encoding='utf-8'))
    with Image.open(folder / 'front-cutout.png') as source:
        with torch.inference_mode():
            latents = engine(image=source.convert('RGBA'), num_inference_steps=steps,
                generator=torch.Generator(device='cpu').manual_seed(saved.get('seed', 12345)), output_type='latent',
                callback=progress, callback_steps=1)
    torch.save(latents.detach().cpu(), folder / 'shape-latents.pt')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--steps', type=int, default=40)
    args = parser.parse_args()
    generate(args.folder, args.steps)
