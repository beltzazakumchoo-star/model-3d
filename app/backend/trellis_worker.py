"""Native 3D shape/material generation, with no photo projection or jaw warping."""
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path


def run(folder, engine_factory=None, settings_factory=None):
    folder = Path(folder)
    root = Path(os.environ.get('FOURVIEW_AI_ROOT', 'D:/FourViewAI'))
    os.environ['LOCALMESH_ROOT'] = str(root / 'trellis2')
    os.environ['HF_HOME'] = str(root / 'trellis2/cache/huggingface')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(root / 'vendor/localmesh-engine'))
    from backend.trellis_support import TIERS, write_progress
    if engine_factory is None:
        from localmesh_engine import Engine, GenerateSettings
        engine_factory, settings_factory = Engine, GenerateSettings
    job = json.loads((folder / 'job.json').read_text(encoding='utf-8'))
    # Use the uploaded cutout as the only conditioning image. The engine
    # predicts a 3D material volume and bakes it to UVs; it never overlays RGB.
    source = folder / 'front-cutout.png'
    if not source.is_file():
        source = folder / 'front.png'
    settings = settings_factory(images=[source], detail=TIERS[job['quality']], seed=12345)
    result = engine_factory().generate(settings, folder / 'trellis-native',
        progress=lambda stage, fraction: write_progress(folder, stage, fraction))
    shutil.copy2(result.glb_path, folder / 'model.glb')
    # Preserve the native PBR GLB byte-for-byte. Other exports are derived.
    import trimesh
    scene = trimesh.load(folder / 'model.glb', force='scene', process=False)
    shape = scene.to_mesh()
    shape.visual = trimesh.visual.ColorVisuals(mesh=shape)
    shape.export(folder / 'shape.glb')
    shape.export(folder / 'model.stl')
    from trimesh.exchange.obj import export_obj
    obj, companions = export_obj(scene, return_texture=True)
    (folder / 'model.obj').write_text(obj, encoding='utf-8')
    with zipfile.ZipFile(folder / 'model-obj.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(folder / 'model.obj', 'model.obj')
        for name, data in (companions or {}).items():
            name = Path(name).name
            (folder / name).write_bytes(data)
            bundle.writestr(name, data)
    if not job.get('color', True):
        shutil.copy2(folder / 'shape.glb', folder / 'model.glb')
    payload = {'vertices': result.vertices, 'faces': result.faces,
        'texture': bool(job.get('color', True)), 'texture_method': 'trellis2-native-pbr',
        'texture_size': 2048 if result.detail == 'draft' else 4096,
        'native_pbr': bool(job.get('color', True)), 'reference_preserved': False,
        'reference_symmetry': False, 'tier_requested': TIERS[job['quality']],
        'tier_used': result.detail, 'peak_vram_gb': result.peak_vram_gb,
        'generation_notes': result.notes}
    (folder / 'trellis-result.json').write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    return payload


if __name__ == '__main__':
    run(Path(sys.argv[1]))
