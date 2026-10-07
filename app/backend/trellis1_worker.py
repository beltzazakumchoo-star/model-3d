"""Staged single-image TRELLIS trial using cached models, without network calls."""
import os
import sys
from pathlib import Path
import json
import time
import gc
import types
import runpy
import traceback

ROOT = Path(os.environ.get("FOURVIEW_AI_ROOT", "D:/FourViewAI"))
for path in reversed([ROOT / "runtime/trellis1/deps", ROOT / "runtime/trellis1/site-packages", ROOT / "vendor/trellis1/bundle/code"]):
    sys.path.insert(0, str(path))
os.environ.update(ATTN_BACKEND="xformers", SPCONV_ALGO="native", XFORMERS_FORCE_DISABLE_TRITON="1",
                  HF_HUB_OFFLINE="1", HF_HOME=str(ROOT / "trellis1/cache/huggingface"), TORCH_HOME=str(ROOT / "trellis1/cache/torch"))
import torch
from PIL import Image
from safetensors.torch import load_file

torch.set_num_threads(4)
from backend.trellis1_compat import install
install()
from trellis.pipelines import TrellisImageTo3DPipeline, samplers
from trellis import models
from torchvision import transforms

SNAPSHOT = ROOT / "trellis1/cache/huggingface/hub/models--jetx--TRELLIS-image-large/snapshots/ab6a010207b4ceab566c5d8927d90a6f69c5f976"
OUT = Path(sys.argv[1]).resolve()
job = json.loads((OUT / "job.json").read_text(encoding="utf-8"))
IMAGE = OUT / "front-cutout.png"
steps = {"draft": 25, "balanced": 35, "detail": 50}[job["quality"]]
texture_size = 4096 if job["quality"] == "detail" else 2048
start = time.monotonic()
status = {"engine": "trellis1", "seed": 42, "source": str(IMAGE), "compatibility": "cutlass + torch-submanifold-convolution", "status": "running"}

def event(stage, **extra):
    status.update(stage=stage, seconds=round(time.monotonic() - start, 1), **extra)
    (OUT / "trellis1-progress.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    progress = OUT / 'worker-progress.tmp'
    labels = {'inference': 'เริ่มสร้างโมเดลจากภาพ',
              'geometry complete': 'สร้างรูปทรงแล้ว กำลังเตรียมลายผิว',
              'baking texture': 'กำลังสร้าง texture ' + ('4K' if texture_size == 4096 else '2K'),
              'complete': 'สร้างโมเดลสำเร็จ', 'failed': 'สร้างโมเดลไม่สำเร็จ'}
    if stage.startswith('loading ') or stage.startswith('loaded '):
        name = stage.split(' ', 1)[1]
        label = ('กำลังวิเคราะห์ภาพ' if name == 'image_cond_model' else
                 'กำลังสร้างรูปทรง 3D' if name.startswith('sparse_structure') else
                 'กำลังสร้างรายละเอียดรูปทรงและสีรอบตัว')
    else:
        label = labels.get(stage, 'กำลังประมวลผลโมเดล')
    progress.write_text(json.dumps({'stage': 'TRELLIS · ' + label}, ensure_ascii=False), encoding='utf-8')
    os.replace(progress, OUT / 'worker-progress.json')
    print("TRELLIS", stage, status["seconds"], extra, flush=True)

def load_component(name):
    event("loading " + name)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        if name == "image_cond_model":
            module = torch.hub.load(str(ROOT / "trellis1/cache/torch/hub/facebookresearch_dinov2_main"),
                                    "dinov2_vitl14_reg", source="local", pretrained=False)
            weights = torch.load(ROOT / "trellis1/cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth",
                                 map_location="cpu", mmap=True, weights_only=True)
        else:
            base = SNAPSHOT / config["models"][name]
            spec = json.loads(base.with_suffix(".json").read_text())
            module = getattr(models, spec["name"])(**spec["args"])
            weights = load_file(str(base.with_suffix(".safetensors")))
        module.load_state_dict(weights, strict=True)
        del weights
        module.half().eval()
    finally:
        torch.set_default_dtype(old_dtype)
    gc.collect()
    module.cuda()
    event("loaded " + name, allocated_gb=round(torch.cuda.memory_allocated() / 2**30, 2))
    return module

config = json.loads((SNAPSHOT / "pipeline.json").read_text())["args"]
pipeline = TrellisImageTo3DPipeline()
pipeline.models = {}
for sampler_name in ("sparse_structure_sampler", "slat_sampler"):
    spec = config[sampler_name]
    setattr(pipeline, sampler_name, getattr(samplers, spec["name"])(**spec["args"]))
    setattr(pipeline, sampler_name + "_params", spec["params"])
pipeline.slat_normalization = config["slat_normalization"]
pipeline.image_cond_model_transform = transforms.Compose([transforms.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225])])
pipeline.rembg_session = None

def move_models(self, names, device, empty_cache):
    if device == "cuda":
        for name in names:
            self.models[name] = load_component(name)
    else:
        for name in names:
            module = self.models.pop(name, None)
            del module
        gc.collect()
        torch.cuda.empty_cache()
        event("released " + ", ".join(names))

pipeline._move_models = types.MethodType(move_models, pipeline)
pipeline._move_all_models_to_cpu = types.MethodType(lambda self: None, pipeline)

try:
    view_names = ('front', 'right', 'back', 'left') if job.get('input_mode') == 'multiview' else ('front',)
    reference_images = [Image.open(OUT / f'{view}-cutout.png').convert('RGBA') for view in view_names]
    image = reference_images[0]
    if image.mode != "RGBA" or image.getextrema()[-1][0] == 255:
        raise ValueError("This offline trial requires an image with a real alpha mask")
    pipeline.preprocess_image(image).save(OUT / "conditioning.png")
    event("inference")
    with torch.inference_mode():
        params = dict(seed=42, formats=["mesh", "gaussian"],
            sparse_structure_sampler_params={"steps": steps}, slat_sampler_params={"steps": steps})
        outputs = (pipeline.run_multi_image(reference_images, mode='stochastic', **params)
                   if len(reference_images) > 1 else pipeline.run(image, **params))
    mesh = outputs["mesh"][0]
    if not mesh.success:
        raise ValueError("TRELLIS did not extract a valid mesh")
    import numpy as np
    import trimesh
    rgb = (mesh.vertex_attrs[:, :3].detach().float().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    asset = trimesh.Trimesh(vertices=mesh.vertices.detach().float().cpu().numpy(),
                           faces=mesh.faces.detach().cpu().numpy(), vertex_colors=rgb, process=False)
    # Match the textured export's z-up to y-up conversion for gray preview.
    original_vertices, original_faces = len(asset.vertices), len(asset.faces)
    asset.vertices = asset.vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    asset.visual = trimesh.visual.ColorVisuals(mesh=asset)
    asset.export(OUT / "shape.glb")
    outputs["gaussian"][0].save_ply(str(OUT / "gaussian.ply"))
    event("geometry complete", vertices=len(asset.vertices), faces=len(asset.faces), geometry_saved=True)
    if job.get('color', True):
        from trellis.utils import postprocessing_utils
        event("baking texture")
        textured = postprocessing_utils.to_glb(outputs["gaussian"][0], mesh,
            simplify=0.8, texture_size=texture_size, verbose=False)
    else:
        textured = asset
    textured.export(OUT / "model.glb")
    import zipfile
    from trimesh.exchange.obj import export_obj
    obj, companions = export_obj(textured, return_texture=True)
    (OUT / 'model.obj').write_text(obj, encoding='utf-8')
    with zipfile.ZipFile(OUT / 'model-obj.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(OUT / 'model.obj', 'model.obj')
        for name, data in (companions or {}).items():
            name = Path(name).name
            (OUT / name).write_bytes(data)
            bundle.writestr(name, data)
    asset.export(OUT / 'model.stl')
    result = {'vertices': len(textured.vertices), 'faces': len(textured.faces),
        'original_faces': original_faces, 'original_vertices': original_vertices,
        'texture_faces': len(textured.faces), 'simplified': bool(job.get('color', True)),
        'texture': bool(job.get('color', True)), 'texture_method': 'trellis1-native-gaussian',
        'texture_size': texture_size if job.get('color', True) else None,
        'reference_preserved': False, 'reference_symmetry': False,
        'reference_views': len(reference_images), 'sampling_steps': steps, 'peak_vram_gb': round(torch.cuda.max_memory_allocated()/2**30, 2),
        'generation_notes': 'AI predicts unseen geometry and color; details may differ from the reference.'}
    (OUT / 'trellis1-result.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
    event("complete", status="completed")
except Exception as exc:
    event("failed", status="failed", error=str(exc))
    traceback.print_exc()
    sys.exit(1)
