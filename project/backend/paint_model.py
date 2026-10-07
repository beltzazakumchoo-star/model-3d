"""Hunyuan3D-Paint with local weights and bounded-memory Windows inference."""
import gc
import json
import os
from pathlib import Path
from types import MethodType

from backend.single_shape import report


def prepare_texture_mesh(mesh, max_faces=100000):
    original_faces = len(mesh.faces)
    # Marching cubes can emit zero-area triangles. Remove them before quadric
    # decimation; otherwise they can collapse the entire mesh into a few faces.
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) > max_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) < min(1000, max_faces // 2, original_faces // 10):
        raise RuntimeError('การเตรียมผิวสูญเสียรูปทรงมากเกินไป กรุณาตรวจ shape.glb')
    return mesh, original_faces


def load_painter(model):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, UNet2DConditionModel
    from hy3dgen.texgen.hunyuanpaint.unet.modules import UNet2p5DConditionModel
    from hy3dgen.texgen.hunyuanpaint.pipeline import HunyuanPaintPipeline

    config = json.loads((model / 'unet/config.json').read_text())
    with init_empty_weights(include_buffers=False):
        unet = UNet2p5DConditionModel(UNet2DConditionModel(**config))
    weights = load_file(str(model / 'unet/diffusion_pytorch_model.safetensors'))
    unet.load_state_dict(weights, strict=True, assign=True)
    unet = unet.eval().to(dtype=torch.float16)
    del weights
    vae = AutoencoderKL.from_pretrained(str(model / 'vae'), torch_dtype=torch.float16, local_files_only=True)
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(str(model / 'scheduler'),
        timestep_spacing='trailing', local_files_only=True)
    # Paint uses checkpoint learned prompt embeddings, not text input.
    pipe = HunyuanPaintPipeline(vae=vae, text_encoder=None, tokenizer=None,
        unet=unet, scheduler=scheduler, feature_extractor=None)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    encode_prompt = pipe.encode_prompt
    def colocated_prompt(_self, *args, **kwargs):
        positive, negative = encode_prompt(*args, **kwargs)
        # Upstream disables CFG in encode_prompt, then concatenates the supplied
        # negative embedding anyway. CPU offload leaves that embedding on CPU.
        return positive, negative.to(device=positive.device, dtype=positive.dtype) if negative is not None else None
    pipe.encode_prompt = MethodType(colocated_prompt, pipe)
    original = unet.forward

    def sequential(_self, sample, timestep, encoder_hidden_states, *args, **kwargs):
        batch = sample.shape[0]
        if batch == 1:
            return original(sample, timestep, encoder_hidden_states, *args, **kwargs)
        def take(value, index):
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == batch:
                return value[index:index+1]
            if isinstance(value, dict):
                return {key: take(item, index) for key, item in value.items()}
            return value
        # Keep all six views together; split only classifier-free guidance.
        results = [original(sample[i:i+1], take(timestep, i), take(encoder_hidden_states, i),
            *args, **take(kwargs, i))[0] for i in range(batch)]
        return (torch.cat(results, dim=0),)

    unet.forward = MethodType(sequential, unet)
    pipe.enable_model_cpu_offload(device='cuda')
    pipe.set_progress_bar_config(disable=True)
    return pipe


def paint(folder, steps=24, texture_size=None, reuse_views=False):
    import numpy as np
    import torch
    import trimesh
    import xatlas
    from PIL import Image
    from backend.cpu_rasterizer import install
    install()
    from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
    from hy3dgen.texgen.pipelines import Hunyuan3DPaintPipeline

    reuse_views = reuse_views or all((folder / f'paint-view-{i}.png').is_file() for i in range(6))
    saved = json.loads((folder / 'job.json').read_text(encoding='utf-8')) if (folder / 'job.json').is_file() else {}
    texture_size = texture_size or (4096 if saved.get('quality') == 'detail' else 2048)
    root = Path(os.environ.get('FOURVIEW_AI_ROOT', 'D:/FourViewAI'))
    report(folder, 'กำลังเตรียม UV สำหรับลายผิว AI รอบตัว…')
    mesh = trimesh.load(folder / 'shape.glb', force='mesh')
    mouth_alignment = {'applied': False}
    if saved.get('reference_symmetry', False):
        from backend.head_alignment import align_side_mouth
        with Image.open(folder / 'front-cutout.png') as source:
            mouth_alignment = align_side_mouth(mesh, source)
        (folder / 'mouth-alignment.json').write_text(json.dumps(mouth_alignment), encoding='utf-8')
        if mouth_alignment.get('applied'):
            import shutil
            if not (folder / 'shape-before-mouth.glb').is_file():
                shutil.copy2(folder / 'shape.glb', folder / 'shape-before-mouth.glb')
            mesh.export(folder / 'shape.glb')
    mesh, original_faces = prepare_texture_mesh(mesh)
    # xatlas is native code: invalid input crashes the process (0xC0000005)
    # instead of raising, so reject it here with a readable error.
    if not np.isfinite(mesh.vertices).all():
        raise RuntimeError('shape.glb มีพิกัดไม่ถูกต้อง (NaN/inf) กรุณาสร้างรูปทรงใหม่')
    report(folder, f'กำลังคลี่ UV ({len(mesh.faces):,} สามเหลี่ยม · {texture_size}px)…')
    atlas = xatlas.Atlas()
    atlas.add_mesh(np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.uint32))
    charts, pack = xatlas.ChartOptions(), xatlas.PackOptions()
    charts.max_iterations = 1
    pack.resolution, pack.padding, pack.bilinear = texture_size, 4, True
    atlas.generate(chart_options=charts, pack_options=pack)
    mapping, faces, uv = atlas[0]
    if not len(faces) or not np.isfinite(uv).all() or faces.max() >= len(uv):
        raise RuntimeError('คลี่ UV ไม่สำเร็จ กรุณาลองระดับคุณภาพอื่น')
    mesh = trimesh.Trimesh(vertices=mesh.vertices[mapping], faces=faces, process=False,
        visual=trimesh.visual.TextureVisuals(uv=uv))
    del atlas
    # Raster and bake on CPU: no CUDA extension/compiler required on Windows.
    render = MeshRender(default_resolution=512, texture_size=texture_size, device='cpu')
    render.mesh_copy = mesh
    # Torch CPU tensors can alias NumPy arrays. Upstream transforms them in
    # place, so copy inputs to preserve the original mesh coordinates and UVs.
    render.set_mesh(np.asarray(mesh.vertices, dtype=np.float32).copy(),
        np.asarray(mesh.faces, dtype=np.int32).copy(),
        vtx_uv=np.asarray(mesh.visual.uv, dtype=np.float32).copy(),
        uv_idx=np.asarray(mesh.faces, dtype=np.int32).copy())
    elevs, azims = [0, 0, 0, 0, 90, -90], [0, 90, 180, 270, 0, 180]
    normals, positions = [], []
    for i, (elev, azim) in enumerate(zip(elevs, azims)):
        report(folder, f'กำลังเตรียมข้อมูลผิวมุมที่ {i+1}/6…')
        normal = render.render_normal(elev, azim, use_abs_coor=True, return_type='pl')
        position = render.render_position(elev, azim, return_type='pl')
        normal.save(folder / f'paint-normal-{i}.png')
        position.save(folder / f'paint-position-{i}.png')
        normals.append(normal)
        positions.append(position)
    camera_info = [(((azim//30)+9)%12)//{-90:3, 0:1, 90:3}[elev]
        + {-90:36, 0:12, 90:40}[elev] for elev, azim in zip(elevs, azims)]
    with Image.open(folder / 'front-cutout.png') as source:
        prompt = Hunyuan3DPaintPipeline.recenter_image(None, source.convert('RGBA')).resize((512, 512))
    if reuse_views:
        views = [Image.open(folder / f'paint-view-{i}.png').convert('RGB') for i in range(6)]
    else:
        report(folder, 'กำลังโหลด Hunyuan3D-Paint…')
        pipe = load_painter(root / 'models/Hunyuan3D-2/hunyuan3d-paint-v2-0')
        def progress(_pipe, step, _time, values):
            report(folder, f'AI สร้างลายผิวรอบตัว · {step+1}/{steps}', step+1, steps)
            return values
        with torch.inference_mode():
            views = pipe([prompt], num_inference_steps=steps, width=512, height=512,
                num_in_batch=6, camera_info_gen=[camera_info], camera_info_ref=[[0]],
                normal_imgs=[normals], position_imgs=[positions],
                generator=torch.Generator(device='cpu').manual_seed(12345),
                callback_on_step_end=progress).images
        if len(views) != 6:
            raise RuntimeError('Paint did not return six views')
        for i, view in enumerate(views):
            view.save(folder / f'paint-view-{i}.png')
        pipe.maybe_free_model_hooks()
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    report(folder, 'กำลังอบลายผิว AI ลง UV texture…')
    from backend.dense_paint import texture_from_views
    with Image.open(folder / 'front-cutout.png') as source:
        texture = texture_from_views(mesh, render, views, source.convert('RGBA'), texture_size, folder, symmetry=saved.get('reference_symmetry', False))
    mesh.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv.copy(),
        material=trimesh.visual.material.PBRMaterial(baseColorTexture=texture,
            baseColorFactor=[255,255,255,255], roughnessFactor=1., metallicFactor=0.))
    result = mesh
    from backend.reference_texture import export_model
    export_model(result, folder)
    metadata = {'texture': True, 'vertex_color': False, 'texture_method': 'hunyuan3d-paint-v2-0',
        'texture_size': texture_size, 'paint_views': 6, 'paint_steps': steps,
        'texture_bake': 'dense-reference-v2', 'reference_preserved': True,
        'reference_symmetry': saved.get('reference_symmetry', False),
        'mouth_alignment': mouth_alignment,
        'original_faces': original_faces, 'texture_faces': len(result.faces),
        'faces': len(result.faces), 'vertices': len(result.vertices),
        'simplified': len(result.faces) < original_faces}
    (folder / 'paint-result.json').write_text(json.dumps(metadata), encoding='utf-8')
    report(folder, 'สร้างลายผิว AI และส่งออกสำเร็จ')


if __name__ == '__main__':
    import argparse
    import faulthandler
    # Native crashes (numba/xatlas/CUDA) exit without a traceback; dump the
    # Python stack into paint_model.log so the failing step is visible.
    faulthandler.enable()
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--steps', type=int, default=24)
    parser.add_argument('--reuse-views', action='store_true')
    args = parser.parse_args()
    paint(args.folder, args.steps, reuse_views=args.reuse_views)






