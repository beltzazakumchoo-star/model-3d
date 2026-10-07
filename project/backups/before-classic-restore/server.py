"""Loopback-only image-to-mesh service. No paid API or remote inference calls."""
import importlib.util
import gc
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import tempfile
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from starlette.background import BackgroundTask

PROJECT = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get('FOURVIEW_AI_ROOT', 'D:/FourViewAI')).resolve()
MODEL = ROOT / 'models' / 'Hunyuan3D-2mv'
OUTPUTS = ROOT / 'outputs'
WEIGHTS = MODEL / 'hunyuan3d-dit-v2-mv' / 'model.fp16.safetensors'
os.environ.setdefault('HF_HOME', str(ROOT / 'cache' / 'huggingface'))
os.environ.setdefault('U2NET_HOME', str(ROOT / 'cache' / 'rembg'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

app = FastAPI(title='FourView Local AI', docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
    allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])
jobs = {}
lock = threading.Lock()
worker = ThreadPoolExecutor(max_workers=1)
pipeline = None
background_session = None
VIEWS = ('front', 'right', 'back', 'left')
QUALITIES = {'draft': (256, 25), 'balanced': (384, 40), 'detail': (512, 50)}


def readiness():
    missing = [p for p in ('torch', 'hy3dgen', 'rembg') if importlib.util.find_spec(p) is None]
    if missing:
        return {'ready': False, 'reason': 'ต้องติดตั้ง ' + ', '.join(missing)}
    if not WEIGHTS.is_file():
        progress = WEIGHTS.with_suffix('.download.json')
        if progress.is_file():
            try:
                from backend.checkpoint_download import SIZE, CHUNK
                state = json.loads(progress.read_text(encoding='utf-8'))
                done, total = len(state.get('completed', [])), (SIZE + CHUNK - 1) // CHUNK
                return {'ready': False, 'reason': f'กำลังดาวน์โหลด AI · {done}/{total} ชิ้น ({done*100//total}%)'}
            except (OSError, ValueError):
                pass
        return {'ready': False, 'reason': 'ยังไม่ได้ดาวน์โหลดโมเดล Hunyuan3D-2mv'}
    if not WEIGHTS.with_suffix('.verified.json').is_file():
        return {'ready': False, 'reason': 'กำลังตรวจสอบ checksum ของโมเดล'}
    if not (ROOT / 'cache' / 'rembg' / 'u2net.onnx').is_file():
        return {'ready': False, 'reason': 'กำลังติดตั้งตัวแยกพื้นหลัง'}
    import torch
    if not torch.cuda.is_available():
        return {'ready': False, 'reason': 'ยังไม่พบ CUDA GPU ที่พร้อมใช้งาน'}
    gpu = torch.cuda.get_device_properties(0)
    return {'ready': True, 'gpu': gpu.name, 'vram_gb': round(gpu.total_memory / 2**30, 1)}


@app.middleware('http')
async def local_origin_only(request, call_next):
    if request.method == 'POST':
        origin = request.headers.get('origin')
        allowed = {f'http://{host}:{port}' for host in ('localhost', '127.0.0.1')
                   for port in (5173, 8008)}
        if origin and origin not in allowed:
            from fastapi.responses import JSONResponse
            return JSONResponse({'detail': 'Use the local FourView app.'}, status_code=403)
    return await call_next(request)


@app.get('/api/health')
def health():
    return {'engine': 'Hunyuan3D-2mv', 'root': str(ROOT),
        'view_generator_ready': (ROOT / 'models/Wonder3D/PINNED_REVISION.txt').is_file(), **readiness()}


@app.get('/api/latest')
def latest_model():
    completed = []
    if OUTPUTS.exists():
        for file in OUTPUTS.glob('*/job.json'):
            try:
                data = json.loads(file.read_text(encoding='utf-8'))
                if data.get('status') == 'completed':
                    completed.append(data)
            except (OSError, ValueError):
                continue
    return max(completed, key=lambda item: item.get('created', 0), default=None)


def update(job_id, **values):
    with lock:
        jobs[job_id].update(values)
        snapshot = dict(jobs[job_id])
    # Keep completed and failed job metadata alongside the exported mesh.
    (OUTPUTS / job_id / 'job.json').write_text(json.dumps(snapshot, ensure_ascii=False), encoding='utf-8')


def prepare_images(folder, views=VIEWS):
    global background_session
    import numpy as np
    from rembg import new_session, remove
    images = {}
    for view in views:
        with Image.open(folder / f'{view}.png') as source:
            image = ImageOps.exif_transpose(source).convert('RGBA')
        alpha = np.array(image.getchannel('A'))
        if not (alpha < 250).any():
            if background_session is None:
                background_session = new_session('u2net', providers=['CPUExecutionProvider'])
            image = remove(image, session=background_session).convert('RGBA')
        alpha = np.asarray(image.getchannel('A'))
        if (alpha > 64).sum() < 100:
            raise ValueError(f'ไม่พบวัตถุในภาพ {view} หลังแยกพื้นหลัง')
        image.save(folder / f'{view}-cutout.png')
        images[view] = image
    return images


def project_vertex_colors(mesh, images):
    """Approximate color from four orthographic views; not UV or generated texture."""
    import numpy as np
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    bounds = mesh.bounds
    extent = np.maximum(bounds[1] - bounds[0], 1e-6)
    normalized = (vertices - bounds[0]) / extent
    accumulated = np.zeros((len(vertices), 3), dtype=np.float64)
    total = np.zeros(len(vertices), dtype=np.float64)
    mappings = {'front': (normalized[:, 0], normals[:, 2]),
                'back': (1-normalized[:, 0], -normals[:, 2]),
                'right': (normalized[:, 2], -normals[:, 0]),
                'left': (1-normalized[:, 2], normals[:, 0])}
    for view, (u, direction) in mappings.items():
        rgba = np.asarray(images[view])
        mask = rgba[:, :, 3] > 64
        yy, xx = np.nonzero(mask)
        px = np.clip(np.rint(xx.min() + u * (xx.max()-xx.min())).astype(int), 0, rgba.shape[1]-1)
        py = np.clip(np.rint(yy.max() - normalized[:, 1] * (yy.max()-yy.min())).astype(int), 0, rgba.shape[0]-1)
        sampled = rgba[py, px]
        weight = np.maximum(direction, 0)**3 * (sampled[:, 3] / 255)
        accumulated += sampled[:, :3] * weight[:, None]
        total += weight
    colors = np.full((len(vertices), 4), 255, dtype=np.uint8)
    colors[:, :3] = [170, 180, 210]
    valid = total > 1e-6
    colors[valid, :3] = np.clip(accumulated[valid] / total[valid, None], 0, 255).astype(np.uint8)
    mesh.visual.vertex_colors = colors


def get_pipeline(job_id):
    global pipeline
    if pipeline is None:
        update(job_id, stage='กำลังโหลดโมเดล AI และแบ่งใช้ RAM/GPU…')
        import torch
        from backend.model_loader import load_multiview
        pipeline = load_multiview(MODEL)
    return pipeline


def generate_job(job_id, quality, color):
    global pipeline, background_session
    folder = OUTPUTS / job_id
    started = jobs[job_id].get('created', time.time())
    try:
        import torch
        if jobs[job_id].get('input_mode') == 'front' and not all((folder / f'{view}.png').is_file() for view in VIEWS):
            update(job_id, status='running', stage='กำลังเตรียมภาพด้านหน้าเพื่อเจนอีก 3 มุม…')
            prepare_images(folder, ('front',))
            background_session = None
            if pipeline is not None:
                pipeline.maybe_free_model_hooks()
                pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
            update(job_id, stage='Wonder3D กำลังเจนภาพซ้าย ขวา และหลังในเครื่อง…')
            with (folder / 'generate-views.log').open('w', encoding='utf-8') as log:
                result = subprocess.run([sys.executable, '-u', '-m', 'backend.generate_views', str(folder)],
                    cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, timeout=900,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            if result.returncode:
                raise RuntimeError('เจนมุมภาพไม่สำเร็จ ดู generate-views.log ในโฟลเดอร์ผลงาน')
            update(job_id, generated_views=True, stage='สร้างภาพอีก 3 มุมแล้ว กำลังสร้างโมเดล 3D…')
        update(job_id, status='running', stage='กำลังแยกพื้นหลังจากภาพ 4 มุม…')
        if all((folder / f'{view}-cutout.png').is_file() for view in VIEWS):
            images = {view: Image.open(folder / f'{view}-cutout.png').convert('RGBA') for view in VIEWS}
        else:
            images = prepare_images(folder)
        background_session = None
        gc.collect()
        resolution, steps = QUALITIES[quality]

        def progress(step, _timestep, _output):
            stage = f'สร้างรูปทรง AI · ขั้นที่ {step+1}/{steps}'
            if step+1 == steps:
                stage = 'กำลังแปลงรูปทรง AI เป็นผิวสามเหลี่ยม…'
            update(job_id, step=step+1, stage=stage)

        if not (folder / 'shape-latents.pt').is_file():
            engine = get_pipeline(job_id)
            update(job_id, stage='กำลังสร้างรูปทรงจากภาพ 4 มุม…', steps=steps, step=0)
            latents = engine(image=images, num_inference_steps=steps,
                          generator=torch.Generator(device='cpu').manual_seed(12345),
                          output_type='latent', callback=progress, callback_steps=1)
            torch.save(latents.detach().cpu(), folder / 'shape-latents.pt')
            for hook in engine._all_hooks:
                hook.offload()
                hook.remove()
            # Offload hooks link to previous modules. Release the loop variable
            # and hook chain before starting the separate decoder process.
            hook = None
            engine._all_hooks.clear()
            engine.components.clear()
            engine.model = engine.conditioner = engine.vae = None
            pipeline = None
            del engine, latents
            gc.collect()
            torch.cuda.empty_cache()
        update(job_id, stage='กำลังถอดผิวสามเหลี่ยมจากรูปทรง AI…')
        if not (folder / 'shape.glb').is_file():
            with (folder / 'decode.log').open('w', encoding='utf-8') as log:
                result = subprocess.run([sys.executable, '-u', '-m', 'backend.decode_saved',
                    str(folder), '--resolution', str(resolution)], cwd=PROJECT,
                    stdout=log, stderr=subprocess.STDOUT, timeout=900,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            if result.returncode:
                raise RuntimeError(f'ถอดผิวโมเดลไม่สำเร็จ (exit {result.returncode}) ดู decode.log ในโฟลเดอร์ผลงาน')
        import trimesh
        mesh = trimesh.load(folder / 'shape.glb', force='mesh')
        if mesh is None or not len(mesh.faces):
            raise ValueError('AI ไม่สามารถสร้างผิวโมเดลจากภาพชุดนี้ได้')
        update(job_id, stage='กำลังเก็บผิวโมเดลและฉายสีจากภาพ…')
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        if color:
            from backend.reference_texture import texture_from_references
            mesh = texture_from_references(mesh, images, folder,
                lambda stage: update(job_id, stage=stage))
        else:
            mesh.visual.vertex_colors = [165, 181, 235, 255]
        from backend.reference_texture import export_model
        export_model(mesh, folder)
        texture_info = json.loads((folder / 'texture-mesh.json').read_text()) if (folder / 'texture-mesh.json').is_file() else {}
        update(job_id, status='completed', stage='สร้างโมเดลสำเร็จ',
               vertices=len(mesh.vertices), faces=len(mesh.faces),
               seconds=round(time.time()-started), color=color, texture=bool(color),
               preview=f'/api/jobs/{job_id}/artifacts/glb', **texture_info)
    except Exception as exc:
        logging.exception('Generation failed for %s', job_id)
        (folder / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        message = str(exc)
        if 'out of memory' in message.lower():
            message = 'หน่วยความจำ GPU ไม่พอ ลองปิดโปรแกรมที่ใช้ GPU แล้วเลือกระดับร่าง 256'
        update(job_id, status='failed', stage='สร้างโมเดลไม่สำเร็จ', error=message)
    finally:
        try:
            import torch
            if pipeline is not None:
                pipeline.maybe_free_model_hooks()
                pipeline.device = torch.device('cuda:0')
            torch.cuda.empty_cache()
        except Exception:
            logging.exception('GPU cleanup failed')


@app.post('/api/jobs', status_code=202)
async def submit(front: UploadFile = File(...), right: UploadFile | None = File(None),
                 back: UploadFile | None = File(None), left: UploadFile | None = File(None),
                 quality: str = Form('draft'), color: bool = Form(True), input_mode: str = Form('multiview')):
    if input_mode not in ('multiview', 'front', 'sheet'):
        raise HTTPException(422, 'Unknown input mode')
    if input_mode == 'multiview' and any(upload is None for upload in (right, back, left)):
        raise HTTPException(422, 'กรุณาเพิ่มภาพให้ครบทั้ง 4 มุม')
    if input_mode == 'front' and not (ROOT / 'models/Wonder3D/PINNED_REVISION.txt').is_file():
        raise HTTPException(503, 'กำลังติดตั้งตัวสร้างมุมภาพ Wonder3D')
    if quality not in QUALITIES:
        raise HTTPException(422, 'Unknown quality preset')
    if not readiness()['ready']:
        raise HTTPException(503, readiness()['reason'])
    # Validate before creating a job or occupying the GPU queue.
    decoded = {}
    for view, upload in zip(VIEWS, (front, right, back, left)):
        if upload is None or (input_mode in ('front', 'sheet') and view != 'front'):
            continue
        payload = await upload.read(12 * 1024 * 1024 + 1)
        if len(payload) > 12 * 1024 * 1024:
            raise HTTPException(413, f'{view}: ภาพใหญ่กว่า 12 MB')
        try:
            with Image.open(io.BytesIO(payload)) as im:
                if im.width * im.height > 40_000_000:
                    raise ValueError('ภาพมีขนาดเกิน 40 ล้านพิกเซล')
                decoded[view] = ImageOps.exif_transpose(im).convert('RGBA')
        except Exception as exc:
            raise HTTPException(422, f'{view}: เปิดภาพไม่สำเร็จ ({exc})') from exc
    source_sheet = None
    if input_mode == 'sheet':
        from backend.split_sheet import split_sheet
        source_sheet = decoded['front']
        try:
            decoded = split_sheet(source_sheet)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    job_id = uuid.uuid4().hex
    with lock:
        if any(j['status'] in ('queued', 'running') for j in jobs.values()):
            raise HTTPException(409, 'มีงานกำลังใช้ GPU อยู่ กรุณารอให้เสร็จ')
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        folder = OUTPUTS / job_id
        folder.mkdir()
        if source_sheet is not None:
            source_sheet.save(folder / 'source.png')
        for view, image in decoded.items():
            image.save(folder / f'{view}.png')
        jobs[job_id] = {'id': job_id, 'status': 'queued', 'stage': 'รอประมวลผล',
                        'quality': quality, 'color': color, 'input_mode': input_mode, 'created': time.time()}
    worker.submit(generate_job, job_id, quality, color)
    return {'id': job_id}


@app.post('/api/jobs/{job_id}/resume', status_code=202)
def resume_job(job_id: str):
    saved = job_status(job_id)
    folder = OUTPUTS / job_id
    can_restart_views = saved.get('input_mode') in ('front', 'sheet') and (folder / 'front.png').is_file()
    if saved['status'] != 'failed' or not (can_restart_views or (folder / 'shape-latents.pt').is_file()):
        raise HTTPException(409, 'งานนี้ยังไม่มีข้อมูลที่กู้ต่อได้')
    with lock:
        if any(j['status'] in ('queued', 'running') for j in jobs.values()):
            raise HTTPException(409, 'กรุณารอให้งานปัจจุบันเสร็จ')
        saved.update(status='queued', stage='กู้รูปทรงที่บันทึกไว้')
        saved.pop('error', None)
        jobs[job_id] = saved
    worker.submit(generate_job, job_id, saved['quality'], saved.get('color', True))
    return {'id': job_id}


@app.get('/api/jobs/{job_id}')
def job_status(job_id: str):
    if not re.fullmatch(r'[a-f0-9]{32}', job_id):
        raise HTTPException(404, 'Job not found')
    with lock:
        if job_id in jobs:
            return dict(jobs[job_id])
    saved = OUTPUTS / job_id / 'job.json'
    if saved.is_file():
        result = json.loads(saved.read_text(encoding='utf-8'))
        if result['status'] in ('queued', 'running'):
            result.update(status='failed', error='ตัวรันถูกปิดระหว่างสร้างโมเดล กรุณาสร้างใหม่')
        return result
    raise HTTPException(404, 'Job not found')


@app.post('/api/jobs/{job_id}/refine-texture', status_code=202)
def refine_texture(job_id: str):
    saved = job_status(job_id)
    if saved['status'] != 'completed':
        raise HTTPException(409, 'ต้องสร้างรูปทรงให้เสร็จก่อน')
    source = OUTPUTS / job_id
    if not (source / 'shape.glb').is_file():
        raise HTTPException(409, 'ไม่พบรูปทรงต้นฉบับ')
    with lock:
        if any(j['status'] in ('queued', 'running') for j in jobs.values()):
            raise HTTPException(409, 'กรุณารอให้งานปัจจุบันเสร็จ')
        new_id = uuid.uuid4().hex
        folder = OUTPUTS / new_id
        folder.mkdir()
        for name in ['shape.glb', 'shape-latents.pt', 'source.png'] + [f'{view}{suffix}.png' for view in VIEWS for suffix in ('', '-cutout')]:
            if (source / name).is_file():
                shutil.copy2(source / name, folder / name)
        jobs[new_id] = {'id': new_id, 'status': 'queued', 'stage': 'เตรียมรายละเอียดผิวจากภาพต้นฉบับ',
            'quality': saved['quality'], 'color': True, 'created': time.time(), 'source_job': job_id,
            'input_mode': saved.get('input_mode', 'multiview'), 'generated_views': saved.get('generated_views', False)}
    worker.submit(generate_job, new_id, saved['quality'], True)
    return {'id': new_id}


@app.get('/api/jobs/{job_id}/artifacts/{format}')
def artifact(job_id: str, format: str, surface: str = 'texture'):
    if format not in ('obj', 'glb', 'stl', 'obj-zip') or job_status(job_id)['status'] != 'completed':
        raise HTTPException(404, 'Artifact not ready')
    if surface not in ('texture', 'shape'):
        raise HTTPException(400, 'Unknown surface')
    if surface == 'shape':
        # Match the gray, untextured viewport when exporting. Build the artifact
        # from the geometry so it works consistently for GLB, OBJ and STL.
        temp = tempfile.TemporaryDirectory(prefix='fourview-gray-')
        try:
            import trimesh
            folder = Path(temp.name)
            source = OUTPUTS / job_id / 'shape.glb'
            if not source.is_file():
                source = OUTPUTS / job_id / 'model.glb'
            mesh = trimesh.load(source, force='mesh')
            gray = [184, 193, 206, 255]
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh, vertex_colors=[gray] * len(mesh.vertices))
            artifact_format = 'zip' if format == 'obj-zip' else format
            generated = folder / ('model.obj' if format in ('obj', 'obj-zip') else f'model.{format}')
            mesh.export(generated, include_normals=True)
            if format == 'obj-zip':
                import zipfile
                bundle = folder / 'model-obj.zip'
                with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
                    archive.write(generated, generated.name)
                    for companion in folder.glob('*.mtl'):
                        archive.write(companion, companion.name)
                generated = bundle
            return FileResponse(generated, filename=f'fourview-{job_id[:8]}.{artifact_format}',
                                media_type='model/gltf-binary' if format == 'glb' else 'application/octet-stream',
                                background=BackgroundTask(temp.cleanup))
        except Exception:
            temp.cleanup()
            raise
    file = OUTPUTS / job_id / ('model-obj.zip' if format == 'obj-zip' else f'model.{format}')
    if not file.is_file():
        raise HTTPException(404, 'Artifact not found')
    return FileResponse(file, filename=f'fourview-{job_id[:8]}.{"zip" if format == "obj-zip" else format}',
                        media_type='model/gltf-binary' if format == 'glb' else 'application/octet-stream')


@app.get('/api/jobs/{job_id}/images/{view}')
def reference_image(job_id: str, view: str):
    job_status(job_id)
    if view not in (*VIEWS, 'source'):
        raise HTTPException(404, 'View not found')
    file = OUTPUTS / job_id / f'{view}.png'
    if not file.is_file():
        raise HTTPException(404, 'Reference image not found')
    return FileResponse(file, media_type='image/png')


if (PROJECT / 'dist').is_dir():
    app.mount('/', StaticFiles(directory=PROJECT / 'dist', html=True), name='frontend')
