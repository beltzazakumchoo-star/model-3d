"""Apply the paint_model crash fixes to an existing backend folder in place.

Each file is patched all-or-nothing: if any anchor is missing (the local
copy differs), that file is left untouched and reported. Re-running is safe.
"""
import sys
from pathlib import Path

PATCHES = {
    'cpu_rasterizer.py': [(
        """        x2, y2, z2 = screen[c]
        denominator""",
        """        x2, y2, z2 = screen[c]
        # Skip vertices at/behind the camera plane (w <= 0 gives inf/NaN).
        if not (pos[a, 3] > 0 and pos[b, 3] > 0 and pos[c, 3] > 0
                and np.isfinite(x0+y0+z0+x1+y1+z1+x2+y2+z2)):
            continue
        denominator""")],
    'dense_paint.py': [(
        """                    if px < 0 or py < 0 or px >= edge or py >= edge or f <= .05:
                        continue""",
        """                    # Written as a positive test so NaN fails it: int(NaN) is
                    # an arbitrary index and njit does not bounds-check.
                    if not (px >= 0 and py >= 0 and px < edge and py < edge and f > .05):
                        continue"""), (
        """    texture,valid,reference = bake_dense(np.asarray(mesh.visual.uv),np.asarray(mesh.faces),
        np.asarray(projections),np.asarray(facing),np.asarray(images),np.asarray(depths),np.asarray(tolerances),size,""",
        """    uv = np.nan_to_num(np.asarray(mesh.visual.uv,dtype=np.float64),nan=0.,posinf=0.,neginf=0.)
    faces = np.ascontiguousarray(mesh.faces,dtype=np.int64)
    if len(faces) and (faces.min()<0 or faces.max()>=len(uv)):
        raise RuntimeError('ดัชนีผิวโมเดลเกินจำนวน UV ระหว่างอบ texture')
    texture,valid,reference = bake_dense(uv,faces,
        np.nan_to_num(np.asarray(projections),nan=-1.,posinf=-1.,neginf=-1.),
        np.nan_to_num(np.asarray(facing)),np.ascontiguousarray(images),np.asarray(depths),np.asarray(tolerances),size,""")],
    'paint_model.py': [(
        "def paint(folder, steps=24, texture_size=None, reuse_views=False):",
        "def paint(folder, steps=24, texture_size=None, reuse_views=False, max_faces=100000):"), (
        """    mesh, original_faces = prepare_texture_mesh(mesh)
    atlas = xatlas.Atlas()""",
        """    mesh, original_faces = prepare_texture_mesh(mesh, max_faces)
    # xatlas is native code: invalid input crashes the process (0xC0000005)
    # instead of raising, so reject it here with a readable error.
    if not np.isfinite(mesh.vertices).all():
        raise RuntimeError('shape.glb มีพิกัดไม่ถูกต้อง (NaN/inf) กรุณาสร้างรูปทรงใหม่')
    report(folder, f'กำลังคลี่ UV ({len(mesh.faces):,} สามเหลี่ยม · {texture_size}px)…')
    atlas = xatlas.Atlas()"""), (
        """    mapping, faces, uv = atlas[0]
""",
        """    mapping, faces, uv = atlas[0]
    if not len(faces) or not np.isfinite(uv).all() or faces.max() >= len(uv):
        raise RuntimeError('คลี่ UV ไม่สำเร็จ กรุณาลองระดับคุณภาพอื่น')
"""), (
        """    import argparse
    parser = argparse.ArgumentParser()""",
        """    import argparse
    import faulthandler
    # Native crashes (numba/xatlas/CUDA) exit without a traceback; dump the
    # Python stack into paint_model.log so the failing step is visible.
    faulthandler.enable()
    parser = argparse.ArgumentParser()"""), (
        """    parser.add_argument('--reuse-views', action='store_true')
""",
        """    parser.add_argument('--reuse-views', action='store_true')
    parser.add_argument('--texture-size', type=int, choices=(1024, 2048, 4096))
    parser.add_argument('--max-faces', type=int, default=100000)
"""), (
        "    paint(args.folder, args.steps, reuse_views=args.reuse_views)",
        "    paint(args.folder, args.steps, args.texture_size, args.reuse_views, args.max_faces)")],
    'server.py': [(
        """def run_worker(job_id, module, arguments=(), timeout=3600):""",
        """class WorkerCrash(RuntimeError):
    \"\"\"Native code (numba, xatlas, CUDA) crashed without a Python exception.\"\"\"


def crash_frame(log_path):
    # Worker processes run with faulthandler; its first frame names the line.
    try:
        text = log_path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return ''
    match = re.search(r'most recent call first\\):\\s*File "([^"]+)", line (\\d+) in (\\S+)', text)
    if not match:
        return ''
    filename = re.split(r'[\\\\/]', match[1])[-1]
    return f'{filename}:{match[2]} {match[3]}'


def worker_failure(name, code, progress_file, log_path):
    stage = ''
    try:
        stage = json.loads(progress_file.read_text(encoding='utf-8')).get('stage', '')
    except (OSError, ValueError):
        pass
    crashed = code & 0xFFFFFFFF == 0xC0000005
    frame = crash_frame(log_path) if crashed else ''
    detail = ' · โปรแกรมหยุดทำงานกะทันหัน (access violation)' if crashed else ''
    during = f' ระหว่าง "{stage}"' if stage else ''
    where = f' [{frame}]' if frame else ''
    message = f'{name} ไม่สำเร็จ (exit {code}){detail}{during}{where} ดู {name}.log ในโฟลเดอร์ผลงาน'
    return WorkerCrash(message) if crashed else RuntimeError(message)


def run_worker(job_id, module, arguments=(), timeout=3600):"""), (
        """    with (folder / (module.rsplit('.', 1)[-1] + '.log')).open('w', encoding='utf-8') as log:""",
        """    log_path = folder / (module.rsplit('.', 1)[-1] + '.log')
    with log_path.open('w', encoding='utf-8') as log:"""), (
        """            env={**os.environ, 'PYTHONUTF8': '1'},""",
        """            env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONFAULTHANDLER': '1'},"""), (
        """                raise RuntimeError(f'{module.rsplit(".", 1)[-1]} ไม่สำเร็จ (exit {process.returncode}) ดู log ในโฟลเดอร์ผลงาน')""",
        """                raise worker_failure(module.rsplit('.', 1)[-1], process.returncode, progress_file, log_path)"""), (
        """            run_worker(job_id, 'backend.paint_model', ('--steps', str(paint_steps)))
""",
        """            try:
                run_worker(job_id, 'backend.paint_model', ('--steps', str(paint_steps)))
            except WorkerCrash:
                # Retry once with a 2K atlas and a lighter mesh; saved
                # paint-view images are reused, so diffusion is not repeated.
                if (folder / 'paint_model.log').is_file():
                    (folder / 'paint_model.log').replace(folder / 'paint_model-crash.log')
                update(job_id, stage='ลองอบลายผิวใหม่แบบประหยัดหน่วยความจำ · Texture 2K…', step=0, steps=0)
                run_worker(job_id, 'backend.paint_model', ('--steps', str(paint_steps),
                    '--texture-size', '2048', '--max-faces', '60000'))
""")],
}


def patch(backend):
    results = {}
    for name, edits in PATCHES.items():
        path = backend / name
        if name == 'server.py' and results.get('paint_model.py') not in ('patched', 'already patched'):
            # server.py passes --texture-size/--max-faces to the new CLI only.
            results[name] = 'skipped (paint_model.py not patched)'
            continue
        if not path.is_file():
            results[name] = 'missing'
            continue
        text = path.read_text(encoding='utf-8')
        if all(new in text for _, new in edits):
            results[name] = 'already patched'
            continue
        if not all(text.count(old) == 1 or new in text for old, new in edits):
            results[name] = 'skipped (local file differs)'
            continue
        for old, new in edits:
            if new not in text:
                text = text.replace(old, new)
        compile(text, str(path), 'exec')
        path.write_text(text, encoding='utf-8')
        results[name] = 'patched'
    return results


if __name__ == '__main__':
    for folder in sys.argv[1:]:
        for name, state in patch(Path(folder)).items():
            print(f'{folder}\\{name}: {state}')
