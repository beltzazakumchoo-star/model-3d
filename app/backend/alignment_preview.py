"""Render the actual mesh and reference in the color projection camera."""
from functools import lru_cache
import io
import json
from pathlib import Path
import threading
import numpy as np
from numba import njit
from PIL import Image
from scipy.ndimage import binary_erosion, binary_dilation
import trimesh
from backend.aligned_vertex import camera, VIEWS

render_lock = threading.Lock()


@lru_cache(maxsize=1)
def geometry(folder):
    mesh = trimesh.load(Path(folder)/'shape.glb', force='mesh')
    return (np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32),
            np.asarray(mesh.vertex_normals, dtype=np.float32))


@njit(nogil=True)
def raster(points, faces, normals, toward, size):
    depth = np.full((size, size), -1e10, dtype=np.float32)
    gray = np.zeros((size, size), dtype=np.uint8)
    for face in faces:
        a, b, c = points[face[0]], points[face[1]], points[face[2]]
        den = (b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den) < 1e-10:
            continue
        for y in range(max(0, int(np.floor(min(a[1], b[1], c[1])))), min(size-1, int(np.ceil(max(a[1], b[1], c[1]))))+1):
            for x in range(max(0, int(np.floor(min(a[0], b[0], c[0])))), min(size-1, int(np.ceil(max(a[0], b[0], c[0]))))+1):
                wa = ((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/den
                wb = ((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/den
                wc = 1-wa-wb
                if min(wa, wb, wc) < -0.001:
                    continue
                z = wa*a[2]+wb*b[2]+wc*c[2]
                if z > depth[y, x]:
                    depth[y, x] = z
                    n = wa*normals[face[0]]+wb*normals[face[1]]+wc*normals[face[2]]
                    light = abs((n*toward).sum())/max(np.sqrt((n*n).sum()), 1e-8)
                    gray[y, x] = int(95+135*min(light, 1))
    return gray, depth > -1e9


def preview(folder, view, options, opacity):
    with render_lock:
        vertices, faces, normals = geometry(str(folder))
        reference = np.asarray(Image.open(folder/f'{view}-cutout.png').convert('RGBA').resize((512, 512), Image.Resampling.LANCZOS))
        mask = reference[:, :, 3] > 128
        yy, xx = np.nonzero(mask)
        if not len(xx):
            raise ValueError('ไม่พบตัวละครในภาพอ้างอิง')
        base_yaw = [0, -90, 180, 90][VIEWS.index(view)]
        pitch = 0
        calibration = folder/'color-alignment.json'
        if options['auto'] and calibration.is_file():
            saved = json.loads(calibration.read_text())[view]
            base_yaw = saved['yaw']-saved.get('manual', {}).get('yaw', 0)
            pitch = saved['pitch']
        settings = options['views'].get(view, {})
        points, toward = camera(vertices, base_yaw+settings.get('yaw', 0), pitch,
            (xx.min(), yy.min(), xx.max(), yy.max()), settings)
        gray, model_mask = raster(points.astype(np.float32), faces, normals, toward.astype(np.float32), 512)
        output = np.full((512, 512, 3), [243, 245, 248], dtype=np.float32)
        output[model_mask] = gray[model_mask, None]
        alpha = reference[:, :, 3:4]/255*opacity
        output = output*(1-alpha)+reference[:, :, :3]*alpha
        model_edge = binary_dilation(model_mask ^ binary_erosion(model_mask))
        image_edge = binary_dilation(mask ^ binary_erosion(mask))
        output[model_edge] = [0, 172, 209]
        if opacity > 0:
            output[image_edge] = [235, 70, 144]
            output[model_edge & image_edge] = [106, 72, 225]
        result = io.BytesIO()
        Image.fromarray(output.astype(np.uint8)).save(result, format='PNG')
        return result.getvalue(), base_yaw, pitch, bool(options['auto'] and calibration.is_file())
