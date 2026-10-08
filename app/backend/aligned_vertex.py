"""Whole-camera calibration and visible-surface vertex colors; no UV or mesh edits."""
import json
import numpy as np
from PIL import Image
from scipy.ndimage import (binary_closing, binary_dilation, binary_erosion,
    distance_transform_edt, gaussian_filter, map_coordinates)
from backend.reference_texture import depth_buffer

VIEWS = ('front', 'right', 'back', 'left')


def parse_alignment(value):
    try:
        data = json.loads(value)
        if not isinstance(data, dict) or not isinstance(data.get('auto', True), bool):
            raise ValueError()
        result = {'auto': data.get('auto', True), 'views': {}}
        for view, settings in data.get('views', {}).items():
            if view not in VIEWS or not isinstance(settings, dict):
                raise ValueError()
            clean = {}
            for key, low, high, default in [('x', -20, 20, 0), ('y', -20, 20, 0),
                    ('scale', .5, 1.5, 1), ('yaw', -30, 30, 0)]:
                number = float(settings.get(key, default))
                if not np.isfinite(number) or not low <= number <= high:
                    raise ValueError()
                clean[key] = number
            result['views'][view] = clean
        return result
    except (ValueError, TypeError, AttributeError):
        raise ValueError('ค่าจัดแนวภาพไม่ถูกต้อง') from None


def camera(vertices, yaw, pitch, box, settings=None):
    yaw, pitch = np.deg2rad(yaw), np.deg2rad(pitch)
    right = np.array([np.cos(yaw), 0, -np.sin(yaw)])
    toward = np.array([np.sin(yaw)*np.cos(pitch), np.sin(pitch), np.cos(yaw)*np.cos(pitch)])
    up = np.cross(toward, right)
    h, v = vertices @ right, vertices @ up
    x0, y0, x1, y1 = box
    px = x0+(h-h.min())/max(np.ptp(h), 1e-8)*(x1-x0)
    py = y1-(v-v.min())/max(np.ptp(v), 1e-8)*(y1-y0)
    if settings:
        scale = settings.get('scale', 1)
        px = (px-(x0+x1)/2)/scale+(x0+x1)/2-settings.get('x', 0)/100*(x1-x0)
        py = (py-(y0+y1)/2)/scale+(y0+y1)/2-settings.get('y', 0)/100*(y1-y0)
    return np.column_stack([px, py, vertices @ toward]), toward


def project_view(mesh, image, view, base_yaw, options, extent):
    """Calibrate one camera; return its pixels, vertex pixel coords, samples and visibility weight."""
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    image = image.resize((768, 768), Image.Resampling.LANCZOS)
    rgba = np.asarray(image)
    mask = rgba[:, :, 3] > 128
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise ValueError(f'{view}: ไม่พบตัวละครในภาพ')
    box = (xx.min(), yy.min(), xx.max(), yy.max())
    best = (-1, base_yaw, 0)
    if options['auto']:
        small_mask = np.asarray(Image.fromarray(mask).resize((128, 128), Image.Resampling.NEAREST))
        # Point-cloud silhouette is only used to rank camera candidates.
        # Final occlusion uses all triangles in a true depth buffer.
        for yaw in (base_yaw-15, base_yaw, base_yaw+15):
            for pitch in (-10, -5, 0, 5, 10):
                points, _ = camera(vertices, yaw, pitch, tuple(np.asarray(box)*127/767))
                p = points[::max(1, len(vertices)//150000)]
                silhouette = np.zeros((128, 128), dtype=bool)
                pixels = np.clip(np.rint(p[:, :2]).astype(int), 0, 127)
                silhouette[pixels[:, 1], pixels[:, 0]] = True
                silhouette = binary_closing(binary_dilation(silhouette))
                iou = np.count_nonzero(silhouette & small_mask)/max(np.count_nonzero(silhouette | small_mask), 1)
                score = iou-abs(yaw-base_yaw)*.0005-abs(pitch)*.0005
                if score > best[0]:
                    best = (score, yaw, pitch)
    settings = options['views'].get(view, {})
    points, toward = camera(vertices, best[1]+settings.get('yaw', 0), best[2], box, settings)
    depth = depth_buffer(points, np.asarray(mesh.faces), 768, 768)
    px, py = points[:, 0], points[:, 1]
    inside = (px >= 0) & (px <= 767) & (py >= 0) & (py <= 767)
    coords = np.array([np.clip(py, 0, 767), np.clip(px, 0, 767)])
    sampled = np.column_stack([map_coordinates(rgba[:, :, c].astype(float), coords, order=1, mode='nearest') for c in range(4)])
    front_depth = depth[np.rint(coords[0]).astype(int), np.rint(coords[1]).astype(int)]
    visible = points[:, 2] >= front_depth-extent*.006
    weight = np.maximum(normals @ toward, 0)**3*inside*visible
    info = {'yaw': float(best[1]+settings.get('yaw', 0)), 'pitch': best[2], 'manual': settings}
    return rgba, coords, sampled, weight, info


def colorize(mesh, images, folder, options, progress):
    vertices = np.asarray(mesh.vertices)
    total = np.zeros(len(vertices))
    accumulated = np.zeros((len(vertices), 3))
    calibration = {}
    extent = max(np.ptp(vertices, axis=0))
    for view, base_yaw in zip(VIEWS, [0, -90, 180, 90]):
        progress(f'กำลังจัดแนวและตรวจส่วนที่ถูกบัง · {view}…')
        _, _, sampled, weight, info = project_view(mesh, images[view], view, base_yaw, options, extent)
        weight = weight*(sampled[:, 3]/255)
        accumulated += sampled[:, :3]*weight[:, None]
        total += weight
        calibration[view] = {**info, 'visible_vertices': int(np.count_nonzero(weight))}
    colors = np.full((len(vertices), 4), 255, dtype=np.uint8)
    colors[:, :3] = [170, 180, 210]
    valid = total > 1e-6
    colors[valid, :3] = np.clip(accumulated[valid]/total[valid, None], 0, 255).astype(np.uint8)
    mesh.visual.vertex_colors = colors
    (folder/'color-alignment.json').write_text(json.dumps(calibration, indent=2), encoding='utf-8')


def colorize_single(mesh, image, folder, options, progress):
    """One reference: sharp where it sees the surface, softened colors behind it.

    The unseen side is not generated. It reuses the blurred reference straight
    through the model so hidden parts keep the nearby color without a second face.
    """
    vertices = np.asarray(mesh.vertices)
    extent = max(np.ptp(vertices, axis=0))
    progress('กำลังจัดแนวภาพและฉายสีด้านที่มองเห็น…')
    rgba, coords, _, weight, info = project_view(mesh, image, 'front', 0, options, extent)
    progress('กำลังเติมสีด้านที่ภาพมองไม่เห็น…')
    mask = rgba[:, :, 3] > 128
    # Cutout edges still hold background-tinted pixels; extend interior colors over them.
    core = binary_erosion(mask, iterations=2)
    nearest = distance_transform_edt(~(core if core.any() else mask), return_distances=False, return_indices=True)
    filled = rgba[:, :, :3][nearest[0], nearest[1]].astype(float)
    yy, xx = np.nonzero(mask)
    sigma = .04*max(np.ptp(xx), np.ptp(yy), 1)
    soft = gaussian_filter(filled, sigma=(sigma, sigma, 0))
    sharp, behind = (np.column_stack([map_coordinates(source[:, :, c], coords, order=1, mode='nearest')
        for c in range(3)]) for source in (filled, soft))
    fill = .02
    colors = np.full((len(vertices), 4), 255, dtype=np.uint8)
    colors[:, :3] = np.clip((sharp*weight[:, None]+behind*fill)/(weight[:, None]+fill), 0, 255).astype(np.uint8)
    mesh.visual.vertex_colors = colors
    (folder/'color-alignment.json').write_text(json.dumps({'front': {**info,
        'visible_vertices': int(np.count_nonzero(weight)), 'unseen': 'blurred-through-projection'}}, indent=2), encoding='utf-8')
