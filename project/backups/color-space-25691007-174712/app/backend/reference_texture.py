"""Local calibrated reference projection with UVs and depth-tested view selection.

This preserves reference pixels; it does not invent hidden surface detail.
"""
import json
import numpy as np
from PIL import Image
from numba import njit
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from scipy.sparse import coo_matrix
import trimesh

VIEWS = ('front', 'right', 'back', 'left')


@njit(nogil=True)
def depth_buffer(points, faces, width, height):
    depth = np.full((height, width), -1e10, dtype=np.float32)
    for face in faces:
        a, b, c = points[face[0]], points[face[1]], points[face[2]]
        denominator = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
        if abs(denominator) < 1e-10:
            continue
        x0 = max(0, int(np.floor(min(a[0], b[0], c[0]))))
        x1 = min(width-1, int(np.ceil(max(a[0], b[0], c[0]))))
        y0 = max(0, int(np.floor(min(a[1], b[1], c[1]))))
        y1 = min(height-1, int(np.ceil(max(a[1], b[1], c[1]))))
        for y in range(y0, y1+1):
            for x in range(x0, x1+1):
                wa = ((b[1]-c[1])*(x-c[0]) + (c[0]-b[0])*(y-c[1])) / denominator
                wb = ((c[1]-a[1])*(x-c[0]) + (a[0]-c[0])*(y-c[1])) / denominator
                wc = 1-wa-wb
                if min(wa, wb, wc) >= -0.001:
                    z = wa*a[2] + wb*b[2] + wc*c[2]
                    if z > depth[y, x]:
                        depth[y, x] = z
    return depth


def projection(vertices, angle, box):
    radians = np.deg2rad(angle)
    direction = np.array([np.sin(radians), 0, np.cos(radians)])
    horizontal = vertices[:, 0]*np.cos(radians) - vertices[:, 2]*np.sin(radians)
    x0, y0, x1, y1 = box
    px = x0 + (horizontal-horizontal.min()) / max(np.ptp(horizontal), 1e-8) * (x1-x0)
    py = y1 - (vertices[:, 1]-vertices[:, 1].min()) / max(np.ptp(vertices[:, 1]), 1e-8) * (y1-y0)
    return np.column_stack((px, py, vertices @ direction)), direction


def coherent_normals(vertices, normals):
    """Filter over world-space distance, independent of triangle density."""
    extent = max(np.ptp(vertices, axis=0))
    coords = (vertices-vertices.min(axis=0)) / extent * 63
    cells = np.clip(np.rint(coords).astype(int), 0, 63)
    keys = np.ravel_multi_index(cells.T, (64,)*3)
    count = np.bincount(keys, minlength=64**3).reshape((64,)*3)
    denominator = gaussian_filter(count.astype(float), 1.1) + 1e-8
    values = []
    for axis in range(3):
        grid = np.bincount(keys, weights=normals[:, axis], minlength=64**3).reshape((64,)*3)
        filtered = gaussian_filter(grid, 1.1) / denominator
        values.append(map_coordinates(filtered, coords.T, order=1, mode='nearest'))
    result = np.column_stack(values)
    return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-8)


def coherent_views(mesh, scores):
    """Penalize changing source image between neighboring triangles."""
    scores = np.asarray(scores).T
    costs = -np.log(np.maximum(scores / np.maximum(scores.max(axis=1, keepdims=True), 1e-8), 1e-5))
    pairs = mesh.face_adjacency
    rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
    cols = np.concatenate((pairs[:, 1], pairs[:, 0]))
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(mesh.faces),)*2).tocsr()
    labels = np.argmin(costs, axis=1)
    # Alternate updates to avoid neighboring faces oscillating together.
    for iteration in range(32):
        support = np.column_stack([graph @ (labels == view).astype(float) for view in range(4)])
        candidate = np.argmin(costs - 0.9*support, axis=1)
        active = np.arange(len(labels)) % 2 == iteration % 2
        labels[active] = candidate[active]
    return labels


def texture_from_references(mesh, images, folder, progress=lambda message: None):
    original_faces = len(mesh.faces)
    if original_faces > 350000:
        progress(f'กำลังเตรียมรูปทรงสำหรับผิวสี · {original_faces:,} → 300,000 สามเหลี่ยม (เก็บต้นฉบับเต็มไว้)…')
        mesh = mesh.simplify_quadric_decimation(face_count=300000)
    (folder / 'texture-mesh.json').write_text(json.dumps({
        'original_faces': original_faces, 'texture_faces': len(mesh.faces),
        'simplified': len(mesh.faces) < original_faces}), encoding='utf-8')
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    # Average normals on the unchanged geometry to avoid alternating camera
    # choices caused by tiny bumps in the reconstructed surface.
    edges = mesh.edges_unique
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    adjacency = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(vertices),)*2).tocsr()
    degree = np.maximum(np.asarray(adjacency.sum(axis=1)), 1)
    normals = np.asarray(mesh.vertex_normals).copy()
    for _ in range(12):
        normals = 0.35*normals + 0.65*(adjacency @ normals)/degree
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    normals = coherent_normals(vertices, normals)
    calibration = {}
    baked_projections, directions, source_colors, depths, confidence = [], [], [], [], []
    tile = 2048
    for index, (view, base_angle) in enumerate(zip(VIEWS, (0, -90, 180, 90))):
        progress('กำลังจัดแนวภาพและรายละเอียดผิว · ' + view)
        source = images[view].convert('RGBA')
        small = np.asarray(source.resize((160, 160), Image.Resampling.LANCZOS))
        target = small[:, :, 3] > 128
        yy, xx = np.nonzero(target)
        box = (xx.min(), yy.min(), xx.max(), yy.max())
        best = (-1, base_angle)
        # Side references can be three-quarter views rather than exact 90°.
        for offset in (-40, -30, -20, -10, 0, 10, 20, 30, 40):
            points, _ = projection(vertices, base_angle+offset, box)
            silhouette = depth_buffer(points, faces, 160, 160) > -1e9
            overlap = np.count_nonzero(silhouette & target) / max(np.count_nonzero(silhouette | target), 1)
            # Silhouette alone can prefer the wrong yaw for symmetric heads.
            # Require a meaningful improvement before deviating from the slot.
            ranking = overlap - abs(offset)*0.0015
            if ranking > best[0]:
                best = (ranking, base_angle+offset)
        rgba = np.asarray(source.resize((tile, tile), Image.Resampling.LANCZOS))
        mask = rgba[:, :, 3] > 128
        yy, xx = np.nonzero(mask)
        points, direction = projection(vertices, best[1], (xx.min(), yy.min(), xx.max(), yy.max()))
        heights = (vertices[:, 1]-vertices[:, 1].min())/max(np.ptp(vertices[:, 1]), 1e-8)
        head_vertices = heights >= 0.72
        head_mask = mask.copy()
        head_mask[int(yy.min()+(yy.max()-yy.min())*0.28)+1:] = False
        hy, hx = np.nonzero(head_mask)
        if len(hx) and head_vertices.any():
            # Fit the head separately: a curled tail shifts the whole-body
            # bounding box, which otherwise places an eye across the snout.
            h = vertices[:, 0]*np.cos(np.deg2rad(best[1]))-vertices[:, 2]*np.sin(np.deg2rad(best[1]))
            hx0, hx1 = h[head_vertices].min(), h[head_vertices].max()
            local_x = hx.min()+(h-hx0)/max(hx1-hx0, 1e-8)*(hx.max()-hx.min())
            local_y = hy.max()-(vertices[:, 1]-vertices[head_vertices, 1].min())/max(np.ptp(vertices[head_vertices, 1]), 1e-8)*(hy.max()-hy.min())
            blend = np.clip((heights-0.64)/0.08, 0, 1)
            points[:, 0] = points[:, 0]*(1-blend)+local_x*blend
            points[:, 1] = points[:, 1]*(1-blend)+local_y*blend
        # Pad transparent pixels with their nearest foreground color, avoiding
        # black/white borders when filtering the texture along the silhouette.
        nearest = distance_transform_edt(~mask, return_distances=False, return_indices=True)
        padded = rgba[nearest[0], nearest[1], :3]
        depth = depth_buffer(points, faces, tile, tile)
        baked_projections.append(points)
        directions.append(direction)
        source_colors.append(padded)
        depths.append(depth)
        # Fade near silhouette borders where two-dimensional artwork is least
        # reliable, keeping its padded RGB available for unseen undersides.
        confidence.append(np.clip(distance_transform_edt(mask)/5, 0, 1).astype(np.float32))
        # Use affine fits for the body and head, with a smooth height transition.
        # Avoid independently fitting scanlines, which bends armor edges.
        calibration[view] = {'yaw': best[1], 'silhouette_iou': round(best[0]+abs(best[1]-base_angle)*0.0015, 4),
            'projection': 'body-head-affine-blend', 'local_silhouette_fit': False}
    from backend.texture_bake import bake
    result = bake(mesh, normals, baked_projections, directions, source_colors, depths, confidence, folder, progress)
    calibration['texture_method'] = 'continuous-uv-bake-head-antighost-v2'
    calibration['head_protection'] = {'transition_height': [0.64, 0.72], 'method': 'anatomical-half-reference', 'head_local_alignment': True}
    (folder / 'texture-calibration.json').write_text(json.dumps(calibration, indent=2), encoding='utf-8')
    return result


def export_model(mesh, folder):
    import zipfile
    mesh.export(folder / 'model.glb', include_normals=True)
    # Trimesh writes the material and PNG alongside OBJ. Include them in a ZIP
    # so importing the downloaded OBJ keeps its reference texture.
    mesh.export(folder / 'model.obj', include_normals=True, include_color=True)
    mesh.export(folder / 'model.stl')
    with zipfile.ZipFile(folder / 'model-obj.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        for pattern in ('model.obj', '*.mtl', 'material*.png'):
            for file in folder.glob(pattern):
                bundle.write(file, file.name)
