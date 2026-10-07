"""Bake continuous, visibility-weighted reference colors into a shared UV atlas."""
import numpy as np
import os
from pathlib import Path
import subprocess
import sys
import time
from numba import njit
from PIL import Image
from scipy.ndimage import distance_transform_edt, maximum_filter, maximum_filter1d, minimum_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
import trimesh


@njit(nogil=True)
def raster_bake(uv, faces, mapping, vertex_count, normals, projections, directions, colors, depths, shadows, fringes, gaps, confidence, heights, positions, center, narrow, scales, insets, rises, roofs, ridges, floors, sills, band, top_origin, top_scale, head_center, head_width, extent, size):
    output = np.zeros((size, size, 3), dtype=np.uint8)
    occupied = np.zeros((size, size), dtype=np.bool_)
    hidden = np.zeros((size, size), dtype=np.bool_)
    seen_colors = np.zeros((vertex_count, 3), dtype=np.float64)
    seen_weight = np.zeros(vertex_count, dtype=np.float64)
    source_size = colors.shape[1]
    tolerance = extent * 0.012
    cut = tolerance*2.6
    for face in faces:
        a, b, c = uv[face[0]], uv[face[1]], uv[face[2]]
        ax, ay = a[0]*(size-1), (1-a[1])*(size-1)
        bx, by = b[0]*(size-1), (1-b[1])*(size-1)
        cx, cy = c[0]*(size-1), (1-c[1])*(size-1)
        denom = (by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
        if abs(denom) < 1e-10:
            continue
        for y in range(max(0, int(np.floor(min(ay, by, cy)))), min(size-1, int(np.ceil(max(ay, by, cy))))+1):
            for x in range(max(0, int(np.floor(min(ax, bx, cx)))), min(size-1, int(np.ceil(max(ax, bx, cx))))+1):
                wa = ((by-cy)*(x-cx)+(cx-bx)*(y-cy))/denom
                wb = ((cy-ay)*(x-cx)+(ax-cx)*(y-cy))/denom
                wc = 1-wa-wb
                if min(wa, wb, wc) < -0.001:
                    continue
                n = wa*normals[face[0]]+wb*normals[face[1]]+wc*normals[face[2]]
                n /= max(np.sqrt((n*n).sum()), 1e-8)
                sampled = np.zeros((4, 3), dtype=np.float32)
                weights = np.zeros(4, dtype=np.float32)
                fallback = np.zeros(4, dtype=np.float32)
                clearest = 0.0
                steepest = 0.0
                pixels = np.zeros((4, 2), dtype=np.float32)
                height = wa*heights[face[0]]+wb*heights[face[1]]+wc*heights[face[2]]
                position = wa*positions[face[0]]+wb*positions[face[1]]+wc*positions[face[2]]
                for view in range(4):
                    p = wa*projections[view, face[0]]+wb*projections[view, face[1]]+wc*projections[view, face[2]]
                    px, py = min(max(p[0], 0), source_size-1.001), min(max(p[1], 0), source_size-1.001)
                    pixels[view, 0], pixels[view, 1] = px, py
                    ix, iy = int(px), int(py)
                    dx, dy = px-ix, py-iy
                    for channel in range(3):
                        sampled[view, channel] = ((1-dx)*(1-dy)*colors[view, iy, ix, channel]
                            + dx*(1-dy)*colors[view, iy, ix+1, channel]
                            + (1-dx)*dy*colors[view, iy+1, ix, channel]
                            + dx*dy*colors[view, iy+1, ix+1, channel])
                    facing = max((n*directions[view]).sum(), 0)
                    steepest = max(steepest, facing)
                    # Continuous angular weights avoid the triangle-level hard
                    # camera changes that tear armor patterns apart.
                    fallback[view] = (facing+0.005)**4
                    sx, sy = int(round(px)), int(round(py))
                    delta = depths[view, sy, sx]-p[2]
                    visible = np.exp(-max(delta-tolerance, 0)/max(tolerance*0.7, 1e-8))
                    # Clearly behind another part (leg behind the tail tuft), or
                    # beside the outline between two parts, where the drawing
                    # rarely matches the mesh: a wider drawn tail spills onto
                    # the leg behind it, a wider drawn udder onto the leg in
                    # front. Such a pixel may belong to the other part.
                    if visible < 0.1 or fringes[view, sy, sx] or shadows[view, sy, sx]-p[2] > tolerance*2.6:
                        visible = 0
                    # Glimpsed through a gap between nearer parts (the far leg
                    # between the near leg and the udder): the drawing's gap is
                    # never the same width, so it shows one of those parts.
                    if gaps[view, sy, sx]-p[2] > tolerance*5:
                        visible = 0
                    fallback[view] *= visible
                    weights[view] = fallback[view]*confidence[view, iy, ix]
                    if visible > 0:
                        clearest = max(clearest, facing)
                occupied[y, x] = True
                hidden[y, x] = False
                tx = min(max(int(round((position[0]-top_origin[0])*top_scale)), 0), insets.shape[1]-1)
                tz = min(max(int(round((position[2]-top_origin[1])*top_scale)), 0), insets.shape[0]-1)
                # Top and underside: the cameras see these edge-on, so a
                # projection drags one pixel row across the surface. Read the
                # nearer side reference at the outline instead, unless a part
                # reaching past this height (leg, udder, ear) stands in the
                # way: the outline pixel would then show that part.
                side = 0
                if ((position-center)*directions[narrow[0]]).sum() < 0:
                    side = 1
                primary = narrow[side]
                borrowed = False
                drop = 0.0
                if n[1] > 0:
                    if roofs[tz, tx]-position[1] <= cut and ridges[side, tz, tx]-position[1] <= 2*cut:
                        inset = insets[tz, tx]
                        if inset > 0:
                            # Wide top: continue the reference over the
                            # shoulder at its own scale, reading back and
                            # forth through the band just below it, so the top
                            # keeps the flank's block pattern without copying
                            # features from further down.
                            borrowed = True
                            rise = rises[tz, tx]
                            along = np.sqrt(inset*inset+rise*rise) % (2*band)
                            drop = (rise+band-abs(along-band))*scales[primary]
                        elif steepest < 0.25:
                            borrowed = True
                elif steepest < 0.25 and position[1]-floors[tz, tx] <= cut and position[1]-sills[side, tz, tx] <= 2*cut:
                    borrowed = True
                if borrowed:
                    px = pixels[primary, 0]
                    py = min(pixels[primary, 1]+drop, source_size-1.001)
                    ix, iy = int(px), int(py)
                    dx, dy = px-ix, py-iy
                    for channel in range(3):
                        sampled[primary, channel] = ((1-dx)*(1-dy)*colors[primary, iy, ix, channel]
                            + dx*(1-dy)*colors[primary, iy, ix+1, channel]
                            + (1-dx)*dy*colors[primary, iy+1, ix, channel]
                            + dx*dy*colors[primary, iy+1, ix+1, channel])
                    weights[:] = 0
                    weights[primary] = 1
                    total = 1.0
                elif clearest < 0.25:
                    # Edge-on with no usable outline, or the camera facing
                    # this surface is blocked by another part. Projecting
                    # anyway paints a second tail on the leg behind it, so
                    # leave it for spread_colors to fill.
                    hidden[y, x] = True
                    continue
                else:
                    total = weights.sum()
                    if total < 1e-8:
                        # Outside the reference silhouette; use the closest
                        # foreground sample rather than projecting a background.
                        weights = fallback
                        total = max(weights.sum(), 1e-12)
                # In the head region, use one reference at each surface point.
                # Averaging two independently drawn eyes/teeth creates duplicate
                # features. Coherent normals keep camera choice spatially stable.
                primary = np.argmax(weights)
                if height > 0.64 and not borrowed:
                    horizontal = position[0]-head_center[0]
                    # Keep a stable source over each anatomical half of the
                    # head, independent of normals on teeth/armor bumps.
                    preferred = 1 if horizontal < 0 else 3
                    if abs(horizontal) < head_width*0.12:
                        preferred = 0 if position[2] >= head_center[2] else 2
                    if weights[preferred] > 1e-10:
                        primary = preferred
                disagreement = 0.0
                for view in range(4):
                    if weights[view] > total*0.15:
                        disagreement = max(disagreement, np.abs(sampled[view]-sampled[primary]).max())
                protection = min(max((height-0.64)/0.08, 0), 1)
                # Reject strongly conflicting samples in the transition band;
                # preserve the existing continuous blend on the lower body.
                if disagreement > 75 and height > 0.64:
                    protection = max(protection, 0.9)
                for view in range(4):
                    weights[view] *= 1-protection
                weights[primary] += total*protection
                for channel in range(3):
                    value = (sampled[:, channel]*weights).sum()/total
                    output[y, x, channel] = min(max(value, 0), 255)
                # Only squarely seen points lend their color to hidden ones;
                # obliquely seen ones sit beside outlines and are often tinted
                # by the neighboring part.
                if borrowed or clearest >= 0.5:
                    for channel in range(3):
                        seen_colors[mapping[face[0]], channel] += wa*output[y, x, channel]
                        seen_colors[mapping[face[1]], channel] += wb*output[y, x, channel]
                        seen_colors[mapping[face[2]], channel] += wc*output[y, x, channel]
                    seen_weight[mapping[face[0]]] += wa
                    seen_weight[mapping[face[1]]] += wb
                    seen_weight[mapping[face[2]]] += wc
    return output, occupied, hidden, seen_colors, seen_weight


@njit(nogil=True)
def fill_hidden(uv, faces, vertex_colors, hidden, output):
    size = output.shape[0]
    for face in faces:
        a, b, c = uv[face[0]], uv[face[1]], uv[face[2]]
        ax, ay = a[0]*(size-1), (1-a[1])*(size-1)
        bx, by = b[0]*(size-1), (1-b[1])*(size-1)
        cx, cy = c[0]*(size-1), (1-c[1])*(size-1)
        denom = (by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
        if abs(denom) < 1e-10:
            continue
        for y in range(max(0, int(np.floor(min(ay, by, cy)))), min(size-1, int(np.ceil(max(ay, by, cy))))+1):
            for x in range(max(0, int(np.floor(min(ax, bx, cx)))), min(size-1, int(np.ceil(max(ax, bx, cx))))+1):
                if not hidden[y, x]:
                    continue
                wa = ((by-cy)*(x-cx)+(cx-bx)*(y-cy))/denom
                wb = ((cy-ay)*(x-cx)+(ax-cx)*(y-cy))/denom
                wc = 1-wa-wb
                if min(wa, wb, wc) < -0.001:
                    continue
                for channel in range(3):
                    value = (wa*vertex_colors[face[0], channel]+wb*vertex_colors[face[1], channel]
                        + wc*vertex_colors[face[2], channel])
                    output[y, x, channel] = min(max(value, 0), 255)


def spread_colors(mesh, colors, weight):
    """Carry seen colors along the surface into parts no reference shows."""
    known = weight > 1e-6
    if not known.any():
        return np.full((len(weight), 3), 128.0)
    result = np.zeros((len(weight), 3))
    result[known] = colors[known]/weight[known, None]
    # Each unseen point takes the color of the closest seen one, measured along
    # the surface: the tail's stays on the tail and the leg's on the leg even
    # where they nearly touch. Copying, not averaging, keeps the udder pink
    # instead of fading into the belly; creases cost extra to cross so a part
    # is filled from itself first.
    edges = mesh.edges_unique
    normals = np.asarray(mesh.vertex_normals)
    crease = 1-(normals[edges[:, 0]]*normals[edges[:, 1]]).sum(axis=1)
    cost = np.asarray(mesh.edges_unique_length)*(1+12*np.clip(crease, 0, 2))+1e-9
    graph = coo_matrix((cost, (edges[:, 0], edges[:, 1])), shape=(len(weight),)*2).tocsr()
    _, _, nearest = dijkstra(graph, directed=False, indices=np.flatnonzero(known),
        min_only=True, return_predecessors=True)
    reached = ~known & (nearest >= 0)
    result[reached] = result[nearest[reached]]
    result[~known & ~reached] = result[known].mean(axis=0)
    return result


def bake(mesh, normals, projections, directions, colors, depths, confidence, folder, progress):
    progress('กำลังคลี่ UV ของรูปทรงเดิมเพื่อทำผิวต่อเนื่อง…')
    np.savez(folder / 'uv-input.npz', vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.uint32))
    with (folder / 'unwrap.log').open('w', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, '-u', '-m', 'backend.unwrap_texture', str(folder)],
            cwd=Path(__file__).resolve().parent.parent, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        started = time.monotonic()
        while True:
            try:
                child.wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                elapsed = round(time.monotonic()-started)
                progress(f'กำลังคลี่ UV · {len(mesh.faces):,} สามเหลี่ยม · {elapsed} วินาที · ใช้ CPU…')
                if elapsed >= 300:
                    if os.name == 'nt':
                        subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                            stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                    else:
                        child.kill()
                    child.wait()
                    raise RuntimeError('คลี่ UV เกิน 5 นาที รูปทรงต้นฉบับบันทึกไว้แล้ว สามารถสร้างต่อได้')
    if child.returncode:
        raise RuntimeError('คลี่ UV ไม่สำเร็จ ดู unwrap.log ในโฟลเดอร์ผลงาน')
    with np.load(folder / 'uv-layout.npz') as layout:
        mapping, indices, uv = layout['mapping'], layout['indices'], layout['uv']
    progress('กำลังอบสี 4K และเกลี่ยรอยต่อระหว่างมุมภาพ…')
    vertices = np.asarray(mesh.vertices)
    heights = (vertices[:, 1]-vertices[:, 1].min())/max(np.ptp(vertices[:, 1]), 1e-8)
    head = vertices[heights >= 0.72]
    head_center = (head.min(axis=0)+head.max(axis=0))/2
    depths = np.asarray(depths, dtype=np.float32)
    directions = np.asarray(directions, dtype=np.float32)
    extent = float(np.ptp(vertices, axis=0).max())
    # Depth of nearby outlines of parts in front. A drawn tail is rarely the
    # exact shape of the mesh tail, so its pixels spill onto what lies behind.
    cut = extent*0.012*2.6
    radius = depths.shape[1]*3//128
    shadows = np.empty_like(depths)
    fringes = np.empty(depths.shape, dtype=np.bool_)
    gaps = np.empty_like(depths)
    window = depths.shape[1]//8
    reach = window//2+1
    for view, depth in enumerate(depths):
        behind = minimum_filter(depth, size=3)
        outline = depth-behind > cut
        shadows[view] = maximum_filter(np.where(outline, depth, np.float32(-1e10)), size=2*radius+1)
        # The part in front is unreliable too, in a narrower band.
        fringes[view] = maximum_filter(outline & (behind > -1e9), size=radius+1)
        # Depth of the nearer surfaces flanking each pixel on both sides
        # (left and right, or above and below) within an eighth of the image.
        for axis in (0, 1):
            run = maximum_filter1d(depth, size=window, axis=axis, mode='constant', cval=-1e10)
            ahead, back = np.roll(run, -reach, axis=axis), np.roll(run, reach, axis=axis)
            np.moveaxis(ahead, axis, 0)[-reach:] = -1e10
            np.moveaxis(back, axis, 0)[:reach] = -1e10
            flanked = np.minimum(ahead, back)
            gaps[view] = flanked if axis == 0 else np.maximum(gaps[view], flanked)
    # Plan view of the open top. insets: on a wide top such as an animal's
    # back, the distance inward from the shoulder line where side references
    # start to stretch (45 degrees). rises: height gained since that line.
    from backend.reference_texture import depth_buffer
    faces = np.asarray(mesh.faces)
    grid = 2048
    top_origin = vertices[:, [0, 2]].min(axis=0)
    top_scale = (grid-1)/max(np.ptp(vertices[:, [0, 2]], axis=0).max(), 1e-8)
    plan = np.column_stack(((vertices[:, 0]-top_origin[0])*top_scale,
        (vertices[:, 2]-top_origin[1])*top_scale, vertices[:, 1]))
    roof = depth_buffer(plan, faces, grid, grid)
    cells = np.clip(np.rint(plan[:, :2]).astype(int), 0, grid-1)
    steepest = (normals @ directions.T.astype(np.float64)).max(axis=1)
    open_top = (normals[:, 1] > 0) & (roof[cells[:, 1], cells[:, 0]]-vertices[:, 1] < cut)
    insets = np.zeros((grid, grid), dtype=np.float32)
    rises = np.zeros((grid, grid), dtype=np.float32)
    sloped = faces[(open_top & (steepest < 0.7))[faces].all(axis=1)]
    if len(sloped):
        cap = depth_buffer(plan, sloped, grid, grid) > -1e9
        # Keep only the wide part. A head or tail is too narrow to show the
        # stretch, and the front and back references still describe it.
        reach = extent*0.07*top_scale
        cap &= distance_transform_edt(~(distance_transform_edt(cap) >= reach)) <= reach
        distance, edge = distance_transform_edt(cap, return_indices=True)
        shoulder = roof[edge[0], edge[1]]
        insets[cap] = distance[cap]/top_scale
        rises[cap] = np.where(shoulder > -1e9, np.maximum(roof-shoulder, 0), 0)[cap]
    scales = np.array([np.ptp(view[:, 1]) for view in projections], dtype=np.float32)/max(np.ptp(vertices[:, 1]), 1e-8)
    # Views ordered by how little of the model lies along their line of sight.
    spans = [np.ptp(vertices @ direction) for direction in directions]
    narrow = np.argsort(spans, kind='stable')[:2].astype(np.int64)
    if abs(float(directions[narrow[0]] @ directions[narrow[1]])+1) > 0.5:
        narrow[1] = narrow[0]
    # Highest roof and lowest floor between each plan cell and the two narrow
    # cameras, to tell when another part stands in an outline's line of sight.
    floor = -depth_buffer(np.column_stack((plan[:, :2], -vertices[:, 1])), faces, grid, grid)
    ridges = np.empty((2, grid, grid), dtype=np.float32)
    sills = np.empty((2, grid, grid), dtype=np.float32)
    for side, view in enumerate(narrow):
        axis = 1 if abs(directions[view][0]) >= abs(directions[view][2]) else 0
        step = -1 if directions[view][0 if axis == 1 else 2] > 0 else 1
        toward = (slice(None, None, step), slice(None)) if axis == 0 else (slice(None), slice(None, None, step))
        ridges[side][toward] = np.maximum.accumulate(roof[toward], axis=axis)
        sills[side][toward] = np.minimum.accumulate(floor[toward], axis=axis)
    texture, occupied, hidden, seen_colors, seen_weight = raster_bake(np.asarray(uv), np.asarray(indices),
        np.asarray(mapping, dtype=np.int64), len(vertices),
        np.asarray(normals[mapping], dtype=np.float32),
        np.asarray(projections, dtype=np.float32)[:, mapping],
        directions, np.asarray(colors), depths, shadows, fringes, gaps, np.asarray(confidence, dtype=np.float32),
        heights[mapping].astype(np.float32), vertices[mapping].astype(np.float32),
        ((vertices.min(axis=0)+vertices.max(axis=0))/2).astype(np.float32), narrow,
        scales, insets, rises, roof, ridges, floor, sills, extent*0.04, top_origin.astype(np.float32), float(top_scale),
        head_center.astype(np.float32), float(np.ptp(head[:, 0])), extent, 4096)
    if not occupied.any():
        raise RuntimeError('ไม่พบพื้นผิว UV สำหรับอบสี')
    if hidden.any():
        fill_hidden(np.asarray(uv), np.asarray(indices),
            spread_colors(mesh, seen_colors, seen_weight)[mapping], hidden, texture)
    # Extrude chart colors into gutters to prevent black lines during mipmapping.
    nearest = distance_transform_edt(~occupied, return_distances=False, return_indices=True)
    texture[~occupied] = texture[nearest[0][~occupied], nearest[1][~occupied]]
    image = Image.fromarray(texture)
    image.save(folder / 'reference-atlas.png')
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=image,
        baseColorFactor=[255, 255, 255, 255], metallicFactor=0, roughnessFactor=1, doubleSided=True)
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices)[mapping], faces=indices,
        vertex_normals=np.asarray(mesh.vertex_normals)[mapping], process=False,
        visual=trimesh.visual.texture.TextureVisuals(uv=uv, material=material))
