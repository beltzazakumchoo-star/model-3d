"""Inverse UV sampling: every atlas texel reads a visible source pixel.

Avoid forward splatting holes and preserve the uploaded reference on its
visible side; generated views supply surfaces absent from that reference.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def bake_dense(uv, faces, projections, facing, images, depths, tolerances, size, reference_count=1):
    output = np.zeros((size, size, 3), np.uint8)
    occupied = np.zeros((size, size), np.bool_)
    reference = np.zeros((size, size), np.uint8)
    edge = images.shape[1]-1
    for face in faces:
        a, b, c = face
        ax, ay = uv[a, 0]*(size-1), (1-uv[a, 1])*(size-1)
        bx, by = uv[b, 0]*(size-1), (1-uv[b, 1])*(size-1)
        cx, cy = uv[c, 0]*(size-1), (1-uv[c, 1])*(size-1)
        denom = (by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
        if abs(denom) < 1e-10:
            continue
        for y in range(max(0, int(min(ay, by, cy))), min(size, int(max(ay, by, cy))+2)):
            for x in range(max(0, int(min(ax, bx, cx))), min(size, int(max(ax, bx, cx))+2)):
                wa = ((by-cy)*(x-cx)+(cx-bx)*(y-cy))/denom
                wb = ((cy-ay)*(x-cx)+(ax-cx)*(y-cy))/denom
                wc = 1-wa-wb
                if min(wa, wb, wc) < -1e-5:
                    continue
                rgb = np.zeros(3, np.float64)
                total = 0.
                original = np.zeros(3, np.float64)
                alpha = 0.
                for view in range(len(images)):
                    p = wa*projections[view,a]+wb*projections[view,b]+wc*projections[view,c]
                    px, py = p[0], p[1]
                    f = wa*facing[view,a]+wb*facing[view,b]+wc*facing[view,c]
                    if px < 0 or py < 0 or px >= edge or py >= edge or f <= .05:
                        continue
                    ix, iy = int(px), int(py)
                    if depths[view,int(round(py)),int(round(px))]-p[2] > tolerances[view]:
                        continue
                    dx, dy = px-ix, py-iy
                    color = ((1-dx)*(1-dy)*images[view,iy,ix].astype(np.float64)
                        + dx*(1-dy)*images[view,iy,ix+1]
                        + (1-dx)*dy*images[view,iy+1,ix]
                        + dx*dy*images[view,iy+1,ix+1])
                    if view >= len(images)-reference_count:
                        # Feather at silhouettes and grazing surfaces. Never
                        # project the visible eye/mouth through to the back.
                        candidate = min(max((f-(.08 if reference_count==2 else .25)) /
                            (.27 if reference_count==2 else .45), 0.), 1.)*min(max((color[3]-220)/35, 0.), 1.)
                        if candidate > alpha:
                            alpha = candidate
                            original = color[:3]
                    else:
                        weight = f**6
                        rgb += color[:3]*weight
                        total += weight
                occupied[y,x] = True
                if total > 1e-12:
                    rgb /= total
                elif alpha > 0:
                    rgb = original
                else:
                    occupied[y,x] = False
                output[y,x] = np.minimum(np.maximum(rgb*(1-alpha)+original*alpha,0),255).astype(np.uint8)
                reference[y,x] = int(alpha*255)
    return output, occupied, reference


def mirror_reference_projection(vertices, normals, projection, yaw, pitch, box):
    """Reflect across input-camera depth, keeping the same image calibration.

    Input single-image models are reconstructed in XY with Z as depth. Flip
    Z, not image X: eyes stay on the head and the tail stays on the tail.
    """
    yaw, pitch = np.deg2rad(yaw), np.deg2rad(pitch)
    right=np.array([np.cos(yaw),0.,-np.sin(yaw)])
    toward=np.array([np.sin(yaw)*np.cos(pitch),np.sin(pitch),np.cos(yaw)*np.cos(pitch)])
    up=np.cross(toward,right)
    plane=float(np.median(vertices[:,2]))
    delta=np.zeros_like(vertices)
    delta[:,2]=2*(plane-vertices[:,2])
    mirrored=projection.copy()
    mirrored[:,0]+=(delta@right)*(box[2]-box[0])/max(np.ptp(vertices@right),1e-8)
    mirrored[:,1]-=(delta@up)*(box[3]-box[1])/max(np.ptp(vertices@up),1e-8)
    mirrored[:,2]+=delta@toward
    reflected_normals=normals.copy()
    reflected_normals[:,2]*=-1
    return mirrored,np.maximum(reflected_normals@toward,0),plane


def texture_from_views(mesh, render, views, source, size, folder, symmetry=False):
    import torch
    from PIL import Image
    from scipy.ndimage import distance_transform_edt, binary_erosion
    from backend.aligned_vertex import project_view
    from backend.reference_texture import depth_buffer
    from hy3dgen.texgen.differentiable_renderer.mesh_render import get_mv_matrix, transform_pos

    print('Dense: prepare cameras', flush=True)
    resolution = 1024
    projections, facing, images, depths, tolerances = [], [], [], [], []
    transformed = render.vtx_pos.detach().numpy().copy()
    transformed_mesh = __import__('trimesh').Trimesh(transformed, mesh.faces, process=False)
    normals = np.asarray(transformed_mesh.vertex_normals)
    for view, elev, azim in zip(views,[0,0,0,0,90,-90],[0,90,180,270,0,180]):
        mv = get_mv_matrix(elev=elev, azim=azim, camera_distance=render.camera_distance, center=None)
        camera = transform_pos(mv, render.vtx_pos, keepdim=True)
        clip = transform_pos(render.camera_proj_mat, camera, keepdim=True).detach().numpy()
        ndc = clip[:,:3]/clip[:,3:4]
        p = np.column_stack(((ndc[:,0]*.5+.5)*(resolution-1),
                             (ndc[:,1]*.5+.5)*(resolution-1), -ndc[:,2]))
        direction = -np.asarray(mv)[2,:3]
        projections.append(p)
        facing.append(np.maximum(normals@direction,0))
        images.append(np.asarray(view.convert('RGBA').resize((resolution,resolution),Image.Resampling.LANCZOS)))
        depths.append(depth_buffer(p, np.asarray(mesh.faces), resolution,resolution))
        tolerances.append(.002)
    print('Dense: align original', flush=True)
    extent = float(np.ptp(mesh.vertices,axis=0).max())
    _, coords, _, _, calibration = project_view(mesh,source,'front',0,{'auto':True,'views':{}},extent)
    from backend.aligned_vertex import camera as reference_camera
    rgba = np.asarray(source.convert('RGBA').resize((resolution,resolution),Image.Resampling.LANCZOS)).copy()
    yy,xx = np.nonzero(rgba[:,:,3]>128)
    p,direction = reference_camera(np.asarray(mesh.vertices),calibration['yaw'],calibration['pitch'],
        (xx.min(),yy.min(),xx.max(),yy.max()))
    base_projection=p.copy()
    ref_facing=np.maximum(np.asarray(mesh.vertex_normals)@direction,0)
    if symmetry:
        # Side-view head details need their own camera. Whole-body silhouette
        # fitting can tilt the jaw to accommodate the curled tail or shoulders.
        vertices=np.asarray(mesh.vertices)
        heights=(vertices[:,1]-vertices[:,1].min())/max(np.ptp(vertices[:,1]),1e-8)
        head_blend=np.clip((heights-.66)/.10,0,1)
        local,_=reference_camera(vertices,0,0,(xx.min(),yy.min(),xx.max(),yy.max()))
        p=p*(1-head_blend[:,None])+local*head_blend[:,None]
        ref_facing=ref_facing*(1-head_blend)+np.maximum(mesh.vertex_normals[:,2],0)*head_blend
        calibration['head_camera']={'yaw':0,'pitch':0,'transition_height':[.66,.76]}
    # Extend edge colors but erode alpha, so background halos cannot be baked.
    core = binary_erosion(rgba[:,:,3]>220,iterations=2)
    nearest = distance_transform_edt(~core,return_distances=False,return_indices=True)
    rgba[:,:,:3] = rgba[nearest[0],nearest[1],:3]
    rgba[:,:,3] = core.astype(np.uint8)*255
    projections.append(p)
    facing.append(ref_facing)
    images.append(rgba)
    depths.append(depth_buffer(p,np.asarray(mesh.faces),resolution,resolution))
    tolerances.append(extent*.006)
    if symmetry:
        mirrored,mirror_facing,plane=mirror_reference_projection(np.asarray(mesh.vertices),
            np.asarray(mesh.vertex_normals),base_projection,calibration['yaw'],calibration['pitch'],
            (xx.min(),yy.min(),xx.max(),yy.max()))
        reflected=vertices.copy()
        reflected[:,2]=2*plane-reflected[:,2]
        local,_=reference_camera(reflected,0,0,(xx.min(),yy.min(),xx.max(),yy.max()))
        mirrored=mirrored*(1-head_blend[:,None])+local*head_blend[:,None]
        mirror_facing=mirror_facing*(1-head_blend)+np.maximum(-mesh.vertex_normals[:,2],0)*head_blend
        projections.append(mirrored)
        facing.append(mirror_facing)
        images.append(rgba)
        # Use the real reference-side depth buffer, not the reflected geometry's
        # own buffer. A bounded tolerance accommodates reconstruction asymmetry.
        depths.append(depths[-1].copy())
        tolerances.append(extent*.025)
        calibration.update(symmetry=True,symmetry_axis='z',symmetry_plane=plane)
    print('Dense: sample atlas', flush=True)
    texture,valid,reference = bake_dense(np.asarray(mesh.visual.uv),np.asarray(mesh.faces),
        np.asarray(projections),np.asarray(facing),np.asarray(images),np.asarray(depths),np.asarray(tolerances),size,
        2 if symmetry else 1)
    if not valid.any():
        raise RuntimeError('ไม่พบพิกเซลผิวที่มองเห็นระหว่างอบ texture')
    # Only pad empty texels; do not smooth or replace valid reference pixels.
    print('Dense: pad atlas', flush=True)
    nearest = distance_transform_edt(~valid,return_distances=False,return_indices=True)
    texture[~valid] = texture[nearest[0][~valid],nearest[1][~valid]]
    Image.fromarray(reference).save(folder/'reference-coverage.png')
    import json
    (folder/'reference-calibration.json').write_text(json.dumps(calibration),encoding='utf-8')
    return Image.fromarray(texture)


