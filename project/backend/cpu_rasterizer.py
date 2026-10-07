"""Inference-only rasterizer for Windows without a CUDA compiler.

Matches Hunyuan's pixel centers, depth ordering and perspective barycentrics.
Only rasterization runs on CPU; diffusion still runs on the CUDA GPU.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def rasterize_numpy(pos, faces, height, width):
    indices = np.zeros((height, width), dtype=np.int32)
    bary = np.zeros((height, width, 3), dtype=np.float32)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    screen = np.empty((len(pos), 3), dtype=np.float32)
    for i in range(len(pos)):
        screen[i, 0] = (pos[i, 0] / pos[i, 3] * .5 + .5) * (width - 1) + .5
        screen[i, 1] = (pos[i, 1] / pos[i, 3] * .5 + .5) * (height - 1) + .5
        screen[i, 2] = pos[i, 2] / pos[i, 3]
    for f in range(len(faces)):
        a, b, c = faces[f]
        x0, y0, z0 = screen[a]
        x1, y1, z1 = screen[b]
        x2, y2, z2 = screen[c]
        denominator = (y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        if abs(denominator) < 1e-10:
            continue
        for y in range(max(0, int(min(y0, y1, y2))), min(height, int(max(y0, y1, y2))+1)):
            for x in range(max(0, int(min(x0, x1, x2))), min(width, int(max(x0, x1, x2))+1)):
                u = ((y1-y2)*(x+.5-x2)+(x2-x1)*(y+.5-y2))/denominator
                v = ((y2-y0)*(x+.5-x2)+(x0-x2)*(y+.5-y2))/denominator
                w = 1-u-v
                if min(u, v, w) < -1e-6:
                    continue
                z = u*z0+v*z1+w*z2
                if z < -1 or z > 1 or z >= depth[y, x]:
                    continue
                depth[y, x] = z
                indices[y, x] = f+1
                u, v, w = u/pos[a, 3], v/pos[b, 3], w/pos[c, 3]
                total = u+v+w
                bary[y, x, 0], bary[y, x, 1], bary[y, x, 2] = u/total, v/total, w/total
    return indices, bary


def rasterize(pos, tri, resolution, **kwargs):
    import torch
    indices, bary = rasterize_numpy(pos[0].detach().float().cpu().numpy(),
        tri.detach().cpu().numpy(), int(resolution[0]), int(resolution[1]))
    return torch.from_numpy(indices).to(pos.device), torch.from_numpy(bary).to(pos.device)


def interpolate(col, findices, barycentric, tri):
    import torch
    faces = torch.clamp(findices.long()-1, min=0)
    values = col[0, tri.long()[faces]]
    return (barycentric.unsqueeze(-1)*values).sum(dim=-2).unsqueeze(0)


def install():
    import sys
    import types
    module = types.ModuleType('custom_rasterizer')
    module.rasterize = rasterize
    module.interpolate = interpolate
    sys.modules['custom_rasterizer'] = module
