"""Refine a sparse surface with floating-point world coordinates."""
import torch
import torch.nn.functional as F
from tqdm import tqdm


class SurfaceDecoder:
    @torch.no_grad()
    def __call__(self, latents, geo_decoder, bounds=1.01, num_chunks=1000,
                 octree_resolution=512, mc_level=0.0, enable_pbar=True, **kwargs):
        if latents.shape[0] != 1:
            raise ValueError('Surface decoding expects one object')
        device, dtype = latents.device, latents.dtype
        limits = [-bounds] * 3 + [bounds] * 3 if isinstance(bounds, (int, float)) else bounds
        minimum = torch.tensor(limits[:3], device=device, dtype=dtype)
        extent = torch.tensor(limits[3:], device=device, dtype=dtype) - minimum
        resolutions = [octree_resolution]
        while resolutions[-1] // 2 >= 63:
            resolutions.append(resolutions[-1] // 2)
        resolutions.reverse()

        def evaluate(indices, resolution):
            # Integer grid indices must become floats before applying spacing.
            spacing = extent / resolution
            values = []
            for start in tqdm(range(0, len(indices), num_chunks),
                              desc=f'Surface refinement {resolution}', disable=not enable_pbar):
                points = indices[start:start+num_chunks].to(dtype) * spacing + minimum
                logits = geo_decoder(queries=points.unsqueeze(0), latents=latents)
                if not torch.isfinite(logits).all():
                    raise RuntimeError('Surface decoder produced non-finite values')
                values.append(logits.reshape(-1).float())
            if not values:
                raise RuntimeError('No surface found while refining this shape')
            return torch.cat(values)

        resolution = resolutions[0]
        axis = torch.arange(resolution+1, device=device)
        indices = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), dim=-1).reshape(-1, 3)
        grid = evaluate(indices, resolution).reshape((resolution+1,) * 3)
        for resolution in resolutions[1:]:
            valid = grid > -9000
            near = valid & ((grid - mc_level).abs() < 0.95)
            for axis_index in range(3):
                first, second = [slice(None)] * 3, [slice(None)] * 3
                first[axis_index], second[axis_index] = slice(None, -1), slice(1, None)
                first, second = tuple(first), tuple(second)
                edge = valid[first] & valid[second] & ((grid[first] >= mc_level) != (grid[second] >= mc_level))
                near[first] |= edge
                near[second] |= edge
            near = F.max_pool3d(near[None, None].to(torch.float16), 3, stride=1, padding=1)[0, 0] > 0
            coarse = torch.nonzero(near, as_tuple=False) * 2
            mask = torch.zeros((resolution+1,) * 3, device=device, dtype=torch.float16)
            mask[coarse[:, 0], coarse[:, 1], coarse[:, 2]] = 1
            mask = F.max_pool3d(mask[None, None], 3, stride=1, padding=1)[0, 0]
            indices = torch.nonzero(mask > 0, as_tuple=False)
            del mask, near, coarse, valid
            values = evaluate(indices, resolution)
            grid = torch.full((resolution+1,) * 3, -10000.0, device=device, dtype=torch.float32)
            grid[indices[:, 0], indices[:, 1], indices[:, 2]] = values
        grid[grid == -10000] = float('nan')
        return grid.unsqueeze(0)
