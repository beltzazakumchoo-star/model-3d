"""Decode a saved shape in a fresh process, keeping diffusion weights out of VRAM."""
import argparse
import os
from pathlib import Path


def read_tensors(path: Path, prefix: str):
    """Read the tensors under one prefix of a safetensors file with plain file reads.

    safe_open maps the whole 4.9 GB checkpoint copy-on-write, so Windows must
    reserve page file space for all of it even though the VAE is 0.4 GB. When
    that reservation failed, the decoder died with an access violation.
    """
    import json
    import struct
    import numpy as np
    import torch
    with path.open('rb') as file:
        header_size = struct.unpack('<Q', file.read(8))[0]
        header = json.loads(file.read(header_size))
        tensors = {}
        for key, entry in header.items():
            if not key.startswith(prefix):
                continue
            if entry['dtype'] != 'F16':
                raise RuntimeError('Unexpected checkpoint dtype: ' + key + ' ' + entry['dtype'])
            begin, end = entry['data_offsets']
            file.seek(8 + header_size + begin)
            values = np.fromfile(file, dtype='<f2', count=(end - begin) // 2)
            if values.size * 2 != end - begin:
                raise RuntimeError('Checkpoint is truncated at ' + key)
            tensors[key[len(prefix):]] = torch.from_numpy(values.reshape(entry['shape']))
    return tensors


def decode(folder: Path, resolution: int, device: str = 'cuda', single_image=False):
    import torch
    import yaml
    from accelerate import init_empty_weights
    from hy3dgen.shapegen.pipelines import instantiate_from_config, export_to_trimesh

    root = Path(os.environ.get('FOURVIEW_AI_ROOT', 'D:/FourViewAI'))
    model = root / ('models/Hunyuan3D-2/hunyuan3d-dit-v2-0' if single_image
                    else 'models/Hunyuan3D-2mv/hunyuan3d-dit-v2-mv')
    config = yaml.safe_load((model / 'config.yaml').read_text(encoding='utf-8'))
    with init_empty_weights(include_buffers=False):
        vae = instantiate_from_config(config['vae'])
    weights = read_tensors(model / 'model.fp16.safetensors', 'vae.')
    missing = vae.load_state_dict(weights, strict=False, assign=True).missing_keys
    if any(not key.startswith(('encoder.', 'pre_kl.')) for key in missing):
        raise RuntimeError('Missing VAE decoder weights: ' + str(missing))
    vae.encoder = torch.nn.Identity()
    vae.pre_kl = torch.nn.Identity()
    # Decode in FP32 to avoid FP16 overflow in fine surface evaluations.
    dtype = torch.float32
    torch.backends.cuda.matmul.allow_tf32 = True
    vae = vae.eval().to(device=device, dtype=dtype)
    vae.enable_flashvdm_decoder(enabled=True, adaptive_kv_selection=False, mc_algo='mc')
    from backend.surface_decoder import SurfaceDecoder
    vae.volume_decoder = SurfaceDecoder()
    latents = torch.load(folder / 'shape-latents.pt', map_location=device, weights_only=True).to(dtype=dtype)
    print('Decoding saved shape on ' + device, flush=True)
    with torch.no_grad():
        latents = vae(latents / vae.scale_factor)
        if not torch.isfinite(latents).all():
            raise RuntimeError('VAE produced non-finite latents')
        print('Latent decoder finished; extracting surface', flush=True)
        meshes = vae.latents2mesh(latents, bounds=1.01, num_chunks=1000,
            octree_resolution=resolution, mc_level=0.0, mc_algo='mc', enable_pbar=True)
    mesh = export_to_trimesh(meshes)[0]
    if mesh is None or not len(mesh.faces):
        raise RuntimeError('No surface was reconstructed')
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.export(folder / 'shape.glb', include_normals=True)
    print(f'Surface saved: {len(mesh.vertices)} vertices, {len(mesh.faces)} triangles', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--single-image', action='store_true')
    args = parser.parse_args()
    decode(args.folder, args.resolution, args.device, args.single_image)
