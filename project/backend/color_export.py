"""Export image-derived sRGB vertex colors as glTF linear float attributes."""
import numpy as np
import json
import struct
from trimesh.exchange.gltf import export_glb


def has_linear_vertex_colors(path):
    with open(path, 'rb') as stream:
        header = stream.read(20)
        length = struct.unpack_from('<I', header, 12)[0]
        tree = json.loads(stream.read(length))
    return tree.get('asset', {}).get('extras', {}).get('fourviewVertexColorSpace') == 'linear'


def export_color_glb(mesh):
    def convert(buffers, tree):
        accessors = list(tree['accessors'].values())
        keys = list(buffers)
        converted = set()
        for item in tree.get('meshes', []):
            for primitive in item['primitives']:
                index = primitive['attributes'].get('COLOR_0')
                if index is None or index in converted:
                    continue
                accessor = accessors[index]
                if accessor['componentType'] != 5121 or accessor['type'] != 'VEC4':
                    raise ValueError('Expected trimesh RGBA byte vertex colors')
                rgba = np.frombuffer(buffers[keys[accessor['bufferView']]], dtype=np.uint8,
                    count=accessor['count'] * 4, offset=accessor.get('byteOffset', 0)).reshape(-1, 4)
                linear = rgba.astype(np.float32) / 255
                rgb = linear[:, :3]
                linear[:, :3] = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
                accessor.update(bufferView=len(buffers), byteOffset=0, componentType=5126,
                    min=linear.min(axis=0).tolist(), max=linear.max(axis=0).tolist())
                accessor.pop('normalized', None)
                buffers[f'fourview-linear-color-{index}'] = linear.astype('<f4').tobytes()
                converted.add(index)
        tree['asset'].setdefault('extras', {})['fourviewVertexColorSpace'] = 'linear'

    return export_glb(mesh, include_normals=True, buffer_postprocessor=convert)
