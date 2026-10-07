import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from backend.color_export import export_color_glb, has_linear_vertex_colors


class ColorExportTests(unittest.TestCase):
    def test_srgb_roundtrip_and_alpha(self):
        mesh = trimesh.creation.box()
        rgba = np.array([[0, 10, 128, 64], [32, 64, 255, 255]] * 4, dtype=np.uint8)
        mesh.visual.vertex_colors = rgba
        original = mesh.visual.vertex_colors.copy()
        data = export_color_glb(mesh)
        length = struct.unpack_from('<I', data, 12)[0]
        tree = json.loads(data[20:20+length])
        index = tree['meshes'][0]['primitives'][0]['attributes']['COLOR_0']
        accessor = tree['accessors'][index]
        view = tree['bufferViews'][accessor['bufferView']]
        start = 28 + length + view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
        linear = np.frombuffer(data, dtype='<f4', count=32, offset=start).reshape(-1, 4)
        rgb = linear[:, :3]
        srgb = np.where(rgb <= .0031308, rgb * 12.92, 1.055 * rgb ** (1/2.4) - .055)
        np.testing.assert_allclose(srgb * 255, rgba[:, :3], atol=.001)
        np.testing.assert_allclose(linear[:, 3], rgba[:, 3] / 255, atol=1e-7)
        np.testing.assert_array_equal(mesh.visual.vertex_colors, original)
        self.assertEqual(accessor['componentType'], 5126)
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / 'test.glb'
            file.write_bytes(data)
            self.assertTrue(has_linear_vertex_colors(file))
            loaded = trimesh.load(file, force='mesh')
            np.testing.assert_allclose(loaded.vertices, mesh.vertices)
            np.testing.assert_array_equal(loaded.faces, mesh.faces)

    def test_uv_texture_is_preserved(self):
        from PIL import Image
        mesh = trimesh.creation.box()
        mesh.visual = trimesh.visual.TextureVisuals(uv=np.zeros((8, 2)),
            image=Image.new('RGB', (2, 2), (32, 64, 128)))
        data = export_color_glb(mesh)
        length = struct.unpack_from('<I', data, 12)[0]
        tree = json.loads(data[20:20+length])
        self.assertIn('images', tree)
        self.assertIn('TEXCOORD_0', tree['meshes'][0]['primitives'][0]['attributes'])


if __name__ == '__main__':
    unittest.main()
