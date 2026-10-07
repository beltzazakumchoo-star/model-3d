import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import trimesh
from backend.trellis_support import readiness
from backend.trellis_worker import run


class TrellisTests(unittest.TestCase):
    def test_missing_dino_does_not_report_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('runtime/trellis2/runtime-verified.json', 'trellis2/weights-source.json'):
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('{}')
            state = readiness(root)
            self.assertFalse(state['trellis_ready'])
            self.assertIn('DINOv3', state['trellis_reason'])

    def test_worker_preserves_native_pbr_and_uses_uploaded_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = folder / 'front-cutout.png'
            source.write_bytes(b'reference')
            (folder / 'job.json').write_text(json.dumps({'quality': 'detail', 'color': True}))
            mesh = trimesh.creation.icosphere(subdivisions=1)
            from PIL import Image
            from trimesh.visual.material import PBRMaterial
            import numpy as np
            mesh.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(mesh.vertices), 2)),
                material=PBRMaterial(baseColorTexture=Image.new('RGB', (8, 8), 'blue'),
                                     roughnessFactor=.37, metallicFactor=.2))
            native = folder / 'native.glb'
            mesh.export(native)
            seen = {}
            class FakeEngine:
                def generate(self, settings, out_dir, progress):
                    seen.update(settings)
                    progress('shape', .4)
                    return SimpleNamespace(glb_path=native, detail='high', vertices=len(mesh.vertices),
                        faces=len(mesh.faces), peak_vram_gb=5.9, notes=[])
            result = run(folder, FakeEngine, lambda **kw: kw)
            self.assertEqual(seen['images'], [source])
            self.assertEqual(seen['detail'], 'high')
            self.assertEqual(native.read_bytes(), (folder / 'model.glb').read_bytes())
            self.assertTrue(result['native_pbr'])
            self.assertFalse(result['reference_symmetry'])
            self.assertTrue((folder / 'model-obj.zip').is_file())


if __name__ == '__main__':
    unittest.main()
