import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi import HTTPException
from backend import server


class EngineValidationTests(unittest.TestCase):
    def test_degenerate_triangles_do_not_collapse_texture_mesh(self):
        import numpy as np
        import trimesh
        from backend.paint_model import prepare_texture_mesh
        mesh = trimesh.creation.icosphere(subdivisions=3)
        original_bounds = mesh.bounds.copy()
        area = mesh.area
        mesh.faces = np.vstack([mesh.faces, np.zeros((1000, 3), dtype=np.int64)])
        result, original_faces = prepare_texture_mesh(mesh, max_faces=500)
        self.assertEqual(original_faces, 2280)
        self.assertGreaterEqual(len(result.faces), 490)
        self.assertGreater(result.area, area * .9)
        np.testing.assert_allclose(result.bounds, original_bounds, atol=.1)

    def test_missing_weights_reject_new_engine(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(server, 'ROOT', Path(temporary)):
            with self.assertRaises(HTTPException) as error:
                server.validate_engine('hunyuan-single', 'image', True)
            self.assertEqual(error.exception.status_code, 503)
            server.validate_engine('legacy', 'image', True)

    def test_single_image_engine_rejects_multiview_input(self):
        with self.assertRaises(HTTPException) as error:
            server.validate_engine('hunyuan-single', 'multiview', True)
        self.assertEqual(error.exception.status_code, 422)

    def test_color_requires_paint_but_shape_only_does_not(self):
        with patch.object(server, 'single_image_readiness', return_value={
                'single_image_ready': True, 'paint_ready': False}):
            server.validate_engine('hunyuan-single', 'image', False)
            with self.assertRaises(HTTPException) as error:
                server.validate_engine('hunyuan-single', 'image', True)
            self.assertEqual(error.exception.status_code, 503)


if __name__ == '__main__':
    unittest.main()
