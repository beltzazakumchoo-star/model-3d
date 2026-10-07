import unittest
import numpy as np
from backend.cpu_rasterizer import rasterize_numpy


class RasterizerTests(unittest.TestCase):
    def test_depth_and_barycentric_interpolation(self):
        vertices = np.array([[-1,-1,.5,1], [1,-1,.5,1], [-1,1,.5,1],
                             [-1,-1,-.5,1], [1,-1,-.5,1], [-1,1,-.5,1]], dtype=np.float32)
        faces = np.array([[0,1,2], [3,4,5]], dtype=np.int32)
        indices, bary = rasterize_numpy(vertices, faces, 5, 5)
        self.assertEqual(indices[1,1], 2)
        np.testing.assert_allclose(bary[1,1], [.5,.25,.25], atol=1e-6)
        self.assertEqual(indices[4,4], 0)
        np.testing.assert_allclose(bary[4,4], [0,0,0])

    def test_degenerate_faces_are_skipped(self):
        vertices = np.array([[0,0,0,1]]*3, dtype=np.float32)
        indices, bary = rasterize_numpy(vertices, np.array([[0,1,2]], dtype=np.int32), 3, 3)
        self.assertFalse(indices.any())
        self.assertFalse(bary.any())


if __name__ == '__main__':
    unittest.main()
