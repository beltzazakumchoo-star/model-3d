import unittest
import numpy as np
from backend.dense_paint import bake_dense


class DenseBakeTests(unittest.TestCase):
    def bake(self, hidden=False):
        uv=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
        faces=np.array([[0,1,2],[1,3,2]])
        projections=np.tile(np.array([[0.,7.,0.],[7.,7.,0.],[0.,0.,0.],[7.,0.,0.]]),(2,1,1))
        images=np.full((2,8,8,4),255,np.uint8)
        images[0,:,:,:3]=30
        # Known sharp uploaded gradient, not generated view pixels.
        images[1,:,:,0]=np.arange(8)[None,:]*30
        images[1,:,:,1]=np.arange(8)[:,None]*30
        depths=np.zeros((2,8,8))
        if hidden:
            depths[1]=1.
        return bake_dense(uv,faces,projections,np.ones((2,4)),images,depths,np.array([.01,.01]),32)

    def test_dense_sampling_preserves_reference_without_splat_holes(self):
        image,valid,reference=self.bake()
        self.assertTrue(valid[1:-1,1:-1].all())
        self.assertEqual(int(reference[16,16]),255)
        self.assertAlmostEqual(int(image[16,16,0]),108,delta=1)
        self.assertAlmostEqual(int(image[16,16,1]),108,delta=1)

    def test_hidden_reference_does_not_project_through_surface(self):
        image,valid,reference=self.bake(hidden=True)
        self.assertEqual(int(reference[16,16]),0)
        self.assertEqual(int(image[16,16,0]),30)


if __name__=='__main__':
    unittest.main()
