import unittest
import numpy as np
from backend.dense_paint import bake_dense, mirror_reference_projection


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

    def test_most_frontal_upload_wins_among_references(self):
        uv=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
        faces=np.array([[0,1,2],[1,3,2]])
        projections=np.tile(np.array([[0.,7.,0.],[7.,7.,0.],[0.,0.,0.],[7.,0.,0.]]),(3,1,1))
        images=np.full((3,8,8,4),255,np.uint8)
        images[0,:,:,:3]=30
        images[1,:,:,:3]=(200,0,0)
        images[2,:,:,:3]=(0,0,200)
        # Both uploads reach full weight; the more frontal one supplies color.
        facing=np.array([[1.]*4,[.8]*4,[.95]*4])
        image,_,_=bake_dense(uv,faces,projections,facing,images,np.zeros((3,8,8)),np.full(3,.01),32,2)
        self.assertEqual(tuple(image[16,16]),(0,0,200))

    def test_hidden_reference_does_not_project_through_surface(self):
        image,valid,reference=self.bake(hidden=True)
        self.assertEqual(int(reference[16,16]),0)
        self.assertEqual(int(image[16,16,0]),30)

    def test_side_reflection_keeps_eye_and_tail_at_same_image_positions(self):
        vertices=np.array([[-1.,-1.,-.2],[1.,1.,-.2],[-1.,-1.,.2],[1.,1.,.2]])
        normals=np.array([[0.,0.,-1.],[0.,0.,-1.],[0.,0.,1.],[0.,0.,1.]])
        projection=np.array([[0.,100.,-.2],[100.,0.,-.2],[0.,100.,.2],[100.,0.,.2]])
        mirrored,facing,plane=mirror_reference_projection(vertices,normals,projection,0,0,(0,0,100,100))
        np.testing.assert_allclose(mirrored[:,:2],projection[:,:2])
        np.testing.assert_allclose(mirrored[:,2],-projection[:,2])
        np.testing.assert_allclose(facing,[1,1,0,0])
        self.assertEqual(plane,0)

    def test_opposite_side_reads_reference_instead_of_generated_color(self):
        uv=np.array([[0.,0.],[1.,0.],[0.,1.]])
        faces=np.array([[0,1,2]])
        projections=np.tile(np.array([[0.,7.,0.],[7.,7.,0.],[0.,0.,0.]]),(3,1,1))
        images=np.full((3,8,8,4),255,np.uint8)
        images[0,:,:,:3]=30
        images[1:,:,:,:3]=[0,180,255]
        facing=np.array([[1.,1.,1.],[0.,0.,0.],[1.,1.,1.]])
        texture,valid,coverage=bake_dense(uv,faces,projections,facing,images,
            np.zeros((3,8,8)),np.ones(3)*.01,32,2)
        np.testing.assert_array_equal(texture[20,5],[0,180,255])
        self.assertEqual(int(coverage[20,5]),255)


if __name__=='__main__':
    unittest.main()
