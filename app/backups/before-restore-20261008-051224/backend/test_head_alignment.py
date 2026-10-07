import unittest
import numpy as np
from backend.head_alignment import mouth_landmark


class MouthTests(unittest.TestCase):
    def test_mouth_notch_precedes_stronger_neck_edge(self):
        mask=np.zeros((512,512),bool)
        for y in range(10,500):
            end=490 if y<110 else 460 if y<126 else 478 if y<156 else 410
            mask[y,350:end]=True
        landmark=mouth_landmark(mask,(10,499,0,500))
        self.assertLess(abs(landmark[0]-110),3)

    def test_solid_profile_is_not_assumed_to_be_an_open_mouth(self):
        mask=np.zeros((512,512),bool)
        mask[10:500,350:490]=True
        self.assertIsNone(mouth_landmark(mask,(10,499,0,500)))


if __name__=='__main__':
    unittest.main()
