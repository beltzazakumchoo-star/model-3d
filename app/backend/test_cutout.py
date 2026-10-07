import unittest
import numpy as np
from PIL import Image
from backend.cutout import clear_enclosed_background


def ring(inside):
    """White background with a blue ring; the cutout wrongly keeps the ring's inside opaque."""
    yy, xx = np.mgrid[:200, :200]
    r = np.hypot(xx-100, yy-100)
    rgb = np.full((200, 200, 3), 253, np.uint8)
    rgb[r < 80] = inside(r[r < 80].shape)
    rgb[(r >= 40) & (r < 80)] = (40, 50, 160)
    alpha = np.where(r < 80, 255, 0).astype(np.uint8)
    return Image.fromarray(rgb), Image.fromarray(np.dstack([rgb, alpha]))


class CutoutTests(unittest.TestCase):
    def test_background_enclosed_by_subject_becomes_transparent(self):
        original, cutout = ring(lambda shape: 253)
        result, cleared = clear_enclosed_background(original, cutout)
        alpha = np.asarray(result)[:, :, 3]
        self.assertGreater(cleared, 4000)
        self.assertEqual(alpha[100, 100], 0)
        self.assertEqual(alpha[100, 160], 255)

    def test_shaded_light_subject_is_kept(self):
        rng = np.random.default_rng(0)
        original, cutout = ring(lambda shape: rng.integers(225, 255, shape+(3,)))
        _, cleared = clear_enclosed_background(original, cutout)
        self.assertEqual(cleared, 0)


if __name__ == '__main__':
    unittest.main()
