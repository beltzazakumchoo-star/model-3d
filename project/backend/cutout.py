"""Clear background that rembg keeps when the subject encloses it (tail loops, gaps between legs)."""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, label


def clear_enclosed_background(original, cutout, tolerance=20, min_fraction=4e-4):
    """Make opaque regions that match the original's flat background transparent.

    rembg keeps background it cannot reach from the border, e.g. white inside
    a curled tail. Those regions were baked as white patches and filled in as
    solid geometry. Only components of at least min_fraction of the image are
    cleared, so small highlights in the subject stay.
    """
    rgb = np.asarray(original.convert('RGB')).astype(np.int16)
    result = np.asarray(cutout.convert('RGBA')).copy()
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    background = np.median(border, axis=0)
    # A flat background has nearly all border pixels at one color.
    if (np.abs(border-background).max(axis=1) <= tolerance).mean() < .9:
        return Image.fromarray(result), 0
    matches = (np.abs(rgb-background).max(axis=2) <= tolerance) & (result[:, :, 3] > 128)
    components, count = label(matches)
    sizes = np.bincount(components.ravel())[1:]
    keep = []
    for index in np.flatnonzero(sizes >= min_fraction*matches.size)+1:
        pixels = rgb[components == index]
        # Enclosed background is flat and equal to the border color (std ~1.5);
        # dark or white clothing near that color still has shading (std 5+).
        if np.abs(pixels.mean(axis=0)-background).max() <= 6 and pixels.std(axis=0).max() <= 4:
            keep.append(index)
    holes = np.isin(components, keep)
    # Include the anti-aliased rim that rembg left around each hole.
    holes = binary_dilation(holes, iterations=2) & (np.abs(rgb-background).max(axis=2) <= 3*tolerance)
    result[holes, 3] = 0
    return Image.fromarray(result), int(holes.sum())
