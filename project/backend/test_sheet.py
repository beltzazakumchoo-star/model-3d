"""Check sheet orientation, validation, persistence, and direct GPU dispatch."""
import asyncio
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException, UploadFile
from PIL import Image
from backend import server
from backend.split_sheet import split_sheet


class SheetTests(unittest.TestCase):
    def sheet(self):
        image = Image.new('RGBA', (301, 201))
        for box, color in [((0, 0, 150, 100), 'red'), ((150, 0, 301, 100), 'green'),
                           ((0, 100, 150, 201), 'blue'), ((150, 100, 301, 201), 'yellow')]:
            image.paste(color, box)
        return image

    def test_split_orientation_and_odd_dimensions(self):
        views = split_sheet(self.sheet())
        for view, color in [('front', 'red'), ('left', 'green'), ('right', 'blue'), ('back', 'yellow')]:
            self.assertEqual(views[view].getpixel((0, 0)), Image.new('RGBA', (1, 1), color).getpixel((0, 0)))
        self.assertEqual(sum(im.width * im.height for im in views.values()), 301 * 201)

    def test_small_sheet_rejected(self):
        with self.assertRaises(ValueError):
            split_sheet(Image.new('RGBA', (127, 200)))

    def test_sheet_submit_without_wonder3d(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = io.BytesIO()
            self.sheet().save(payload, format='PNG')
            payload.seek(0)
            with patch.object(server, 'OUTPUTS', Path(directory)), patch.object(server, 'ROOT', Path(directory)), \
                 patch.object(server, 'jobs', {}), patch.object(server, 'readiness', return_value={'ready': True}), \
                 patch.object(server.worker, 'submit') as dispatch:
                result = asyncio.run(server.submit(front=UploadFile(filename='sheet.png', file=payload),
                    right=None, back=None, left=None, quality='draft', color=True, input_mode='sheet', alignment='{}'))
                folder = Path(directory) / result['id']
                self.assertTrue((folder / 'source.png').is_file())
                for view, expected in split_sheet(self.sheet()).items():
                    with Image.open(folder / f'{view}.png') as saved:
                        self.assertEqual(saved.tobytes(), expected.tobytes())
                self.assertEqual(server.jobs[result['id']]['input_mode'], 'sheet')
                dispatch.assert_called_once_with(server.generate_job, result['id'], 'draft', True)

    def test_single_image_submit_keeps_only_front(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = io.BytesIO()
            Image.new('RGBA', (200, 300), 'red').save(payload, format='PNG')
            payload.seek(0)
            with patch.object(server, 'OUTPUTS', Path(directory)), patch.object(server, 'ROOT', Path(directory)), \
                 patch.object(server, 'jobs', {}), patch.object(server, 'readiness', return_value={'ready': True}), \
                 patch.object(server.worker, 'submit') as dispatch:
                result = asyncio.run(server.submit(front=UploadFile(filename='one.png', file=payload),
                    right=None, back=None, left=None, quality='draft', color=True, input_mode='image', alignment='{}'))
                folder = Path(directory) / result['id']
                self.assertEqual(sorted(file.name for file in folder.glob('*.png')), ['front.png'])
                self.assertEqual(server.jobs[result['id']]['input_mode'], 'image')
                dispatch.assert_called_once_with(server.generate_job, result['id'], 'draft', True)

    def test_single_image_colors_every_vertex(self):
        import numpy as np
        import trimesh
        from backend.aligned_vertex import colorize_single
        mesh = trimesh.creation.icosphere(subdivisions=3)
        image = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
        image.paste((200, 40, 40, 255), (64, 64, 192, 128))
        image.paste((40, 40, 200, 255), (64, 128, 192, 192))
        with tempfile.TemporaryDirectory() as directory:
            colorize_single(mesh, image, Path(directory), {'auto': True, 'views': {}}, lambda stage: None)
            self.assertTrue((Path(directory) / 'color-alignment.json').is_file())
        colors = np.asarray(mesh.visual.vertex_colors)
        vertices = np.asarray(mesh.vertices)
        # Top half is red and bottom half blue on the seen side and behind it.
        for facing in (vertices[:, 2] > .5, vertices[:, 2] < -.5):
            top, bottom = colors[facing & (vertices[:, 1] > .4)], colors[facing & (vertices[:, 1] < -.4)]
            self.assertTrue((top[:, 0] > top[:, 2]).all())
            self.assertTrue((bottom[:, 2] > bottom[:, 0]).all())


if __name__ == '__main__':
    unittest.main()
