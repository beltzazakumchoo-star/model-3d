"""Isolate xatlas' blocking native UV solver from the HTTP server."""
import sys
from pathlib import Path
import numpy as np
import xatlas

if __name__ == '__main__':
    folder = Path(sys.argv[1])
    with np.load(folder / 'uv-input.npz') as source:
        atlas = xatlas.Atlas()
        atlas.add_mesh(source['vertices'], source['faces'])
    pack = xatlas.PackOptions()
    pack.resolution = 4096
    pack.padding = 8
    pack.bilinear = True
    charts = xatlas.ChartOptions()
    charts.max_iterations = 1
    atlas.generate(chart_options=charts, pack_options=pack)
    mapping, indices, uv = atlas[0]
    np.savez(folder / 'uv-layout.npz', mapping=mapping, indices=indices, uv=uv)
