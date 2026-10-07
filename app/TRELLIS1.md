# TRELLIS v1 local integration

Select **TRELLIS · สร้างรูปทรงและสีรอบตัวจากภาพ (ทดลอง)** in single-image mode.
The existing Hunyuan engine and unfinished TRELLIS.2 installation remain available separately.

This integration uses the cached TRELLIS image-large checkpoint and DINOv2.
It predicts sparse 3D geometry and Gaussian color, then bakes the color to a UV texture.
It does not apply Hunyuan photo projection, reference symmetry, or jaw warping.
Unseen geometry and colors are predictions and can differ from the input; this is not Tripo3D parity.

Presets: draft 25 steps / 2K, balanced 35 steps / 2K, detail 50 steps / 4K.
More sampling steps and a larger texture do not guarantee better anatomy or correct hidden details.
The original dense shape is saved as `shape.glb`; colored GLB uses a simplified UV mesh.
OBJ ZIP includes MTL and texture; STL includes geometry only.

Runtime is isolated through `runtime/trellis1/deps` and `runtime/trellis1/site-packages`
while reusing the main torch 2.7.1+cu128 installation. Staged loading releases each network
before loading the next to fit the RTX 5060 Laptop 8 GB and 16 GB system RAM.
The worker forces xformers Cutlass attention and uses a PyTorch implementation of stride-one
submanifold convolution because the installed spconv native kernels do not support this GPU.
The convolution was compared against independent dense PyTorch convolutions, including
two batches, shuffled sparse coordinates, and dilation 2.

The vendor pipeline imports rembg lazily; the server also skips its import when an alpha
mask is already supplied. Opaque images still use the existing local background remover.
Inference is offline. No OpenAI API key or gated DINOv3 download is needed for this engine.

Successful trial evidence: `trellis1/trial/front-dragon-seed42/trial.json` and
`trellis1/trial/latest-dragon-seed42/trial.json` (approximately 192 seconds at 25 steps / 2K).
Readiness marker: `runtime/trellis1/verified.json`; worker log and progress are stored
in each output job directory. A readiness marker alone does not certify image fidelity.

Backend files: `trellis1_support.py`, `trellis1_worker.py`, `trellis1_compat.py`.
Weights: snapshot `ab6a010207b4ceab566c5d8927d90a6f69c5f976`.
Vendor: `vendor/trellis1/bundle/code` (Windows fork of Microsoft TRELLIS).
Backups are in `app/backups/trellis1-*`.
