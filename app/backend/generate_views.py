"""Generate missing orthographic views locally in an isolated Wonder3D process."""
import argparse
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap


def generate(folder, seed=12345, steps=30):
    root = Path(os.environ.get('FOURVIEW_AI_ROOT', 'D:/FourViewAI'))
    sys.path.insert(0, str(root / 'vendor/Wonder3D-deps'))
    sys.path.insert(0, str(root / 'vendor/Wonder3D-pipeline'))
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import huggingface_hub
    # The isolated Diffusers 0.19 runtime imports this removed helper, but all
    # weights are loaded from pinned local directories and never call it.
    if not hasattr(huggingface_hub, 'cached_download'):
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from diffusers import AutoencoderKL, DDIMScheduler
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel
    import mvdiffusion.models.transformer_mv2d as attention_module

    def efficient(query, key, value, mask, scale):
        return F.scaled_dot_product_attention(query[:, None], key[:, None], value[:, None],
            attn_mask=mask[:, None] if mask is not None else None, dropout_p=0, scale=scale)[:, 0]

    # Preserve the upstream view/domain rearrangements and replace only the
    # quadratic attention score allocation with Torch's CUDA SDPA kernel.
    attention_module.__dict__['_fourview_attention'] = efficient
    for processor in (attention_module.MVAttnProcessor, attention_module.JointAttnProcessor):
        source = textwrap.dedent(inspect.getsource(processor.__call__))
        if processor is attention_module.MVAttnProcessor:
            source = source.replace('multiview_attention=True\n',
                'multiview_attention=True, sparse_mv_attention=False, mvcd_attention=False\n')
        source = source.replace('attention_probs = attn.get_attention_scores(query, key, attention_mask)\n    hidden_states = torch.bmm(attention_probs, value)',
            'hidden_states = _fourview_attention(query, key, value, attention_mask, attn.scale)')
        namespace = {}
        exec(compile(source, '<fourview-sdpa>', 'exec'), attention_module.__dict__, namespace)
        processor.__call__ = namespace['__call__']

    spec = importlib.util.spec_from_file_location('fourview_wonder_pipeline', root / 'vendor/Wonder3D-pipeline/pipeline.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = root / 'models/Wonder3D'
    options = dict(torch_dtype=torch.float16, local_files_only=True)
    print('Loading local view generator', flush=True)
    unet = UNetMV2DConditionModel.from_pretrained(str(model / 'unet'), **options)
    pipe = module.MVDiffusionImagePipeline(
        vae=AutoencoderKL.from_pretrained(str(model / 'vae'), **options),
        image_encoder=CLIPVisionModelWithProjection.from_pretrained(str(model / 'image_encoder'), **options),
        unet=unet, scheduler=DDIMScheduler.from_pretrained(str(model / 'scheduler'), local_files_only=True),
        feature_extractor=CLIPImageProcessor.from_pretrained(str(model / 'feature_extractor'), local_files_only=True),
        safety_checker=None, requires_safety_checker=False)
    pipe.vae.enable_slicing()
    pipe.to('cuda')
    # Preserve all six views and both domains together, but split CFG's
    # conditional/unconditional groups to stay inside an 8 GB GPU.
    from types import MethodType
    from diffusers.models.unet_2d_condition import UNet2DConditionOutput
    original_forward = pipe.unet.forward
    def bounded_cfg(_self, sample, timestep, encoder_hidden_states=None, class_labels=None, **kwargs):
        if sample.shape[0] != 24:
            return original_forward(sample, timestep, encoder_hidden_states=encoder_hidden_states,
                                    class_labels=class_labels, **kwargs)
        output = [original_forward(sample[i:i+12], timestep,
            encoder_hidden_states=encoder_hidden_states[i:i+12],
            class_labels=class_labels[i:i+12], **kwargs).sample for i in (0, 12)]
        return UNet2DConditionOutput(sample=torch.cat(output))
    pipe.unet.forward = MethodType(bounded_cfg, pipe.unet)
    # Canonical orthographic views, normals then RGB, as in the official runner.
    angles = torch.tensor([0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2, 7*np.pi/4])
    cameras = torch.stack((torch.zeros(6), torch.zeros(6), angles), dim=1)
    pipe.camera_embedding = torch.cat((
        torch.cat((cameras, torch.tensor([[1., 0.]]).repeat(6, 1)), dim=1),
        torch.cat((cameras, torch.tensor([[0., 1.]]).repeat(6, 1)), dim=1)), dim=0).half()
    rgba = Image.open(folder / 'front-cutout.png').convert('RGBA')
    box = rgba.getbbox()
    if box is None:
        raise ValueError('ไม่พบตัวละครในภาพด้านหน้า')
    cropped = rgba.crop(box)
    cropped.thumbnail((204, 204), Image.Resampling.LANCZOS)
    condition = Image.new('RGB', (256, 256), 'white')
    condition.paste(cropped, ((256-cropped.width)//2, (256-cropped.height)//2), cropped)
    condition.save(folder / 'view-generator-input.png')
    print('Generating six consistent views locally', flush=True)
    with torch.inference_mode():
        result = pipe(condition, num_inference_steps=steps, guidance_scale=3.0, eta=1.0,
            generator=torch.Generator(device='cuda').manual_seed(seed), output_type='pil').images
    if len(result) != 12:
        raise RuntimeError('View generator returned an unexpected image count')
    # Wonder3D azimuth +90 is the side with muzzle pointing left in the image.
    for view, index in {'left': 2, 'back': 3, 'right': 4}.items():
        result[6+index].save(folder / f'{view}.png')
    for index in range(6):
        result[index].save(folder / f'generated-normal-{index}.png')
        result[6+index].save(folder / f'generated-view-{index}.png')
    (folder / 'generated-views.json').write_text(json.dumps({
        'engine': 'Wonder3D', 'resolution': 256, 'steps': steps, 'seed': seed, 'guidance_scale': 3.0,
        'left_index': 2, 'back_index': 3, 'right_index': 4}), encoding='utf-8')
    print('Missing views saved', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--steps', type=int, default=30)
    args = parser.parse_args()
    generate(args.folder, args.seed, args.steps)
