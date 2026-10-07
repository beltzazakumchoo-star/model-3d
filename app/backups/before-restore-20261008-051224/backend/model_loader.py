"""Load checkpoint tensors without a second full copy in RAM."""
from pathlib import Path


def load_multiview(model_directory: Path, subfolder='hunyuan3d-dit-v2-mv'):
    import torch
    import yaml
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors.torch import load_file
    from hy3dgen.shapegen.pipelines import (
        Hunyuan3DDiTFlowMatchingPipeline, instantiate_from_config,
    )

    folder = model_directory / subfolder
    config = yaml.safe_load((folder / 'config.yaml').read_text(encoding='utf-8'))
    tensors = load_file(str(folder / 'model.fp16.safetensors'), device='cpu')
    components = {}
    for section in ('model', 'vae', 'conditioner'):
        # Parameters start on the meta device, while non-persistent buffers
        # remain initialized on CPU. assign=True then reuses mapped weights.
        with init_empty_weights(include_buffers=False):
            module = instantiate_from_config(config[section])
        prefix = section + '.'
        weights = {key[len(prefix):]: value for key, value in tensors.items() if key.startswith(prefix)}
        if not weights:
            raise RuntimeError('Checkpoint is missing component: ' + section)
        incompatible = module.load_state_dict(weights, strict=False, assign=True)
        for key in incompatible.missing_keys:
            if section != 'vae' or not key.startswith(('encoder.', 'pre_kl.')):
                raise RuntimeError('Missing checkpoint tensor: ' + section + '.' + key)
        # Image-to-shape only decodes latents. Do not allocate or upload the
        # training-only point-cloud encoder omitted from this checkpoint.
        if section == 'vae':
            module.encoder = torch.nn.Identity()
            module.pre_kl = torch.nn.Identity()
        for name, parameter in module.named_parameters():
            if parameter.is_meta:
                set_module_tensor_to_device(module, name, 'cpu',
                    value=torch.zeros(parameter.shape, dtype=torch.float16))
        components[section] = module.eval()
    engine = Hunyuan3DDiTFlowMatchingPipeline(
        **components,
        scheduler=instantiate_from_config(config['scheduler']),
        image_processor=instantiate_from_config(config['image_processor']),
        device='cpu', dtype=torch.float16,
    )
    engine.vae.enable_flashvdm_decoder(enabled=True, adaptive_kv_selection=False, mc_algo='mc')
    # Compute the two classifier-free guidance predictions sequentially.
    # This preserves guidance while reducing activation peaks on 8 GB GPUs.
    from types import MethodType
    original_forward = engine.model.forward

    def sequential_forward(_module, x, timestep, contexts, **kwargs):
        batch = x.shape[0]
        if batch <= 1:
            return original_forward(x, timestep, contexts, **kwargs)

        def take(value, index):
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == batch:
                return value[index:index+1]
            if isinstance(value, dict):
                return {key: take(item, index) for key, item in value.items()}
            return value

        return torch.cat([original_forward(x[index:index+1], take(timestep, index),
            take(contexts, index), **take(kwargs, index)) for index in range(batch)], dim=0)

    engine.model.forward = MethodType(sequential_forward, engine.model)
    # This upstream pipeline is not a DiffusionPipeline, but its offload
    # methods expect the same component registry and exclusion list.
    engine.components = components
    engine._exclude_from_cpu_offload = []
    engine.enable_model_cpu_offload(device='cuda')
    engine.device = torch.device('cuda:0')
    return engine
