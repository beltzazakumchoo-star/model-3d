"""Inference-only RTX 50 compatibility, installed within the TRELLIS worker.

Submanifold convolution preserves active coordinates and checkpoint weight layout.
It uses PyTorch kernels in place of spconv binaries missing sm_120 support.
"""
import itertools
import torch

def submanifold_forward(self, x):
    if self.training and torch.is_grad_enabled():
        raise RuntimeError("The trial convolution supports inference only")
    if not self.subm or any(s != 1 for s in self.stride):
        raise NotImplementedError("Only stride-one submanifold convolution is supported")
    kernels = tuple(self.kernel_size)
    dilation = tuple(self.dilation)
    key = ("torch_subm", kernels, dilation, x.indices.data_ptr(), x.indices.shape[0])
    cached = x.indice_dict.get(key)
    coords = x.indices.long()
    if cached is None:
        dims = tuple(int(v) for v in x.spatial_shape)
        def encode(c):
            return ((c[:, 0] * dims[0] + c[:, 1]) * dims[1] + c[:, 2]) * dims[2] + c[:, 3]
        hashes = encode(coords)
        sorted_hashes, order = hashes.sort()
        pairs = []
        for offset in itertools.product(*(range(k) for k in kernels)):
            delta = torch.tensor([(v - k // 2) * d for v, k, d in zip(offset, kernels, dilation)], device=coords.device)
            neighbor = coords.clone()
            neighbor[:, 1:] += delta
            valid = ((neighbor[:, 1:] >= 0) & (neighbor[:, 1:] < torch.tensor(dims, device=coords.device))).all(dim=1)
            wanted = encode(neighbor)
            positions = torch.searchsorted(sorted_hashes, wanted).clamp(max=len(coords) - 1)
            valid &= sorted_hashes[positions] == wanted
            dst = valid.nonzero().flatten()
            src = order[positions[dst]]
            pairs.append((dst.int(), src.int()))
        cached = (x.indices, pairs)
        x.indice_dict[key] = cached
    pairs = cached[1]
    weights = self.weight.reshape(self.out_channels, -1, self.in_channels)
    output = torch.zeros((len(coords), self.out_channels), device=x.features.device, dtype=torch.float32)
    for index, (dst, src) in enumerate(pairs):
        for start in range(0, len(dst), 32768):
            target = dst[start:start + 32768].long()
            source = src[start:start + 32768].long()
            values = x.features[source] @ weights[:, index].T
            output.index_add_(0, target, values.float())
    if self.bias is not None:
        output += self.bias.float()
    return x.replace_feature(output.to(x.features.dtype))

def install():
    import spconv.pytorch as spconv
    import xformers.ops as xops
    original = xops.memory_efficient_attention
    def cutlass_attention(*args, **kwargs):
        kwargs.setdefault("op", xops.MemoryEfficientAttentionCutlassOp)
        return original(*args, **kwargs)
    xops.memory_efficient_attention = cutlass_attention
    spconv.SubMConv3d.forward = submanifold_forward

def verify():
    """Compare against independent dense convolution, including batch boundaries."""
    import spconv.pytorch as spconv
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for kernel, dilation in [(1, 1), (3, 1), (3, 2)]:
        coords = torch.cartesian_prod(torch.arange(2), torch.arange(5), torch.arange(5), torch.arange(5)).int()
        coords = coords[torch.rand(len(coords)) > 0.35]
        coords = coords[torch.randperm(len(coords))].cuda()
        features = torch.randn(len(coords), 8, device="cuda")
        conv = spconv.SubMConv3d(8, 4, kernel, dilation=dilation, bias=True).cuda().eval()
        x = spconv.SparseConvTensor(features, coords, [5, 5, 5], 2)
        with torch.no_grad():
            actual = submanifold_forward(conv, x).features
            dense = torch.zeros(2, 8, 5, 5, 5, device="cuda")
            b, z, y, xx = coords.long().unbind(dim=1)
            dense[b, :, z, y, xx] = features
            weight = conv.weight.permute(0, 4, 1, 2, 3).contiguous()
            expected = torch.nn.functional.conv3d(dense, weight, conv.bias, padding=(kernel // 2) * dilation, dilation=dilation)
            expected = expected[b, :, z, y, xx]
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        print("TORCH_SUBM_VERIFIED", kernel, dilation, flush=True)
