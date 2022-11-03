# Pytorch Grid Sample 2D with second-order gradients
Simple CUDA implementation of torch.nn.functional.grid_sample with 2D inputs.

## Usage
For comparison against autograd based grid_sampler, @see tests/test_grid.py

```python
# copy the grid_encoder folder to appropriate directory
import torch
from .grid_encoder import grid_sample
B = 10
C = 3
IH = 100
IW = 150
OH = 13
OW = 17

features = torch.randn(B, C, IH, IW).cuda()
grid = torch.rand(B, OH, OW, 2).cuda() * 2 - 1

features.requires_grad=True
grid.requires_grad=True

f_torch = torch.nn.functional.grid_sample(features, grid, mode='bilinear', align_corners=True)
f_custom = grid_sample(features, grid)

grad_out = torch.randn_like(f_torch)
grad_out.requires_grad = True

# first order gradients
d_f__d_feat_torch = torch.autograd.grad(f_torch, features, grad_out, retain_graph=True, create_graph=True, allow_unused=True)[0]
d_f__d_grid_torch = torch.autograd.grad(f_torch, grid, grad_out, retain_graph=True, create_graph=True, allow_unused=True)[0]

d_f__d_feat_custom = torch.autograd.grad(f_custom, features, grad_out, retain_graph=True, create_graph=True, allow_unused=True)[0]
d_f__d_grid_custom = torch.autograd.grad(f_custom, grid, grad_out, retain_graph=True, create_graph=True, allow_unused=True)[0]

print((d_f__d_feat_torch - d_f__d_feat_custom).abs().max())
print((d_f__d_grid_torch - d_f__d_grid_custom).abs().max())

# second order gradients (only supporting 2nd order gradient from d_out/grids as of now)
grad_d_f__d_grid_custom  = torch.ones_like(d_f__d_grid_custom)

# second order of (df/d_grid) / d_(grad_out)
d_f__d_grid____d_grad_out = torch.autograd.grad(d_f__d_grid_custom, grad_out, grad_d_f__d_grid_custom, retain_graph=True, create_graph=True, allow_unused=True)[0]

# second order of (df/d_grid) / d_features
d_f__d_grid____d_feat = torch.autograd.grad(d_f__d_grid_custom, features, grad_d_f__d_grid_custom, retain_graph=True, create_graph=True, allow_unused=True)[0]
```

## Limitations
- This function only works for 'bilinear' mode with 'aligned corners'
- Only works for 'CUDA' enabled devices
- Only computes second order gradients from d_out/grids with respect to:
  - first order gradient: (d_out/ grids) / d_(grad_out)
  - features: (d_out / grids) / d_features
