import torch
import sys
sys.path.append('.')
sys.path.append('..')
from grid_encoder import grid_sample

def grid_sample_2d(mat, sample):
    '''
    Implementing grid sampling using pure pytorch code. 
    This allows multi-order gradients via autograd, but is slower.
    '''
    # Nx2
    FD, C, RH, RW = mat.shape
    _, OH, N, _ = sample.shape
    sx = (sample[..., 0].clamp(-1, 1)+ 1) / 2 * (RW - 1)
    sy = (sample[..., 1].clamp(-1, 1)+ 1) / 2 * (RH - 1)
    sample = torch.stack((sx, sy), -1).view(FD, 1, -1, 2)

    tl = sample.floor().long()
    l = tl[..., 0]
    t = tl[..., 1]
    w = sample - tl
    br = sample.ceil().long()
    r = br[..., 0]
    b = br[..., 1]
    wx = w[..., 0]
    wy = w[..., 1]
    tl = t * RW + l
    tr = t * RW + r
    bl = b * RW + l
    br = b * RW + r
    ftl = mat.view(FD, C, -1).gather(2, tl.expand(-1, C, -1))
    ftr = mat.view(FD, C, -1).gather(2, tr.expand(-1, C, -1))
    fbl = mat.view(FD, C, -1).gather(2, bl.expand(-1, C, -1))
    fbr = mat.view(FD, C, -1).gather(2, br.expand(-1, C, -1))
    # compute corners
    ft = ftl * (1 - wx) + ftr * wx
    fb = fbl * (1 - wx) + fbr * wx

    # FD x C
    return (ft * (1 - wy) + fb * wy).view(FD, -1, OH, N)


# -- setting up test cases
B = 10
C = 16
IH = 150
IW = 10
OH = 100
OW = 17

features = torch.randn(B, C, IH, IW).cuda()
grid = torch.randn(B, OH, OW, 2).cuda() * 2 - 1
grid = grid.clamp(-1, 1)
# grid[..., 0] = 1
# grid[..., 1] = 1 

features.requires_grad=True
grid.requires_grad=True


# sampling outputs
f_torch = grid_sample_2d(features, grid)
_f_torch = torch.nn.functional.grid_sample(features, grid, mode='bilinear', align_corners=True)
f_custom = grid_sample(features, grid)

# print the differences
print((f_torch - f_custom).abs().max())
print((_f_torch - f_custom).abs().max())

grad_out = torch.randn_like(f_torch)
grad_out.requires_grad = True

# first order gradients
d_f__d_feat_torch = torch.autograd.grad(f_torch, features, grad_out, retain_graph=True, create_graph=True, allow_unused=True)[0]
d_f__d_grid_torch = torch.autograd.grad(f_torch, grid, grad_out, retain_graph=True, create_graph=True, allow_unused=True)[0]
d_f__d_feat_torch_ = torch.autograd.grad(_f_torch, features, grad_out, retain_graph=True, create_graph=True, allow_unused=True)[0]
d_f__d_grid_torch_ = torch.autograd.grad(_f_torch, grid, grad_out, retain_graph=True, create_graph=True, allow_unused=True)[0]

d_f__d_feat_custom = torch.autograd.grad(f_custom, features, grad_out, retain_graph=True, create_graph=True, allow_unused=True)[0]
d_f__d_grid_custom = torch.autograd.grad(f_custom, grid, grad_out, retain_graph=True, create_graph=True, allow_unused=True)[0]

# print the differences
print('first order')
print((d_f__d_feat_torch - d_f__d_feat_custom).abs().max())
print((d_f__d_grid_torch - d_f__d_grid_custom).abs().max())
# print('first order')
# print((d_f__d_feat_torch_ - d_f__d_feat_custom).abs().max())
# print((d_f__d_grid_torch_ - d_f__d_grid_custom).abs().max())
# print((d_f__d_feat_torch_ - d_f__d_feat_torch).abs().max())
# print((d_f__d_grid_torch_ - d_f__d_grid_torch).abs().max())

# second order gradients (only supporting 2nd order gradient from d_out/grids as of now)
grad_d_f__d_grid = torch.randn_like(d_f__d_grid_custom)

# second order of (df/d_grid) / d_(grad_out)
d_f__d_grid____d_grad_out_torch = torch.autograd.grad(d_f__d_grid_torch, grad_out, grad_d_f__d_grid, retain_graph=True, create_graph=True, allow_unused=True)[0]
d_f__d_grid____d_grad_out_custom = torch.autograd.grad(d_f__d_grid_custom, grad_out, grad_d_f__d_grid, retain_graph=True, create_graph=True, allow_unused=True)[0]

# second order of (df/d_grid) / d_features
d_f__d_grid____d_feat_torch = torch.autograd.grad(d_f__d_grid_torch, features, grad_d_f__d_grid, retain_graph=True, create_graph=True, allow_unused=True)[0]
d_f__d_grid____d_feat_custom = torch.autograd.grad(d_f__d_grid_custom, features, grad_d_f__d_grid, retain_graph=True, create_graph=True, allow_unused=True)[0]

# print the differences
print((d_f__d_grid____d_grad_out_torch - d_f__d_grid____d_grad_out_custom).abs().max())
print((d_f__d_grid____d_feat_torch - d_f__d_grid____d_feat_custom).abs().max())
print()