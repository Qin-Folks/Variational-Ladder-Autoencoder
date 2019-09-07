import torch
import numpy as np
import torch.nn.functional as F
from torchvision.utils import save_image


def idx_to_one_hot(idx, dig_to_use):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arg_idx = torch.tensor(list([dig_to_use.index(x) for x in idx]))
    n = len(dig_to_use)
    assert torch.max(arg_idx).item() < n
    if arg_idx.dim() == 0:
        arg_idx = arg_idx.unsqueeze(0)
    if arg_idx.dim() == 1:
        arg_idx = arg_idx.unsqueeze(1)

    onehot = torch.zeros(arg_idx.size(0), n)
    onehot = onehot.to(device)
    arg_idx = arg_idx.to(device)
    onehot.scatter_(1, arg_idx, 1)
    return onehot


def idx_to_multi_hot(idx, num_obj, dig_to_use, loc_sensitive=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = len(dig_to_use)

    multi_hot = None
    arg_idx = []
    for an_idx in idx:
        arg_idx.append(list([dig_to_use.index(x) for x in an_idx]))
    arg_idx = torch.tensor(np.array(arg_idx))

    assert torch.max(arg_idx).item() < n
    for i in range(num_obj):
        an_idx = arg_idx[:, i].unsqueeze(-1)
        onehot = torch.zeros(an_idx.size(0), n)
        onehot = onehot.to(device)
        an_idx = an_idx.to(device)
        onehot.scatter_(1, an_idx, 1)
        if multi_hot is None:
            multi_hot = onehot
        else:
            if loc_sensitive:
                multi_hot = torch.cat((multi_hot, onehot), dim=-1)
            else:
                multi_hot += onehot
    return multi_hot


def stn(image, z_where, out_dims, inverse=False, box_attn_window_color=None):
    """ spatial transformer network used to scale and shift input according to z_where in:
            1/ x -> x_att   -- shapes (H, W) -> (attn_window, attn_window) -- thus inverse = False
            2/ y_att -> y   -- (attn_window, attn_window) -> (H, W) -- thus inverse = True

    inverting the affine transform as follows: A_inv ( A * image ) = image
    A = [R | T] where R is rotation component of angle alpha, T is [tx, ty] translation component
    A_inv rotates by -alpha and translates by [-tx, -ty]

    if x' = R * x + T  -->  x = R_inv * (x' - T) = R_inv * x - R_inv * T

    here, z_where is 3-dim [scale, tx, ty] so inverse transform is [1/scale, -tx/scale, -ty/scale]
    R = [[s, 0],  ->  R_inv = [[1/s, 0],
         [0, s]]               [0, 1/s]]
    """
    orig_img_size = image.shape
    if box_attn_window_color is not None:
        # draw a box around the attention window by overwriting the boundary pixels in the given color channel
        with torch.no_grad():
            box = torch.zeros_like(image.expand(-1, 3, -1, -1))
            c = box_attn_window_color % 3  # write the color bbox in channel c, as model time steps
            box[:, c, :, 0] = 1
            box[:, c, :, -1] = 1
            box[:, c, 0, :] = 1
            box[:, c, -1, :] = 1
            # add box to image and clap at 1 if overlap
            image = torch.clamp(image + box, 0, 1)

    # 1. construct 2x3 affine matrix for each datapoint in the minibatch
    theta = torch.zeros(2, 3).repeat(image.shape[0], 1, 1).to(image.device)
    # set scaling
    theta[:, 0, 0] = theta[:, 1, 1] = z_where[:, 0] if not inverse else 1 / (z_where[:, 0] + 1e-9)
    # set translation
    theta[:, :, -1] = z_where[:, 1:] if not inverse else - z_where[:, 1:] / (z_where[:, 0].view(-1, 1) + 1e-9)
    # 2. construct sampling grid
    grid = F.affine_grid(theta, torch.Size(out_dims))
    # 3. sample image from grid
    return F.grid_sample(image, grid)
