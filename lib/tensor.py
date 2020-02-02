import os
import cv2
import numpy as np
from skimage import transform
import torch
import torch.nn.functional as NF


def parse_batch(batch, device):
    batch['image1'] = batch['image1'].to(device, dtype=torch.float)
    batch['img1_info'] = batch['img1_info'].to(device, dtype=torch.float)
    batch['homo12'] = batch['homo12'].to(device, dtype=torch.float)
    batch['image2'] = batch['image2'].to(device, dtype=torch.float)
    batch['img2_info'] = batch['img2_info'].to(device, dtype=torch.float)
    batch['homo21'] = batch['homo21'].to(device, dtype=torch.float)
    batch['img1_raw'] = batch['img1_raw'].to(device, dtype=torch.float)
    batch['img2_raw'] = batch['img2_raw'].to(device, dtype=torch.float)
    return batch

def parse_output(output):
    output['dense_features1'] = output['dense_features1'].squeeze(0)
    output['scores1'] = output['scores1'].squeeze(0)
    output['dense_features2'] = output['dense_features2'].squeeze(0)
    output['scores2'] = output['scores2'].squeeze(0)
    return output

def parse_batch_test(batch, device):
    batch['image1'] = batch['image1'].to(device, dtype=torch.float)
    batch['homo12'] = batch['homo12'].to(device, dtype=torch.float)
    batch['image2'] = batch['image2'].to(device, dtype=torch.float)
    batch['homo21'] = batch['homo21'].to(device, dtype=torch.float)
    return batch


def L2Norm(input, dim=-1):
    input = input / (torch.norm(input, p=2, dim=dim, keepdim=True) + 1e-6)
    return input

def to_npy(t):
    t = (t * 255).permute(1, 2, 0).cpu().detach().numpy()
    return t

def gen_identity_orient(bs, device):
    orient = torch.FloatTensor([1, 0])
    orient = orient.repeat(bs, 1).reshape(bs, 2).to(device)
    return orient

def sample_patches(img, patch_size):

    c, h, w = img.shape # Tensor format: C, H, W
    assert(h >= patch_size and w >= patch_size)
    # print('img size', c, h, w)

    # Generate patch offset
    step = int(0.75 * patch_size) # 96
    x = [i for i in range(0, w - patch_size, step)]
    y = [i for i in range(0, h - patch_size, step)]
    x.append(w - patch_size) # Add last patch
    y.append(h - patch_size)
    patch_n = len(x) * len(y)

    y_offset, x_offset = torch.meshgrid([
        torch.IntTensor(y), 
        torch.IntTensor(x)
    ])
    x_offset = x_offset.contiguous().view(-1).unsqueeze(1)
    y_offset = y_offset.contiguous().view(-1).unsqueeze(1)

    # Generate patch coordinates
    y_c, x_c = torch.meshgrid([
        torch.linspace(0, patch_size-1, patch_size),
        torch.linspace(0, patch_size-1, patch_size)
    ])
    x_c = x_c.contiguous().view(-1).to(torch.int) # (patch_size * patch_size)
    x_c = x_c.repeat(patch_n).view(patch_n, -1) # (patch_n, patch_size * patch_size)
    y_c = y_c.contiguous().view(-1).to(torch.int)
    y_c = y_c.repeat(patch_n).view(patch_n, -1)

    # print('_c shape', x_c.shape)
    # print('_offset shape', y_offset.shape)

    x = x_offset + x_c # (patch_n, patch_size * patch_size)
    y = y_offset + y_c
    pixel_idx = (y * w + x).view(-1).to(torch.long) # (patch_n * patch_size * patch_size)
    # print('shape', x.shape, y.shape)
    
    img_flat = img.view(c, -1) # (C, H * W)
    patches_0 = img_flat[0].gather(0, pixel_idx).reshape(patch_n, patch_size, patch_size)
    patches_1 = img_flat[1].gather(0, pixel_idx).reshape(patch_n, patch_size, patch_size)
    patches_2 = img_flat[2].gather(0, pixel_idx).reshape(patch_n, patch_size, patch_size)
    patches = torch.stack([patches_0, patches_1, patches_2], dim=0)
    patches = patches.permute(1, 0, 2, 3)
    # print('patches', patches.shape)

    return patches

def sample_patches_orient(img, patch_size, orients, device):

    c, h, w = img.shape # Tensor format: C, H, W
    assert(h >= patch_size and w >= patch_size)
    # print('img size', c, h, w)
    patch_n = orients.shape[0]

    # Generate patch offset
    step = int(0.75 * patch_size) # 96
    patch_size_half = patch_size // 2 # 64
    x = [i + patch_size_half for i in range(0, w - patch_size, step)]
    y = [i + patch_size_half for i in range(0, h - patch_size, step)]
    x.append(w - patch_size_half) # Add last patch
    y.append(h - patch_size_half)
    patch_n = len(x) * len(y)

    y_offset, x_offset = torch.meshgrid([
        torch.FloatTensor(y).to(device=device), 
        torch.FloatTensor(x).to(device=device)
    ])
    x_offset = x_offset.contiguous().view(-1).unsqueeze(1)
    y_offset = y_offset.contiguous().view(-1).unsqueeze(1)

    # Generate patch coordinates
    y_c, x_c = torch.meshgrid([
        torch.linspace(-patch_size_half, patch_size_half-1, patch_size, device=device),
        torch.linspace(-patch_size_half, patch_size_half-1, patch_size, device=device)
    ])
    x_c = x_c.contiguous().view(-1) # (patch_size * patch_size)
    y_c = y_c.contiguous().view(-1) # (patch_size * patch_size)
    one_t = x_c.new_full(x_c.size(), fill_value=1)
    grid = torch.stack((x_c, y_c, one_t))
    grid = grid.view(-1).repeat(patch_n).view(patch_n, 3, -1)

    # Set rotate matrix
    cos = orients[:, 0].unsqueeze(1)
    sin = orients[:, 1].unsqueeze(1)
    zeros = cos.new_full(cos.size(), fill_value=0)
    ones = cos.new_full(cos.size(), fill_value=1)
    rot_mat = torch.cat((cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones), dim=1)
    rot_mat = rot_mat.view(-1, 3, 3)

    # Apply rotation matrix
    # print('rot_mat shape:', rot_mat.shape, 'grid shape:', grid.shape)
    # print('rot_mat', rot_mat)
    # print('grid', grid)
    grid = torch.matmul(rot_mat, grid)
    x = (x_offset + grid[:, 0, :]).view(-1) # (patch_n * patch_size * patch_size)
    y = (y_offset + grid[:, 1, :]).view(-1)

    # Setup interpolation
    x0, y0 = x.floor(), y.floor()
    x1, y1 = x0 + 1, y0 + 1

    x0 = x0.clamp(min=0, max=w-1)
    x1 = x1.clamp(min=0, max=w-1)
    y0 = y0.clamp(min=0, max=h-1)
    y1 = y1.clamp(min=0, max=h-1)

    # Interpolation weights
    # print('interpolation:', x0.shape, x1.shape, x.shape)
    w_a = (x1 - x) * (y1 - y)
    w_b = (x1 - x) * (y - y0)
    w_c = (x - x0) * (y1 - y)
    w_d = (x - x0) * (y - y0)

    x0, x1, y0, y1 = x0.long(), x1.long(), y0.long(), y1.long()

    base_y0 = y0 * w
    base_y1 = y1 * w

    # All idx are in (patch_n * patch_size * patch_size)
    idx_a = base_y0 + x0 # Left-top pixel
    idx_b = base_y1 + x0 # Left-bottom pixel
    idx_c = base_y0 + x1 # Right-top pixel
    idx_d = base_y1 + x1 # Right-bottom pixel

    img_flat = img.view(c, -1) # (C, H * W)
    patches = torch.zeros((c, patch_n * patch_size * patch_size)).to(device=device)
    for i in range(c):
        img_a = img_flat[i].gather(0, idx_a)
        img_b = img_flat[i].gather(0, idx_b)
        img_c = img_flat[i].gather(0, idx_c)
        img_d = img_flat[i].gather(0, idx_d)
        patches[i] = w_a * img_a + w_b * img_b + w_c * img_c + w_d * img_d

    patches = torch.stack([patches[0], patches[1], patches[2]], dim=0)
    patches = patches.reshape(c, patch_n, patch_size, patch_size).permute(1, 0, 2, 3)

    return patches

def show_patches(patches, mean=None, std=None, prefix=''):
    
    out_folder = 'tmp/patches'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    print('Show Patches:', patches.shape, 'Save to', out_folder)
    print('Norm mean:', mean, 'Norm std:', std)
    
    bs, c, h, w = patches.shape
    mean = torch.FloatTensor(mean).reshape(3, 1, 1).to(patches.device)
    std = torch.FloatTensor(std).reshape(3, 1, 1).to(patches.device)

    if prefix != '':
        prefix += '_'

    for i in range(bs):
        img = patches[i] * std + mean
        img = to_npy(img)
        out_path = os.path.join(out_folder, f'{prefix}{i}.png')
        cv2.imwrite(out_path, img)


def warp_img(img, homo):

    img = img.unsqueeze(0)
    b, c, h, w = img.shape

    y_c, x_c = torch.meshgrid([
        torch.arange(h), torch.arange(w)
    ])
    x_c = x_c.contiguous().view(-1) # (patch_size * patch_size)
    y_c = y_c.contiguous().view(-1) # (patch_size * patch_size)
    one_t = x_c.new_full(x_c.size(), fill_value=1)
    grid = torch.stack((x_c, y_c, one_t))
    grid = grid.view(-1).repeat(b).view(b, 3, -1)
    grid = grid.type_as(homo).to(homo.device)

    grid_w = torch.matmul(homo, grid).to(torch.float)
    grid_w = grid_w.permute(0, 2, 1)  # (B, H*W, 3)
    grid_w = grid_w.div(grid_w[:, :, 2].unsqueeze(-1) + 1e-8)  # (B, H*W, 3)
    grid_w = grid_w.view(b, h, w, -1)[:, :, :, :2]  # (B, H, W, 2)
    grid_w[:, :, :, 0] = grid_w[:, :, :, 0].div(w - 1) * 2 - 1
    grid_w[:, :, :, 1] = grid_w[:, :, :, 1].div(h - 1) * 2 - 1

    img_w = NF.grid_sample(img, grid_w)  # (B, C, H, W)

    return img_w.squeeze(0)


def distance_matrix_vector(anchor, positive):
    """
    Given batch of anchor descriptors and positive descriptors calculate distance matrix
    :param anchor: (B, 128)
    :param positive: (B, 128)
    :return:
    """
    eps = 1e-8
    FeatSimi_Mat = 2 - 2 * (anchor.dot(np.transpose(positive)))  # [0, 4]
    FeatSimi_Mat = np.clip(FeatSimi_Mat, eps, 4.0)
    FeatSimi_Mat = np.sqrt(FeatSimi_Mat)  # euc [0, 2]
    return FeatSimi_Mat



def im_rescale(im, output_size):
    h, w = im.shape[:2]
    if isinstance(output_size, int):
        if h > w:
            new_h, new_w = output_size * h / w, output_size
        else:
            new_h, new_w = output_size, output_size * w / h
    else:
        new_h, new_w = output_size
    new_h, new_w = int(new_h), int(new_w)
    img = transform.resize(im, (new_h, new_w), mode='constant')

    return img, h, w, new_w / w, new_h / h
