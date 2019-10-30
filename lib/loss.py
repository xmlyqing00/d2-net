import matplotlib
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn.functional as F

from lib.utils import (
    grid_positions,
    upscale_positions,
    downscale_positions,
    savefig,
    imshow_image
)
from lib.exceptions import NoGradientError, EmptyTensorError

matplotlib.use('Agg')


def loss_function(
        model,
        batch,
        device,
        margin=1,
        safe_radius=4,
        scaling_steps=3,
        plot=False):
    output = model({  # ['dense_features1', 'scores1', 'dense_features2', 'scores2']
        'image1': batch['image1'].to(device),
        'image2': batch['image2'].to(device)
    })

    loss = torch.tensor(np.array([0], dtype=np.float32), device=device)
    has_grad = False

    n_valid_samples = 0
    for idx_in_batch in range(batch['image1'].size(0)):
        # Annotations
        depth1 = batch['depth1'][idx_in_batch].to(
            device)  # [h1, w1] (256, 256)
        intrinsics1 = batch['intrinsics1'][idx_in_batch].to(device)  # [3, 3]
        pose1 = batch['pose1'][idx_in_batch].view(
            4, 4).to(device)  # [4, 4] extrinsics
        bbox1 = batch['bbox1'][idx_in_batch].to(device)  # [2] top_left_corner

        depth2 = batch['depth2'][idx_in_batch].to(device)
        intrinsics2 = batch['intrinsics2'][idx_in_batch].to(device)
        pose2 = batch['pose2'][idx_in_batch].view(4, 4).to(device)
        bbox2 = batch['bbox2'][idx_in_batch].to(device)

        # Network output
        # (512, 32, 32)
        dense_features1 = output['dense_features1'][idx_in_batch]
        c, h1, w1 = dense_features1.size()  # c=512, h1=32, w1=32 # TODO rename c to c1
        scores1 = output['scores1'][idx_in_batch].view(-1)  # 32*32=1024

        dense_features2 = output['dense_features2'][idx_in_batch]
        _, h2, w2 = dense_features2.size()  # TODO assert c2 == c1
        scores2 = output['scores2'][idx_in_batch]

        all_descriptors1 = F.normalize(
            dense_features1.view(
                c, -1), dim=0)  # (512, 32*32=1024)
        descriptors1 = all_descriptors1

        all_descriptors2 = F.normalize(dense_features2.view(c, -1), dim=0)

        # Warp the positions from image 1 to image 2
        # 制作 32*32 大小的棋盘 # (2, 1024) 坐标从 [0, 0] 到 [31, 31]
        fmap_pos1 = grid_positions(h1, w1, device)
        # 上采样棋盘坐标, 因为 VGG 提特征将 (256, 256) 的图像缩小到了 (32, 32) # 上采样会损失很多定位精度 #
        # shape 是 [2, 1024], 将 32*32 个 xy 坐标上采样, 变为 [3.5, 3.5] 到 [251.5, 251.5]
        pos1 = upscale_positions(fmap_pos1, scaling_steps=scaling_steps)
        try:  # 将图像1的点变换到图像2上, 但仅仅保留那些比较好的点, 用下标 ids 记录 # '好' 的定义是 1024 个点在变换前后对应的深度图的深度大于0, 并且没有越界, 并且估计和采样的深度误差在 0.05 范围内的点
            pos1, pos2, ids = warp(  # 数量从 1024 降到了 173
                pos1,
                depth1, intrinsics1, pose1, bbox1,
                depth2, intrinsics2, pose2, bbox2
            )
        except EmptyTensorError:
            continue
        fmap_pos1 = fmap_pos1[:, ids]  # 找到对应的 采样前棋盘坐标 fmap_pos1
        descriptors1 = descriptors1[:, ids]  # 找到对应的特征
        scores1 = scores1[ids]  # 找到对应的响应值

        # Skip the pair if not enough GT correspondences are available
        if ids.size(0) < 128:
            continue

        # Descriptors at the corresponding positions
        fmap_pos2 = torch.round(  # 将从 pos1 变换过来的 pos2 再降采样到 (32, 32) 的尺度, 并取最近整数
            downscale_positions(pos2, scaling_steps=scaling_steps)
        ).long()
        descriptors2 = F.normalize(  # 取 fmap_pos2 对应的 特征
            dense_features2[:, fmap_pos2[0, :], fmap_pos2[1, :]],
            dim=0
        )
        positive_distance = 2 - 2 * (  # 计算配对 descriptor 的相似度, 值域是 [0, 2], 值越小代表越相似
            descriptors1.t().unsqueeze(1) @ descriptors2.t().unsqueeze(2)
        ).squeeze()  # (173, 1, 512) @ (173, 512, 1) => (173, 1, 1) => squeeze => (173)

        all_fmap_pos2 = grid_positions(h2, w2, device)  # 制作 32*32 大小的棋盘
        position_distance = torch.max(
            torch.abs(
                fmap_pos2.unsqueeze(2).float() -
                all_fmap_pos2.unsqueeze(1)
            ),
            dim=0
        )[0]  # (173, 1024) # 计算 fmap_pos2 到 棋盘每个点的距离(取像素 x 坐标距离和像素 y 坐标距离较大的值)
        # safe_radius=4 如果超出 4 个像素就视为不是该点近邻的点
        is_out_of_safe_radius = position_distance > safe_radius
        # (173, 1024) # 计算图像1点和所有图2点的特征距离
        distance_matrix = 2 - 2 * (descriptors1.t() @ all_descriptors2)
        negative_distance2 = torch.min(  # 剔除掉近邻的点, 也就是在4个像素距离内(包含4个像素)的点, 然后找 descriptor 1 的 hardest negative sample
            distance_matrix + (1 - is_out_of_safe_radius.float()) * 10.,
            dim=1
        )[0]  # (173) 找到 hardest negative sample
        # 将刚才针对 图像1 的计算再在 图像2 上进行一遍来找 descriptor 2 的 hardest negative sample
        all_fmap_pos1 = grid_positions(h1, w1, device)
        position_distance = torch.max(
            torch.abs(
                fmap_pos1.unsqueeze(2).float() -
                all_fmap_pos1.unsqueeze(1)
            ),
            dim=0
        )[0]
        is_out_of_safe_radius = position_distance > safe_radius
        distance_matrix = 2 - 2 * (descriptors2.t() @ all_descriptors1)
        negative_distance1 = torch.min(
            distance_matrix + (1 - is_out_of_safe_radius.float()) * 10.,
            dim=1
        )[0]
        # hard loss
        diff = positive_distance - torch.min(
            negative_distance1, negative_distance2
        )

        scores2 = scores2[fmap_pos2[0, :],
                          fmap_pos2[1, :]]  # (173) # 取出 score 2

        loss = loss + (  # 这里用 F.relu 取代了 hard_loss 中的 max() 函数
            torch.sum(scores1 * scores2 * F.relu(margin + diff)) /
            torch.sum(scores1 * scores2)
        )  # 这里希望 F.relu(margin + diff) 越小的话 scores1 * scores2 越大越好

        has_grad = True  # 如果能运行到这里, 那么代表这个 训练样本 经受了考验
        n_valid_samples += 1

        if plot and batch['batch_idx'] % batch['log_interval'] == 0:
            pos1_aux = pos1.cpu().numpy()
            pos2_aux = pos2.cpu().numpy()
            k = pos1_aux.shape[1]
            col = np.random.rand(k, 3)
            n_sp = 4
            plt.figure()
            plt.subplot(1, n_sp, 1)
            im1 = imshow_image(
                batch['image1'][idx_in_batch].cpu().numpy(),
                preprocessing=batch['preprocessing']
            )
            plt.imshow(im1)
            plt.scatter(
                pos1_aux[1, :], pos1_aux[0, :],
                s=0.25**2, c=col, marker=',', alpha=0.5
            )
            plt.axis('off')
            
            plt.subplot(1, n_sp, 2)
            plt.imshow(
                output['scores1'][idx_in_batch].data.cpu().numpy(),
                cmap='Reds'
            )
            plt.axis('off')
            
            plt.subplot(1, n_sp, 3)
            im2 = imshow_image(
                batch['image2'][idx_in_batch].cpu().numpy(),
                preprocessing=batch['preprocessing']
            )
            plt.imshow(im2)
            plt.scatter(
                pos2_aux[1, :], pos2_aux[0, :],
                s=0.25**2, c=col, marker=',', alpha=0.5
            )
            plt.axis('off')
            
            plt.subplot(1, n_sp, 4)
            plt.imshow(
                output['scores2'][idx_in_batch].data.cpu().numpy(),
                cmap='Reds'
            )
            plt.axis('off')
            
            savefig('train_vis/%s.%02d.%02d.%d.png' % (
                'train' if batch['train'] else 'valid',
                batch['epoch_idx'],
                batch['batch_idx'] // batch['log_interval'],
                idx_in_batch
            ), dpi=300)
            plt.close()

    if not has_grad:
        raise NoGradientError

    loss = loss / n_valid_samples

    return loss


def interpolate_depth(pos, depth):
    device = pos.device

    ids = torch.arange(0, pos.size(1), device=device)  # 32*32=1024

    h, w = depth.size()  # h = w = 256

    i = pos[0, :]  # y coordinator
    j = pos[1, :]  # x coordinator

    # Valid corners
    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

    valid_corners = torch.min(
        torch.min(valid_top_left, valid_top_right),
        torch.min(valid_bottom_left, valid_bottom_right)
    )

    i_top_left = i_top_left[valid_corners]
    j_top_left = j_top_left[valid_corners]

    i_top_right = i_top_right[valid_corners]
    j_top_right = j_top_right[valid_corners]

    i_bottom_left = i_bottom_left[valid_corners]
    j_bottom_left = j_bottom_left[valid_corners]

    i_bottom_right = i_bottom_right[valid_corners]
    j_bottom_right = j_bottom_right[valid_corners]

    ids = ids[valid_corners]
    if ids.size(0) == 0:
        raise EmptyTensorError

    # Valid depth
    valid_depth = torch.min(
        torch.min(
            depth[i_top_left, j_top_left] > 0,
            depth[i_top_right, j_top_right] > 0
        ),
        torch.min(
            depth[i_bottom_left, j_bottom_left] > 0,
            depth[i_bottom_right, j_bottom_right] > 0
        )
    )

    i_top_left = i_top_left[valid_depth]
    j_top_left = j_top_left[valid_depth]

    i_top_right = i_top_right[valid_depth]
    j_top_right = j_top_right[valid_depth]

    i_bottom_left = i_bottom_left[valid_depth]
    j_bottom_left = j_bottom_left[valid_depth]

    i_bottom_right = i_bottom_right[valid_depth]
    j_bottom_right = j_bottom_right[valid_depth]

    ids = ids[valid_depth]
    if ids.size(0) == 0:
        raise EmptyTensorError

    # Interpolation
    i = i[ids]  # y
    j = j[ids]  # x
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    interpolated_depth = (
        w_top_left * depth[i_top_left, j_top_left] +
        w_top_right * depth[i_top_right, j_top_right] +
        w_bottom_left * depth[i_bottom_left, j_bottom_left] +
        w_bottom_right * depth[i_bottom_right, j_bottom_right]
    )

    pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

    return [interpolated_depth, pos, ids]


def uv_to_pos(uv):
    return torch.cat([uv[1, :].view(1, -1), uv[0, :].view(1, -1)], dim=0)


def warp(
        pos1,
        depth1, intrinsics1, pose1, bbox1,
        depth2, intrinsics2, pose2, bbox2
):
    device = pos1.device

    # 根据上采样以后的 (32, 32) pos1(position) 从 (256, 256) 的深度图中采样(插值)深度
    # Z1:采样后的(32, 32)深度图
    # pos1: (32, 32)个坐标中有效的坐标,有效的定义是深度要大于0并且坐标没有越界
    # id1:有效的pos1的下标
    Z1, pos1, ids = interpolate_depth(pos1, depth1)

    # COLMAP convention # pos1(position)加入 bbox1(原图裁剪的左上角坐标) 的偏移 再加个 0.5
    # 应该是为了取整
    u1 = pos1[1, :] + bbox1[1] + .5  # 像素横坐标
    v1 = pos1[0, :] + bbox1[0] + .5  # 像素纵坐标

    X1 = (u1 - intrinsics1[0, 2]) * (Z1 / intrinsics1[0, 0])
    Y1 = (v1 - intrinsics1[1, 2]) * (Z1 / intrinsics1[1, 1])

    XYZ1_hom = torch.cat([
        X1.view(1, -1),
        Y1.view(1, -1),
        Z1.view(1, -1),
        torch.ones(1, Z1.size(0), device=device)
    ], dim=0)  # 相机1齐次坐标
    XYZ2_hom = torch.chain_matmul(
        pose2, torch.inverse(pose1), XYZ1_hom)  # 相机2齐次坐标
    XYZ2 = XYZ2_hom[: -1, :] / XYZ2_hom[-1, :].view(1, -1)  # 相机2归一化坐标

    uv2_hom = torch.matmul(intrinsics2, XYZ2)  # 图2像素齐次坐标
    uv2 = uv2_hom[: -1, :] / uv2_hom[-1, :].view(1, -1)  # 图2像素坐标
    # 加入 bbox2 在图像2 裁剪的偏移, 将 uv2 转换到裁剪后 (256, 256) 的图像上的坐标
    u2 = uv2[0, :] - bbox2[1] - .5
    v2 = uv2[1, :] - bbox2[0] - .5
    uv2 = torch.cat([u2.view(1, -1), v2.view(1, -1)], dim=0)

    # uv_to_pos 是为了交换 array 的 xy, uv 是 (x, y), 而 pos 是 (y, x)
    annotated_depth, pos2, new_ids = interpolate_depth(uv_to_pos(uv2), depth2)
    # 再从 图像2 的深度图中筛选一遍 深度大于0 并且没有越界的像素点
    ids = ids[new_ids]
    pos1 = pos1[:, new_ids]
    estimated_depth = XYZ2[2, new_ids]
    # 再筛选一遍估计和采样的深度误差在 0.05 范围内的点
    inlier_mask = torch.abs(estimated_depth - annotated_depth) < 0.05

    ids = ids[inlier_mask]
    if ids.size(0) == 0:
        raise EmptyTensorError

    pos2 = pos2[:, inlier_mask]
    pos1 = pos1[:, inlier_mask]

    return pos1, pos2, ids
