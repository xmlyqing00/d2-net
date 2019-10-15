import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.exceptions import EmptyTensorError
from lib.utils import interpolate_dense_features, upscale_positions


def process_multiscale(image, model, scales=None):
    if scales is None:
        scales = [.5, 1, 2]
    b, _, h_init, w_init = image.size()  # b = 1, h_int = 256, w_int = 256
    device = image.device
    assert(b == 1)

    all_keypoints = torch.zeros([3, 0])  # 代表 cyx 坐标, c 代表深度的坐标
    all_descriptors = torch.zeros([  # (512, 0) 512 维度的向量
        model.dense_feature_extraction.num_channels, 0
    ])
    all_scores = torch.zeros(0)  # 关键点的响应

    previous_dense_features = None
    banned = None
    for idx, scale in enumerate(scales):
        current_image = F.interpolate(  # 如果需要将图像缩放到多个尺度, 就是用插值
            image, scale_factor=scale,
            mode='bilinear', align_corners=True
        )
        _, _, h_level, w_level = current_image.size()  # 插值之后的高和宽

        # (1, 512, 63, 63) 提取的特征维度, 在训练时, 256 是被缩小到 32.
        dense_features = model.dense_feature_extraction(current_image)
        del current_image

        _, _, h, w = dense_features.size()  # h = 63, w = 63

        # Sum the feature maps.
        if previous_dense_features is not None:
            # 如果之前预测得到了 feature, 则采样到和当前一样的大小
            # 由于多尺度默认输入是 [0.5, 1., 2.], 所以这里的插值是上采样
            # 如果反过来是 [2., 1., 0.5], 就会变成下采样了

            # 插值采样到一样的大小之后
            # 之前的 feature map 和 现在的 feature map 进行相加得到新的
            # previous_dense_features
            dense_features += F.interpolate(
                previous_dense_features, size=[h, w],
                mode='bilinear', align_corners=True
            )
            del previous_dense_features

        # Recover detections.
        # (1, 512, 63, 63)
        # 如果是多尺度, 通过融合以后的 feature map 进行关键点的预测.
        # 预测得到一个 (1, 512, 63, 63) 的 detection, 值为 0 或 1
        detections = model.detection(dense_features)
        if banned is not None:
            # banned 表示上一次检测到的关键点, 在多尺度下 为了避免重复检测, ban 掉它.
            # 和 feature 一样, 如果之前有 banned 就插值到和本次一样 # (1, 1, h, w)
            banned = F.interpolate(banned.float(), size=[h, w]).byte()
            # 如果上一次有在这一次的某个位置检测到关键点, 就将这个位置的值设为 0
            detections = torch.min(detections.byte(), (1 - banned))
            banned = torch.max(  # 将之前检测的结果和现在检测的结果进行融合
                torch.max(detections, dim=1)[0].unsqueeze(1), banned
            )
        else:
            # (1, 1, 63, 63)
            # 因为 detection 的值为 0 或 1, 所以取最大有 1 的话代表这个点有关键点
            banned = torch.max(detections, dim=1)[0].unsqueeze(1)
        # (3, 752) # 提取关键点的对应整数坐标 cyx
        fmap_pos = torch.nonzero(detections[0]).t().cpu()
        del detections

        # Recover displacements.
        # 输入 dense_features 是 (1, 512, 63, 63)
        # 输出是 (2, 512, 63, 63) 代表的每个点 yx
        # 的坐标偏移
        displacements = model.localization(dense_features)[0]
        # 从 displacements 中取出 fmap_pos y 坐标对应的偏移量
        displacements_i = displacements[
            0, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]
        ]  # (752)
        # 从 displacements 中取出 fmap_pos x 坐标对应的偏移量
        displacements_j = displacements[
            1, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]
        ]  # (752)
        del displacements

        # 将偏移量绝对值小于 0.5 的关键点的 mask 设为 1
        # 只要有一个超过或等于 0.5, 就为 0
        mask = torch.min(
            torch.abs(displacements_i) < 0.5,
            torch.abs(displacements_j) < 0.5
        )  # (752) mask.sum() = 718
        # 只取出偏移量小于 0.5 的关键点整数坐标
        fmap_pos = fmap_pos[:, mask]  # (3, 718)
        valid_displacements = torch.stack([
            displacements_i[mask],
            displacements_j[mask]
        ], dim=0).cpu()  # (2, 718)
        del mask, displacements_i, displacements_j

        # fmap_pos[1 :, :].shape => (2, 718)
        fmap_keypoints = fmap_pos[1:, :].float() + valid_displacements  # (yx)
        del valid_displacements

        try:
            # (根据坐标将特征进行插值得到最终的特征)
            # raw_descriptors => (512, 718), ids => 718
            raw_descriptors, _, ids = interpolate_dense_features(
                fmap_keypoints.to(device),  # (2, 718) - yx
                dense_features[0]  # dense_features[0].shape => (512, 63, 63)
            )
        except EmptyTensorError:
            continue
        fmap_pos = fmap_pos[:, ids]  # (3, 718) 整数坐标
        fmap_keypoints = fmap_keypoints[:, ids]  # (2, 718) 亚像素坐标
        del ids

        keypoints = upscale_positions(
            fmap_keypoints, scaling_steps=2)  # 为了抵消 vgg16 1/4 下采样
        del fmap_keypoints

        descriptors = F.normalize(raw_descriptors, dim=0).cpu()  # L2-norm
        del raw_descriptors

        keypoints[0, :] *= h_init / h_level  # 为了抵消开头的 resize
        keypoints[1, :] *= w_init / w_level  # 为了抵消开头的 resize

        fmap_pos = fmap_pos.cpu()
        keypoints = keypoints.cpu()

        keypoints = torch.cat([  # 变为齐次坐标
            keypoints,
            torch.ones([1, keypoints.size(1)]) * 1 / scale,  # 考虑当前 scale 的影响
        ], dim=0)

        # 根据关键点整数坐标从 feature 中取得对应的 score
        # 再根据下标(idx=0, 1, 2)依次除以 (idx+1 = 1, 2, 3)
        scores = dense_features[0, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]].cpu(
        ) / (idx + 1)
        del fmap_pos

        all_keypoints = torch.cat(
            [all_keypoints, keypoints], dim=1)  # (2, 718)
        all_descriptors = torch.cat(
            [all_descriptors, descriptors], dim=1)  # (512, 718)
        all_scores = torch.cat([all_scores, scores], dim=0)  # (718)
        del keypoints, descriptors

        previous_dense_features = dense_features
        del dense_features
    del previous_dense_features, banned

    keypoints = all_keypoints.t().numpy()  # (718, 3)
    del all_keypoints
    scores = all_scores.numpy()  # (718)
    del all_scores
    descriptors = all_descriptors.t().numpy()  # (718, 512)
    del all_descriptors
    return keypoints, scores, descriptors
