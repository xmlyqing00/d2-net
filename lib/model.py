import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


class DenseFeatureExtractionModule(nn.Module):
    def __init__(self, finetune_feature_extraction=False, use_cuda=True):
        super(DenseFeatureExtractionModule, self).__init__()

        model = models.vgg16()
        vgg16_layers = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2',
            'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2',
            'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3',
            'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3',
            'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
            'pool5'
        ]
        conv4_3_idx = vgg16_layers.index('conv4_3')

        self.model = nn.Sequential(
            *list(model.features.children())[: conv4_3_idx + 1]
        )

        self.num_channels = 512

        # Fix forward parameters
        for param in self.model.parameters():
            param.requires_grad = False
        if finetune_feature_extraction:
            # Unlock conv4_3
            for param in list(self.model.parameters())[-2 :]:
                param.requires_grad = True

        if use_cuda:
            # noinspection PyArgumentList
            self.model = self.model.cuda()

    def forward(self, batch):
        output = self.model(batch)
        return output


class SoftDetectionModule(nn.Module):
    def __init__(self, soft_local_max_size=3):
        super(SoftDetectionModule, self).__init__()

        self.soft_local_max_size = soft_local_max_size

        self.pad = self.soft_local_max_size // 2

    def forward(self, batch):
        b = batch.size(0)

        batch = F.relu(batch) # (2, 512, 32, 32)

        max_per_sample = torch.max(batch.view(b, -1), dim=1)[0] # 找到两幅图像预测得到的 score 的最大值, 在 HWD 下的最大值.
        exp = torch.exp(batch / max_per_sample.view(b, 1, 1, 1)) # (2, 512, 32, 32) # 值域在[1,e]之间
        sum_exp = ( # 用 silding window 滑过 exp 进行求和
            self.soft_local_max_size ** 2 *
            F.avg_pool2d(
                F.pad(exp, [self.pad] * 4, mode='constant', value=1.),
                self.soft_local_max_size, stride=1
            )
        ) # pad 补 1 是因为 exp 的取值范围在 [1,e] 之间 # (2, 512, 32, 32)
        local_max_score = exp / sum_exp # (2, 512, 32, 32) # 值域在 [0, 1) 之间 表示该点在临近9个像素内的score权重

        depth_wise_max = torch.max(batch, dim=1, keepdim=True)[0] # (2, 1, 32, 32) # 在深度上的最大值
        depth_wise_max_score = batch / depth_wise_max # (2, 512, 32, 32) # 再求每个score与自己所在深度最大值的比值

        all_scores = local_max_score * depth_wise_max_score # (2, 512, 32, 32) # 将 临近9像素比值和深度比值进行相乘,
        score = torch.max(all_scores, dim=1)[0] # (2, 32, 32) # 再在找到深度上找到最大值

        score = score / torch.sum(score.view(b, -1), dim=1).view(b, 1, 1) # (2, 32, 32) # 再求与自身所有score求和的比值, 得到最终响应值

        return score


class D2Net(nn.Module):
    def __init__(self, model_file=None, use_cuda=True):
        super(D2Net, self).__init__()

        self.dense_feature_extraction = DenseFeatureExtractionModule(
            finetune_feature_extraction=True,
            use_cuda=use_cuda
        )

        self.detection = SoftDetectionModule()

        if model_file is not None:
            self.load_state_dict(torch.load(model_file)['model'])

    def forward(self, batch):
        b = batch['image1'].size(0)

        dense_features = self.dense_feature_extraction(
            torch.cat([batch['image1'], batch['image2']], dim=0)
        )# dense_features is (2, 512, 32, 32) and batch['image1'] is (1, 3, 256, 256)

        scores = self.detection(dense_features) # (2, 32, 32)

        dense_features1 = dense_features[: b, :, :, :] # (1, 512, 32, 32)
        dense_features2 = dense_features[b :, :, :, :] # (1, 512, 32, 32)

        scores1 = scores[: b, :, :] # (1, 32, 32)
        scores2 = scores[b :, :, :] # (1, 32, 32)

        return {
            'dense_features1': dense_features1, # (1, 512, 32, 32)
            'scores1': scores1, # (1, 32, 32)
            'dense_features2': dense_features2, # (1, 512, 32, 32)
            'scores2': scores2 # (1, 32, 32)
        }
