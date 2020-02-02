import sys
import os
import torch
import argparse
import json
import configparser
import numpy as np
import cv2
from tqdm import tqdm

import torch.utils
from torch.utils.data import DataLoader, sampler
from torchvision import transforms

from lib.model_test import D2Net
from lib.utils import count_parameters_in_MB, AvgrageMeter
from lib.system import gct
from lib.loss import D2Loss_hpatches
from lib.pyramid import process_multiscale
from lib.dataset_hpatches import (
    HPatchesDataset,
)
from lib.tensor import parse_batch_test, distance_matrix_vector

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--cpu', action='store_true', help='Only use cpu.')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--report_freq', type=float, default=1, help='report frequency')
parser.add_argument('--model_path', type=str, default='EXP/model.pt', help='path of pretrained model')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--dataset', default='view', type=str, help='Optional dataset: [view, view_e, view_m, view_h]')
args = parser.parse_args()


def main():
    if not torch.cuda.is_available():
        print(gct(), 'no gpu device available')
        sys.exit(1)

    print(gct(), 'Args = %s', args)

    cfg = configparser.ConfigParser()
    cfg.read('settings.conf')

    if args.cpu:
        device = torch.device('cpu')
        use_cuda = False
    else:
        device = torch.device('cuda')
        use_cuda = True

    criterion = D2Loss_hpatches(scaling_steps=3)
    criterion = criterion.to(device)

    model = D2Net(model_file=args.model_path, use_cuda=use_cuda).to(device)

    print(gct(), 'Param size = %fMB', count_parameters_in_MB(model))

    if args.dataset[:4] == 'view':
        csv_file = cfg['hpatches']['view_csv']
        root_dir = cfg['hpatches'][f'{args.dataset}_root']

    dataset = HPatchesDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=json.loads(cfg['hpatches']['view_mean']),
                std=json.loads(cfg['hpatches']['view_std']))
        ]),
        test_flag=True
    )

    print(gct(), 'Load test dataset.')
    print(gct(), f'Root dir: {root_dir}. #Image pair: {len(dataset)}')

    dataset_len = len(dataset)
    split_idx = list(range(dataset_len))
    part_pos = [
        int(dataset_len * float(cfg['hpatches']['train_part_pos'])),
        int(dataset_len * float(cfg['hpatches']['eval_part_pos']))]
    test_queue = DataLoader(
        dataset,
        batch_size=int(cfg['params']['train_bs']),
        sampler=sampler.SubsetRandomSampler(split_idx[part_pos[1]:]),  # split_idx[part_pos[1]:]
        shuffle=False,
        pin_memory=True, num_workers=2
    )

    result_folder = 'results/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    test_acc = infer(test_queue, model, criterion, result_folder, device)
    print('Test loss', test_acc)


def nearest_neighbor_distance_ratio_match(des1, des2, kp2, threshold):
    des_dist_matrix = distance_matrix_vector(des1, des2)
    # sorted, indices = des_dist_matrix.sort(axis=-1)
    indices = np.argsort(des_dist_matrix, axis=-1)
    sorted = np.take_along_axis(des_dist_matrix, indices, axis=-1)
    Da, Db, Ia = sorted[:, 0], sorted[:, 1], indices[:, 0]
    DistRatio = Da / Db
    predict_label = DistRatio < threshold
    # nn_kp2 = kp2.index_select(dim=0, index=Ia.view(-1))
    nn_kp2 = np.take(kp2, indices=Ia, axis=0)
    return predict_label, nn_kp2


def partition_arg_topK(matrix, K, axis=0):
    """
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]


def arg_topK(arr, topk):
    indices = np.argsort(arr)
    return indices[:topk]


def infer(test_queue, model, criterion, result_folder, device, topk=512):
    avg_loss = AvgrageMeter()
    model.eval()

    for step, valid_sample in enumerate(tqdm(test_queue)):

        with torch.no_grad():
            valid_sample = parse_batch_test(valid_sample, device)
            kp1, score1, des1 = process_multiscale(
                valid_sample['image1'], model, scales=[1]
            )

            kp2, score2, des2 = process_multiscale(
                valid_sample['image2'], model, scales=[1]
            )

            if len(score1) > topk:
                ids = arg_topK(score1, topk)
                kp1 = kp1[ids, :]
                des1 = des1[ids, :]
                #
                ids = arg_topK(score2, topk)
                kp2 = kp2[ids, :]
                des2 = des2[ids, :]

            # output = {
            #     'f1': des1.squeeze(0),
            #     'f2': des2.squeeze(0),
            #     's1': score1,
            #     's2': score2
            # }
            #
            # loss = criterion(valid_sample, output)

            predict_label, nn_kp2 = nearest_neighbor_distance_ratio_match(des1, des2, kp2, 0.8)
            idx = predict_label.nonzero()[0]
            # mkp1 = kp1.index_select(dim=0, index=idx.long())  # predict match keypoints in I1
            mkp1 = np.take(kp1, indices=idx, axis=0)
            # mkp2 = nn_kp2.index_select(dim=0, index=idx.long())  # predict match keypoints in I2
            mkp2 = np.take(nn_kp2, indices=idx, axis=0)

            keypoints1 = list(map(to_cv2_kp, mkp1))
            keypoints2 = list(map(to_cv2_kp, mkp2))
            DMatch = list(map(to_cv2_dmatch, np.arange(0, len(keypoints1))))

            print('Matches num:', len(DMatch), valid_sample['name1'], valid_sample['name2'])

            # matches1to2	Matches from the first image to the second one, which means that
            # keypoints1[i] has a corresponding point in keypoints2[matches[i]] .
            img1 = valid_sample['image1_raw'].squeeze(0).cpu().numpy().transpose(1, 2, 0)
            img2 = valid_sample['image2_raw'].squeeze(0).cpu().numpy().transpose(1, 2, 0)
            img1 = (img1 * 255).astype(np.uint8)
            img2 = (img2 * 255).astype(np.uint8)
            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               flags=2)
            matches_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, DMatch, None, **draw_params)
            matches_img_path = os.path.join(result_folder,
                                            valid_sample['name1'][0] + '_' + valid_sample['name2'][0] + '.png')
            cv2.imwrite(matches_img_path, matches_img)
            # cv2.imshow('matches', matches_img)
            # cv2.waitKey()

        # avg_loss.update(loss.item())

        # print(gct(), f'[{step+1 :03d} / {len(test_queue):03d}]', f'Eval loss: {loss.item():.03f}')

    # print(gct(), f'Eval size: {len(test_queue)}', f'Eval loss: {avg_loss.avg:.03f}')
    # return avg_loss.avg
    return 0


def to_cv2_kp(kp):
    # kp is like [batch_idx, y, x, channel]
    return cv2.KeyPoint(kp[1], kp[0], 0)


def to_cv2_dmatch(m):
    return cv2.DMatch(m, m, m, m)


if __name__ == '__main__':
    main()
