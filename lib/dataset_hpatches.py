import os
import numpy as np
import pandas as pd
import cv2
import torch
from skimage import io, transform, color
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from lib.system import gct
from lib.tensor import im_rescale


class HPatchesDataset(Dataset):

    def __init__(self, csv_file, root_dir, output_size=None, transform=None, test_flag=False):

        self.root_dir = root_dir
        self.hpatches_frame = pd.read_csv(os.path.join(root_dir, csv_file))
        self.output_size = output_size
        self.transform = transform
        self.test_flag = test_flag

    def __len__(self):
        return len(self.hpatches_frame)

    def __getitem__(self, idx):
        object_name = self.hpatches_frame.iloc[idx, 0]

        img1_name = self.hpatches_frame.iloc[idx, 1]
        img1_path = os.path.join(self.root_dir, object_name, img1_name)
        img1 = io.imread(img1_path)

        img2_name = self.hpatches_frame.iloc[idx, 2]
        img2_path = os.path.join(self.root_dir, object_name, img2_name)
        img2 = io.imread(img2_path)

        homo12 = np.asmatrix(
            self.hpatches_frame.iloc[idx, 3:].values.astype('float').reshape(3, 3)
        )
        homo21 = homo12.I

        if self.test_flag:
            sample = {
                'image1': self.transform(img1),
                'image2': self.transform(img2),
                'homo12': torch.from_numpy(np.asarray(homo12)),
                'homo21': torch.from_numpy(np.asarray(homo21))
            }
        else:
            sample = {
                'image1': img1,
                'image2': img2,
                'homo12': homo12,
                'homo21': homo21,
            }

            if self.transform:
                sample = self.transform(sample)

        sample["name1"] = img1_name
        sample["name2"] = img2_name
        # sample["file"] = self.files[idx]

        return sample


class Rescale(object):
    '''Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    '''

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        img1, img2, homo12, homo21 = (
            sample['image1'],
            sample['image2'],
            sample['homo12'],
            sample['homo21'],
        )
        img1_raw, img2_raw = img1, img2

        img1, img1h, img1w, sw1, sh1 = im_rescale(img1, self.output_size)
        img2, img2h, img2w, sw2, sh2 = im_rescale(img2, self.output_size)

        'x and y axes are axis 1 and 0 respectively'
        tform1 = np.mat(
            [[1 / sw1, 0, 0], [0, 1 / sh1, 0], [0, 0, 1]], dtype=homo12.dtype
        )
        tform2 = np.mat([[sw2, 0, 0], [0, sh2, 0], [0, 0, 1]], dtype=homo12.dtype)
        homo12 = tform2 * homo12 * tform1

        tform1 = np.mat([[sw1, 0, 0], [0, sh1, 0], [0, 0, 1]], dtype=homo12.dtype)
        tform2 = np.mat(
            [[1 / sw2, 0, 0], [0, 1 / sh2, 0], [0, 0, 1]], dtype=homo12.dtype
        )
        homo21 = tform1 * homo21 * tform2

        sample['image1'] = img1
        sample['image2'] = img2
        sample['img1_info'] = np.array([sh1, sw1])
        sample['img2_info'] = np.array([sh2, sw2])
        sample['img1_raw'] = img1_raw
        sample['img2_raw'] = img2_raw
        sample['homo12'] = homo12
        sample['homo21'] = homo21

        return sample


class LargerRescale(object):
    '''Rescale the image in a sample to a given size which larger than specific size .

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    '''

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        img1, img2, homo12, homo21 = (
            sample['image1'],
            sample['image2'],
            sample['homo12'],
            sample['homo21'],
        )
        sH, sW = self.output_size
        H1, W1, _ = img1.shape
        H2, W2, _ = img2.shape

        if H1 > sH and H2 > sH and W1 > sW and W2 > sW:
            rescale = Rescale(self.output_size)
            sample = rescale(sample)

        return sample


class RandomCrop(object):
    '''Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop is made.
    '''

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        img1, img2, homo12, homo21 = (
            sample['image1'],
            sample['image2'],
            sample['homo12'],
            sample['homo21'],
        )
        new_h, new_w = self.output_size

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # img1 is smaller than expect size
        # don't random crop but rescale
        if h1 <= new_h or w1 <= new_w:
            img1 = transform.resize(img1, (new_h, new_w), mode='constant')
            sw1, sh1 = new_w / w1, new_h / h1
            tform1_12 = np.mat(
                [[1 / sw1, 0, 0], [0, 1 / sh1, 0], [0, 0, 1]], dtype=homo12.dtype
            )
            tform1_21 = np.mat(
                [[sw1, 0, 0], [0, sh1, 0], [0, 0, 1]], dtype=homo12.dtype
            )
        else:
            top1 = np.random.randint(0, h1 - new_h)
            left1 = np.random.randint(0, w1 - new_w)
            img1 = img1[top1 : top1 + int(new_h), left1 : left1 + int(new_w)]
            tform1_12 = np.mat(
                [[1, 0, left1], [0, 1, top1], [0, 0, 1]], dtype=homo12.dtype
            )
            tform1_21 = np.mat(
                [[1, 0, -left1], [0, 1, -top1], [0, 0, 1]], dtype=homo21.dtype
            )

        # same as img2
        if h2 <= new_h or w2 <= new_w:
            img2 = transform.resize(img2, (new_h, new_w), mode='constant')
            sw2, sh2 = new_w / w2, new_h / h2
            tform2_12 = np.mat(
                [[sw2, 0, 0], [0, sh2, 0], [0, 0, 1]], dtype=homo12.dtype
            )
            tform2_21 = np.mat(
                [[1 / sw2, 0, 0], [0, 1 / sh2, 0], [0, 0, 1]], dtype=homo12.dtype
            )
        else:
            top2 = np.random.randint(0, h2 - new_h)
            left2 = np.random.randint(0, w2 - new_w)
            img2 = img2[top2 : top2 + int(new_h), left2 : left2 + int(new_w)]
            tform2_12 = np.mat(
                [[1, 0, -left2], [0, 1, -top2], [0, 0, 1]], dtype=homo12.dtype
            )
            tform2_21 = np.mat(
                [[1, 0, left2], [0, 1, top2], [0, 0, 1]], dtype=homo21.dtype
            )

        homo12 = tform2_12 * homo12 * tform1_12
        homo21 = tform1_21 * homo21 * tform2_21

        sample['image1'] = img1
        sample['image2'] = img2
        sample['homo12'] = homo12
        sample['homo21'] = homo21

        return sample


class ToTensor(object):
    '''Convert ndarrays in sample to Tensors.'''

    def __call__(self, sample):
        img1, img1_info, img2, img2_info, homo12, homo21, img1_raw, img2_raw = (
            sample['image1'],
            sample['img1_info'],
            sample['image2'],
            sample['img2_info'],
            sample['homo12'],
            sample['homo21'],
            sample['img1_raw'],
            sample['img2_raw'],
        )
        
        sample['image1'] = TF.to_tensor(img1)
        sample['image2'] = TF.to_tensor(img2)
        sample['img1_info'] = torch.from_numpy(img1_info)
        sample['img2_info'] = torch.from_numpy(img2_info)
        sample['homo12'] = torch.from_numpy(np.asarray(homo12))
        sample['homo21'] = torch.from_numpy(np.asarray(homo21))
        sample['img1_raw'] = TF.to_tensor(img1_raw)
        sample['img2_raw'] = TF.to_tensor(img2_raw)

        return sample


class Normalize(object):
    def __init__(self, mean, std):
        # assert isinstance(mean, float)
        # assert isinstance(std, float)

        self.mean = mean
        self.std = std

    def __call__(self, sample):
        '''
        Parameters: tensor (Tensor) – Numpy image of size (H, W, C) to be normalized.
                    mean (sequence) – Sequence of means for each channel.
                    std (sequence) – Sequence of standard deviations for each channely.
        Returns:    Normalized Tensor image.
        Return type:Tensor
        :param sample: sample dict
        :return: sample dict
        '''

        TF.normalize(sample['image1'], self.mean, self.std, inplace=True)
        TF.normalize(sample['image2'], self.mean, self.std, inplace=True)
        return sample


class Grayscale(object):
    def __call__(self, sample):
        '''
        Parameters: np.array – np.array image of size (H, W, 1)
        Returns:    gray image
        Return type:np.array
        :param sample: sample list
        :return: sample list
        '''
        sample['image1'] = np.expand_dims(color.rgb2gray(sample['image1']), -1)
        sample['image2'] = np.expand_dims(color.rgb2gray(sample['image2']), -1)
        return sample

if __name__ == '__main__':
    pass