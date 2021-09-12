import os
import os.path
import errno
import numpy as np
import sys
import torch.utils.data as data
import csv
import cv2
import random

from PIL import Image
from user_define import config as cf


class glioma(data.Dataset):
    """自定义数据集"""

    def __init__(self, images_root, images_class, images_num, is_balanced, transform=None):
        self.images_root = images_root
        self.images_class = images_class
        self.images_num = images_num
        self.is_balanced = is_balanced
        self.transform = transform
        self.class_balanced_num = self._get_balanced_class_nums()
        self.class_to_idx = self._find_class_idx()
        self.samples = self._get_samples()

    def _get_balanced_class_nums(self):
        balanced_nums = {}
        for labels in self.images_class:
            if self.is_balanced:
                cls_nums = len(labels)
            else:
                cls_nums = 1
            for target in labels:
                balanced_nums[target] = self.images_num // cls_nums
        print(balanced_nums)
        return balanced_nums

    def _get_samples(self):
        samples = []

        print(self.class_to_idx)
        for target in sorted(self.class_to_idx.keys()):
            d = os.path.join(self.images_root, str(target))
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                random.seed(705)
                random.shuffle(fnames)
                target_nums = 0
                for fname in fnames:
                    if fname.split('.')[-1] == 'png':
                        path = os.path.join(root, fname)
                        if target_nums >= self.class_balanced_num[target]:
                            break
                        item = (path, self.class_to_idx[target])
                        samples.append(item)
                        target_nums += 1
        return samples

    def _find_class_idx(self):
        class_to_idx = {}
        for i in range(len(self.images_class)):
            for j in self.images_class[i]:
                class_to_idx[j] = i
        return class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        img_path, label = self.samples[item]
        img = Image.open(img_path).convert('RGB')
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.samples[item]))

        if self.transform is not None:
            img = self.transform(img)

        return img, label


