#!/usr/bin/env python
# encoding: utf-8
'''
@author: zhoujun
@time: 2020/4/21 下午10:01
'''
import os
import numpy as np
from PIL import Image
from paddle.io import Dataset, DataLoader

from .reader import preprocess, resize_img_baseline


class ImageDataset(Dataset):
    def __init__(self, data_list, input_size, max_char_per_line, mean_color, label_dict, mode):
        super().__init__()
        self.mode = mode
        self.input_size = input_size
        self.max_char_per_line = max_char_per_line
        self.mean_color = mean_color
        self.data_list = self.load_data(data_list)
        self.label_dict = label_dict

    def __getitem__(self, idx):
        image_path, label = self.data_list[idx]
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.mode == 'train':
            img = preprocess(img)
        img = resize_img_baseline(img, self.input_size)
        img = img.convert('L')
        img = np.array(img).astype('float32') - self.mean_color
        img = img[np.newaxis, ...]
        label = [int(self.label_dict[c]) for c in label]
        len_label = len(label)
        label = label + [-1] * (self.max_char_per_line - len(label))
        return img, label, len_label

    def __len__(self):
        return len(self.data_list)

    def load_data(self, data_list):
        d = []
        for line in data_list:
            parts = line.strip('\n').split('jpg\t')
            image_path = parts[0]
            image_path = image_path + 'jpg'
            if not os.path.exists(image_path):
                print('文件不存在', image_path)
                continue
            label = parts[-1]
            if len(label) == 0 or len(label) >= self.max_char_per_line:
                continue
            d.append((image_path, label))
        return d
