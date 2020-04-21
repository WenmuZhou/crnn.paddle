#!/usr/bin/env python
# encoding: utf-8
'''
@author: zhoujun
@time: 2020/4/21 下午10:01
'''

# -*- coding: utf-8 -*-
# @Time    : 2018/8/24 10:20
# @Author  : zhoujun
import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(os.path.abspath(__name__)).parent))
import argparse
import cv2
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from itertools import groupby
from utils import punctuation_mend


def split(data_dict, num=10):
    if num == 1:
        x = [x[0] for x in data_dict]
        y = [x[1] for x in data_dict]
    else:
        x = []
        y = []
        for k, g in groupby(data_dict, key=lambda item: item[0] // num):
            cur_sum = sum([x[1] for x in g])
            print('{}-{}: {}'.format(k * num, (k + 1) * num - 1, cur_sum))
            x.append((k + 1) * num - 1)
            y.append(cur_sum)
    return x, y


def show_dict(data_dict: dict, num, title):
    from matplotlib import pyplot as plt
    data_dict = sorted(data_dict.items(), key=lambda item: item[0])
    x, y = split(data_dict, num)
    y = np.array(y)
    y = y / y.sum()
    print(x[y.argmax()])
    plt.figure()
    plt.title(title)
    plt.plot(x, y)
    plt.savefig('1.jpg')
    # plt.show()


def get_key(label_file_list, ignore_chinese_punctuation, show_max_img=False):
    data_list = []
    label_list = []
    len_dict = defaultdict(int)
    h_dict = defaultdict(int)
    w_dict = defaultdict(int)
    for label_path in label_file_list:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines(), desc=label_path):
                line = line.strip('\n').replace('.jpg ', '.jpg\t').replace('.png ', '.png\t').split('\t')
                if len(line) > 1 and os.path.exists(line[0]):
                    data_list.append(line[0])
                    label = line[1]
                    if ignore_chinese_punctuation:
                        label = punctuation_mend(label)
                    label_list.append(label)
                    len_dict[len(line[1])] += 1
                    if show_max_img:
                        img = cv2.imread(line[0])
                        h, w = img.shape[:2]
                        h_dict[h] += 1
                        w_dict[w] += 1
    if show_max_img:
        print('******************分析宽度******************')
        show_dict(w_dict, 10, 'w')
        print('******************分析高度******************')
        show_dict(h_dict, 1, 'h')
        print('******************分析label长度******************')
        show_dict(len_dict, 1, 'label')
    a = ''.join(sorted(set((''.join(label_list)))))
    return a


if __name__ == '__main__':
    # 根据label文本生产key
    from config import train_parameters
    from utils import save

    parser = argparse.ArgumentParser()
    parser.add_argument('--label_file', nargs='+', help='label file', default=[""])
    args = parser.parse_args()

    label_file = [train_parameters['train_list'], train_parameters['eval_list']]
    alphabet = get_key(label_file, True, show_max_img=False).replace(' ', '')
    save(list(alphabet), 'dict.txt')
    # np.save('alphabet.npy', alphabet)
    print(alphabet)
