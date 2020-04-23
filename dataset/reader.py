# -*- coding: UTF-8 -*-
"""
数据读取器，定义读取一张图片的部分，针对不同数据集，需要自定义修改
"""
import traceback

import PIL
import numpy as np
import random
import math
import os
import codecs

from PIL import ImageEnhance
from PIL import Image
from config import train_parameters

def resize_img(img, input_size):
    target_size = input_size
    percent_h = float(target_size[1]) / img.size[1]
    percent_w = float(target_size[2]) / img.size[0]
    percent = min(percent_h, percent_w)
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    w_off = (target_size[2] - resized_width) / 2
    w_off = 0
    h_off = (target_size[1] - resized_height) / 2
    img = img.resize((resized_width, resized_height), Image.BICUBIC)
    array = np.ndarray((target_size[1], target_size[2], 3), np.uint8)
    array[:, :, 0] = 127
    array[:, :, 1] = 127
    array[:, :, 2] = 127
    ret = Image.fromarray(array)
    random_w_off = np.random.randint(0, w_off + 1)
    ret.paste(img, (random_w_off, int(h_off)))
    return ret

def resize_img_baseline(img, input_size):
    target_size = input_size
    percent_h = float(target_size[1]) / img.size[1]
    percent_w = float(target_size[2]) / img.size[0]
    percent = min(percent_h, percent_w)
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    w_off = (target_size[2] - resized_width) / 2
    h_off = (target_size[1] - resized_height) / 2
    img = img.resize((resized_width, resized_height), Image.ANTIALIAS)
    array = np.ndarray((target_size[1], target_size[2], 3), np.uint8)
    array[:, :, 0] = 127
    array[:, :, 1] = 127
    array[:, :, 2] = 127
    ret = Image.fromarray(array)
    ret.paste(img, (np.random.randint(0, w_off + 1), int(h_off)))
    return ret


def random_brightness(img):
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_distort_strategy']['brightness_prob']:
        brightness_delta = train_parameters['image_distort_strategy']['brightness_delta']
        delta = np.random.uniform(-brightness_delta, brightness_delta) + 1
        img = ImageEnhance.Brightness(img).enhance(delta)
    return img


def random_contrast(img):
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_distort_strategy']['contrast_prob']:
        contrast_delta = train_parameters['image_distort_strategy']['contrast_delta']
        delta = np.random.uniform(-contrast_delta, contrast_delta) + 1
        img = ImageEnhance.Contrast(img).enhance(delta)
    return img


def random_saturation(img):
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_distort_strategy']['saturation_prob']:
        saturation_delta = train_parameters['image_distort_strategy']['saturation_delta']
        delta = np.random.uniform(-saturation_delta, saturation_delta) + 1
        img = ImageEnhance.Color(img).enhance(delta)
    return img


def random_hue(img):
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_distort_strategy']['hue_prob']:
        hue_delta = train_parameters['image_distort_strategy']['hue_delta']
        delta = np.random.uniform(-hue_delta, hue_delta)
        img_hsv = np.array(img.convert('HSV'))
        img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
        img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')
    return img


def distort_image(img):
    prob = np.random.uniform(0, 1)
    # Apply different distort order
    if prob > 0.5:
        img = random_brightness(img)
        img = random_contrast(img)
        img = random_saturation(img)
        img = random_hue(img)
    else:
        img = random_brightness(img)
        img = random_saturation(img)
        img = random_hue(img)
        img = random_contrast(img)
    return img


def rotate_image(img):
    """
    图像增强，增加随机旋转角度
    """
    prob = np.random.uniform(0, 1)
    if prob > 0.5:
        angle = np.random.randint(-8, 8)
        img = img.rotate(angle)
    return img


def random_expand(img, keep_ratio=True):
    if np.random.uniform(0, 1) < train_parameters['image_distort_strategy']['expand_prob']:
        return img

    max_ratio = train_parameters['image_distort_strategy']['expand_max_ratio']
    w, h = img.size
    c = 3
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)
    oh = int(h * ratio_y)
    ow = int(w * ratio_x)
    off_x = random.randint(0, ow - w)
    off_y = random.randint(0, oh - h)

    out_img = np.zeros((oh, ow, c), np.uint8)
    for i in range(c):
        out_img[:, :, i] = train_parameters['mean_color']

    out_img[off_y: off_y + h, off_x: off_x + w, :] = img

    return Image.fromarray(out_img)


def preprocess(img):
    if train_parameters['apply_distort']:
        img = distort_image(img)
    img = random_expand(img)
    img = rotate_image(img)
    return img


def custom_reader(file_list, input_size, mode):
    def reader():
        np.random.shuffle(file_list)
        for line in file_list:
            # img_name, label
            parts = line.strip('\n').split('jpg\t')
            image_path = parts[0]
            image_path = image_path + 'jpg'
            if not os.path.exists(image_path):
                print('文件不存在', image_path)
                continue
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            label = [int(train_parameters['label_dict'][c]) for c in parts[-1] if c in train_parameters['label_dict']]
            if len(label) == 0 or len(label) >= train_parameters['max_char_per_line']:
                continue
            if mode == 'train':
                img = preprocess(img)
            img = resize_img_baseline(img, input_size)
            img = img.convert('L')
            # img.save(image_path)
            img = np.array(img).astype('float32') - train_parameters['mean_color']
            # img *= 0.007843
            img = img[np.newaxis, ...]
            # print("{0} {1}".format(image_path, label))
            yield img, label

    return reader


def custom_reader1():
    def reader():
        for i in range(100000):
            image = np.random.random([1, 48, 512]).astype('float32')
            label = i
            yield image, label

    return reader

if __name__ == '__main__':
    # from paddle import fluid

    # total_step = 30
    # LR = 0.001
    # with fluid.dygraph.guard():
    #     lr = fluid.layers.piecewise_decay([total_step // 3, total_step * 2 // 3], [LR, LR * 0.1, LR * 0.01])
    #     # lr = fluid.layers.polynomial_decay(LR,total_step,1e-7,power=0.9)
    #     from crnn import CRNN

    #     crnn = CRNN(train_parameters["class_dim"] + 1, batch_size=16)
    #     optimizer = fluid.optimizer.Adam(learning_rate=lr, parameter_list=crnn.parameters())
    #     step = []
    #     lr = []
    #     for x in range(total_step):
    #         step.append(x)
    #         l = fluid.dygraph.to_variable(np.array([1]))
    #         optimizer.minimize(l)
    #         lr.append(optimizer.current_step_lr())
    #         print(x,optimizer.current_step_lr())

    #     from matplotlib import pyplot as plt

    #     plt.plot(step, lr)
    #     plt.show()
    import paddle
    temp_reader = custom_reader1()
    train_reader = paddle.batch(temp_reader,
                                    batch_size=128)
    import time
    tic = time.time()
    for i, data in enumerate(train_reader()):
        if (i+1) % 100 == 0:
            print(time.time()-tic,len(data),i)
            tic = time.time()