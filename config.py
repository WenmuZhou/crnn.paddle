# -*- coding: UTF-8 -*-
"""
CRNN的网络，文字识别的配置部分
"""
import os
from utils import load

train_parameters = {
    "model_name": "crnn",
    "input_size": [1, 48, 512],
    "train_list": r"/mnt/d/dataset/ocr_simple_dataset/train.txt",
    "eval_list": r"/mnt/d/dataset/ocr_simple_dataset/val.txt",
    "max_char_per_line": 24,
    "label_list": load('dict.txt'),
    "class_dim": -1,
    "label_dict": {},
    "r_label_dict": {},
    "image_count": -1,
    "continue_train": False,
    "pretrained": False,
    "save_model_dir": "output/crnn",
    'start_epoch':0,
    "num_epochs": 80,
    "train_batch_size": 2,
    "eval_batch_size": 2,
    "mean_color": 127.5,
    "apply_distort": True,
    "image_distort_strategy": {
        "expand_prob": 0.5,
        "expand_max_ratio": 2,
        "hue_prob": 0.5,
        "hue_delta": 18,
        "contrast_prob": 0.5,
        "contrast_delta": 0.5,
        "saturation_prob": 0.5,
        "saturation_delta": 0.5,
        "brightness_prob": 0.5,
        "brightness_delta": 0.125
    },
    "learning_rate": 0.001,
}

os.makedirs(train_parameters['save_model_dir'], exist_ok=True)
train_parameters["label_dict"] = {c: i for i, c in enumerate(train_parameters['label_list'])}
train_parameters["r_label_dict"] = {i: c for i, c in enumerate(train_parameters['label_list'])}
train_parameters['class_dim'] = len(train_parameters['label_dict'])
