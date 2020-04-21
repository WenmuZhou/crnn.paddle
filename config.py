# -*- coding: UTF-8 -*-
"""
CRNN的网络，文字识别的配置部分
"""
import codecs
import os

train_parameters = {
    "model_name": "crnn",
    "input_size": [1, 48, 512],
    "train_list": r"/media/zj/资料/zj/dataset/test_crnn/train_linux.txt",
    "eval_list": r"/media/zj/资料/zj/dataset/test_crnn/val_linux.txt",
    "max_char_per_line": 24,
    "label_list": [x.strip('\n') for x in open('dict.txt').readlines()],
    "class_dim": -1,
    "label_dict": {},
    "r_label_dict": {},
    "image_count": -1,
    "continue_train": True,
    "pretrained": False,
    "pretrained_model_dir": "pretrained-model",
    "save_model_dir": "crnn-model/crnn",
    "num_epochs": 80,
    "train_batch_size": 32,
    "eval_batch_size": 32,
    "use_gpu": True,
    "ignore_thresh": 0.7,
    "mean_color": 127.5,
    "mode": "train",
    "multi_data_reader_count": 4,
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
    "learning_rate": 0.0001,
}


train_parameters["label_dict"] = {c: i for i, c in enumerate(train_parameters['label_list'])}
train_parameters["r_label_dict"] = {i: c for i, c in enumerate(train_parameters['label_list'])}
train_parameters['class_dim'] = len(train_parameters['label_dict'])