# -*- coding: UTF-8 -*-
"""
加载模型验证
"""
import utils
import paddle.fluid as fluid
import paddle
from PIL import Image
import time
import numpy as np
import os
import config
import sys

from reader import resize_img
from crnn import CRNN
from config import train_parameters


def infer(infer_path):
    utils.logger.info("start infer")
    config.init_train_parameters("infer")
    with fluid.dygraph.guard():
        params, _ = fluid.dygraph.load_dygraph(config.train_parameters['save_model_dir'])
        crnn = CRNN(train_parameters['model_name'], train_parameters["class_dim"])
        crnn.load_dict(params)
        crnn.eval()
        start_time = time.time()

        img = Image.open(infer_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img, img_length = resize_img(img, train_parameters['input_size'])
        img = img.convert('L')
        img = np.array(img).astype('float32') - train_parameters['mean_color']
        img = img[np.newaxis, ...]
        img = fluid.dygraph.to_variable(np.array([img]).astype('float32'))
        image_length = fluid.dygraph.to_variable(np.array([img_length]).astype('int64'))
        pred, img_length = crnn(img, img_length)
        pred_val = pred.numpy()
        utils.logger.info("pred shape:{}".format(pred_val.shape))
        utils.logger.info("pred value:{}".format(pred_val))
        utils.logger.info("img_length value:{}".format(img_length))
        # pred = fluid.layers.transpose(pred, [1, 0, 2])
        pred_max_idx = np.argmax(pred_val, axis=1)
        utils.logger.info("pred_max_idx value:{}".format(pred_max_idx))
        output, output_length = fluid.layers.ctc_greedy_decoder(pred, train_parameters['class_dim'], image_length)
        output = output.numpy()
        output_length = output_length.numpy()
        utils.logger.info(
            "output:{} output_length:{} cost time: {}".format(output, output_length, time.time() - start_time))


if __name__ == "__main__":
    image_name = sys.argv[1]
    print(os.getcwd())
    image_path = image_name
    infer(image_path)
