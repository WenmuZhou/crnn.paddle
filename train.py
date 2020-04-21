# -*- coding: UTF-8 -*-
"""
训练常基于CRNN的网络，文字识别
"""
import paddle.fluid as fluid
import paddle
import math
import time
import reader
import numpy as np
import utils
from utils import setup_logger
from config import train_parameters
from crnn import CRNN

import os

os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = '0.82'

logger = setup_logger('train.log')


def eval_model(crnn):
    logger.info("start to eval")
    file_list = open(train_parameters['eval_list']).readlines()
    # file_list = [x.replace('D:/', '/mnt/d/') for x in file_list]
    temp_reader = reader.custom_reader(file_list, train_parameters['input_size'], 'eval')
    eval_reader = paddle.batch(temp_reader, batch_size=train_parameters['eval_batch_size'])

    for batch_id, data in enumerate(eval_reader()):
        dy_x_data = np.array([x[0] for x in data]).astype('float32')
        image_length = np.array([x[1] for x in data]).astype('int64')
        label_len = np.array([len(x[2]) for x in data])

        y_data = [x[2] for x in data]

        img = fluid.dygraph.to_variable(dy_x_data)
        out = crnn(img)
        image_length = fluid.dygraph.to_variable(np.array(image_length).astype('int64'))
        # logger.info("intput image length:{}".format(image_length))
        # logger.info("in eval pred:{}".format(out))
        # outputctc_greedy_decoder, output_length = fluid.layers.ctc_greedy_decoder(out, blank=train_parameters["class_dim"], input_length=image_length)
        # logger.info("decoded output length:{}".format(output_length))
        # print(np.array(output))
        output = utils.greedy_decode(out.numpy(), blank=train_parameters["class_dim"],
                                     input_length=image_length.numpy())
        total = 0
        right = 0
        for y_p in zip(y_data, output):
            y = y_p[0]
            p = y_p[1]
            y_s = "".join([train_parameters['r_label_dict'][c] for c in y])
            p_s = "".join([train_parameters['r_label_dict'][c] for c in p])
            # logger.info("pred:{} answer:{}".format(p_s, y_s))
            if y_s == p_s:
                right += 1
            total += 1
        logger.info("eval right ratio:{:.2%}".format(1.0 * right / total))
        return 1.0 * right / total


def train():
    epoch_num = train_parameters["num_epochs"]
    batch_size = train_parameters["train_batch_size"]

    with fluid.dygraph.guard():
        crnn = CRNN(train_parameters["class_dim"] + 1, batch_size=batch_size)
        optimizer = fluid.optimizer.Adam(learning_rate=train_parameters['learning_rate'],
                                         parameter_list=crnn.parameters())

        # 数据加载
        file_list = open(train_parameters['train_list']).readlines()
        # file_list = [x.replace('D:/', '/mnt/d/') for x in file_list]
        temp_reader = reader.custom_reader(file_list, train_parameters['input_size'], 'train')
        train_reader = paddle.batch(temp_reader, batch_size=batch_size)

        batch_num = len(file_list) // batch_size
        if train_parameters["continue_train"]:
            # 加载上一次训练的模型，继续训练
            model, _ = fluid.dygraph.load_dygraph(train_parameters['save_model_dir'])
            crnn.load_dict(model)
            logger.info("load model from {}".format(train_parameters['save_model_dir']))

        max_char_per_line = train_parameters['max_char_per_line']
        current_best = -1
        for epoch in range(epoch_num):
            crnn.train()
            tic = time.time()
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array([x[0] for x in data]).astype('float32')
                image_length = np.array([x[1] for x in data]).astype('int64')
                label_len = np.array([len(x[2]) for x in data]).astype('int64')
                y_data = np.array([x[2] + [0] * (max_char_per_line - len(x[2])) for x in data]).astype("int32")

                img = fluid.dygraph.to_variable(dy_x_data)
                out = crnn(img)

                label = fluid.dygraph.to_variable(y_data)
                label_len = fluid.dygraph.to_variable(label_len)
                image_length = fluid.dygraph.to_variable(image_length)

                label.stop_gradient = True
                label_len.stop_gradient = True
                image_length.stop_gradient = True

                out = fluid.layers.transpose(out, [1, 0, 2])
                loss = fluid.layers.warpctc(input=out, label=label, input_length=image_length, label_length=label_len,
                                            blank=train_parameters["class_dim"], norm_by_times=True)
                avg_loss = fluid.layers.reduce_mean(loss)

                if batch_id % 20 == 0:
                    # ratio = eval_model(crnn)
                    logger.info(
                        "loss at epoch [{}/{}], step [{}/{}], loss: {:.6f}, time: {:.4f}".format(epoch, epoch_num,
                                                                                                 batch_id,
                                                                                                 batch_num,
                                                                                                 avg_loss.numpy()[0],
                                                                                                 time.time() - tic))
                    tic = time.time()

                avg_loss.backward()
                optimizer.minimize(avg_loss)
                crnn.clear_gradients()

            crnn.eval()
            ratio = eval_model(crnn)
            if ratio >= current_best:
                fluid.dygraph.save_dygraph(crnn.state_dict(), train_parameters['save_model_dir'])
                current_best = ratio
                logger.info("save model to {}, current best right ratio:{:.2%}"
                            .format(train_parameters['save_model_dir'], ratio))

    logger.info("train end")


if __name__ == '__main__':
    train()