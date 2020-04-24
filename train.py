# -*- coding: UTF-8 -*-
"""
训练常基于CRNN的网络，文字识别
"""
import paddle.fluid as fluid
import time
from dataset import get_loader
import numpy as np
from utils import setup_logger, greedy_decode, save, load
from config import train_parameters
from crnn import CRNN

import os

os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = '0.82'

logger = setup_logger(os.path.join(train_parameters['save_model_dir'], 'train.log'))


def acc_batch(preds, labels):
    labels = [x[:np.where(x == -1)[0][0]] for x in labels]
    output = greedy_decode(preds, blank=train_parameters["class_dim"])
    total = 0
    right = 0
    for y, p in zip(labels, output):
        y_s = "".join([train_parameters['r_label_dict'][c] for c in y])
        p_s = "".join([train_parameters['r_label_dict'][c] for c in p])
        if y_s == p_s:
            right += 1
        total += 1
    return right, total


def eval_model(crnn, place):
    logger.info("start to eval")
    file_list = open(train_parameters['eval_list']).readlines()
    eval_reader = get_loader(file_list=file_list, input_size=train_parameters['input_size'], max_char_per_line=train_parameters['max_char_per_line'],
                             mean_color=train_parameters['mean_color'], batch_size=train_parameters['eval_batch_size'],
                             label_dict=train_parameters['label_dict'], mode='eval', place=place)

    all_acc = 0
    all_num = 0
    for batch_id, (img, label, label_len) in enumerate(eval_reader()):
        out = crnn(img)
        cur_acc, cur_num = acc_batch(out.numpy(), label.numpy())
        all_acc += cur_acc
        all_num += cur_num
    return 1.0 * all_acc / all_num


def train():
    epoch_num = train_parameters["num_epochs"]
    batch_size = train_parameters["train_batch_size"]

    place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()
    logger.info('train with {}'.format(place))
    with fluid.dygraph.guard(place):
        # 数据加载
        file_list = open(train_parameters['train_list']).readlines()
        train_reader = get_loader(file_list=file_list, input_size=train_parameters['input_size'], max_char_per_line=train_parameters['max_char_per_line'],
                                  mean_color=train_parameters['mean_color'], batch_size=train_parameters['train_batch_size'], mode='train',
                                  label_dict=train_parameters['label_dict'], place=place)

        batch_num = len(file_list) // batch_size

        crnn = CRNN(train_parameters["class_dim"] + 1, batch_size=batch_size)
        total_step = batch_num * epoch_num
        LR = train_parameters['learning_rate']
        # lr = fluid.layers.polynomial_decay(train_parameters['learning_rate'], batch_num * epoch_num, 1e-7, power=0.9)
        lr = fluid.layers.piecewise_decay([total_step // 3, total_step * 2 // 3], [LR, LR * 0.1, LR * 0.01])
        optimizer = fluid.optimizer.Adam(learning_rate=lr, parameter_list=crnn.parameters())

        if train_parameters["continue_train"]:
            # 加载上一次训练的模型，继续训练
            params_dict, opt_dict = fluid.load_dygraph('{}/crnn_latest'.format(train_parameters['save_model_dir']))
            crnn.set_dict(params_dict)
            optimizer.set_dict(opt_dict)
            logger.info("load model from {}".format(train_parameters['save_model_dir']))

        current_best = -1
        start_epoch = 0
        for epoch in range(start_epoch, epoch_num):
            crnn.train()
            tic = time.time()
            for batch_id, (img, label, label_len) in enumerate(train_reader()):
                out = crnn(img)

                out_for_loss = fluid.layers.transpose(out, [1, 0, 2])
                input_length = np.array([out.shape[1]] * out.shape[0]).astype("int64")
                input_length = fluid.dygraph.to_variable(input_length)
                input_length.stop_gradient = True
                loss = fluid.layers.warpctc(input=out_for_loss, label=label, input_length=input_length, label_length=label_len,
                                            blank=train_parameters["class_dim"], norm_by_times=True)
                avg_loss = fluid.layers.reduce_mean(loss)

                cur_acc_num, cur_all_num = acc_batch(out.numpy(), label.numpy())
                if batch_id % 1 == 0:
                    logger.info(
                        "epoch [{}/{}], step [{}/{}], loss: {:.6f}, acc: {:.4f}, lr: {}, time: {:.4f}".format(epoch, epoch_num,
                                                                                                              batch_id,
                                                                                                              batch_num,
                                                                                                              avg_loss.numpy()[0],
                                                                                                              cur_acc_num / cur_all_num,
                                                                                                              optimizer.current_step_lr(),
                                                                                                              time.time() - tic))
                    tic = time.time()
                avg_loss.backward()
                optimizer.minimize(avg_loss)
                crnn.clear_gradients()

            fluid.save_dygraph(crnn.state_dict(), '{}/crnn_latest'.format(train_parameters['save_model_dir']))
            fluid.save_dygraph(optimizer.state_dict(), '{}/crnn_latest'.format(train_parameters['save_model_dir']))
            crnn.eval()
            ratio = eval_model(crnn, place=place)
            if ratio >= current_best:
                fluid.save_dygraph(crnn.state_dict(), '{}/crnn_best'.format(train_parameters['save_model_dir']))
                fluid.save_dygraph(optimizer.state_dict(), '{}/crnn_best'.format(train_parameters['save_model_dir']))
                current_best = ratio
                logger.info("save model to {}, current best right ratio:{:.2%}".format(train_parameters['save_model_dir'], ratio))
    logger.info("train end")


if __name__ == '__main__':
    train()
