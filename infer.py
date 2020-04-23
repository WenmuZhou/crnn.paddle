# -*- coding: UTF-8 -*-
"""
加载模型验证
"""
import utils
import paddle.fluid as fluid
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
from paddle.fluid.dygraph import TracedLayer

from dataset.reader import resize_img
from crnn import CRNN
from config import train_parameters


def precess_img(img_path):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = resize_img(img, train_parameters['input_size'])
    img = img.convert('L')
    img = np.array(img).astype('float32') - train_parameters['mean_color']
    img = img[np.newaxis, ...]
    img = np.expand_dims(img, 0)
    return img


def infer(files, save_static_path=None):
    result_list = []
    place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()
    print('train with {}'.format(place))
    with fluid.dygraph.guard(place):
        params, _ = fluid.load_dygraph('{}/crnn_best'.format('output/baidu_model'))#train_parameters['save_model_dir']))
        # crnn = CRNN(train_parameters["class_dim"] + 1, 1)
        crnn = CRNN(3828, 1)
        crnn.load_dict(params)
        crnn.eval()
        for file in tqdm(files):
            img = precess_img(file)
            img = fluid.dygraph.to_variable(img).astype('float32')
            if save_static_path is not None:
                out_dygraph, static_layer = TracedLayer.trace(crnn, inputs=[img])
                # 将转换后的模型保存
                static_layer.save_inference_model(save_static_path, feed=[0], fetch=[0])
            pred = crnn(img)
            output = utils.greedy_decode(pred.numpy(), blank=train_parameters["class_dim"])
            p_s = "".join([train_parameters['r_label_dict'][c] for c in output[0]])
            result_list.append('{0}\t{1}'.format(os.path.basename(file), p_s))
            break
    return result_list


def static_infer(files, save_static_path):
    # 静态图中需要使用执行器执行之前已经定义好的网络
    place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()
    print('train with {}'.format(place))
    exe = fluid.Executor(place)
    program, feed_vars, fetch_vars = fluid.io.load_inference_model(save_static_path, exe)
    # 静态图中需要调用执行器的run方法执行计算过程
    result_list = []
    for file in tqdm(files):
        img = precess_img(file)
        fetch, = exe.run(program, feed={feed_vars[0]: img}, fetch_list=fetch_vars)
        output = utils.greedy_decode(fetch, blank=train_parameters["class_dim"])
        p_s = "".join([train_parameters['r_label_dict'][c] for c in output[0]])
        result_list.append('{0}\t{1}'.format(os.path.basename(file), p_s))
    return result_list


if __name__ == "__main__":
    # image_path = sys.argv[1]
    from utils import save

    files = ['D:/dataset/data/val/0_song5_0_3.jpg'] # gt is oco9w
    # files = [os.path.join('/home/aistudio/data/data10879', 'test_images', file) for file in
    #          os.listdir(os.path.join('/home/aistudio/data/data10879', 'test_images')) if file.endswith('.jpg')]
    save_static_path = None#'./saved_infer_model'
    result_list = infer(files, save_static_path)
    # result_list = static_infer(files, save_static_path)
    # save(result_list, 'predict.txt')
    print(result_list)
