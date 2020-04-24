# -*- coding: utf-8 -*-
# @Time    : 2020/4/23 13:11
# @Author  : zhoujun
import numpy as np


def get_loader(**kwargs):
    import paddle
    if paddle.__version__ == '0.0.0' or paddle.__version__ >= '2.0.0':
        return torch_loader(**kwargs)
    else:
        return pd_loader(**kwargs)
    pass

#
# def collate_fn(batch):
#     img_list = []
#     label_list = []
#     for img, label in batch:
#         img_list.append(img)
#         label_list.append(label)
#     img = np.stack(img_list, axis=0)
#     return [img, label_list]


def torch_loader(**kwargs):
    from dataset.dataset import ImageDataset
    from paddle.io import DataLoader
    dataset = ImageDataset(data_list=kwargs['file_list'], input_size=kwargs['input_size'], max_char_per_line=kwargs['max_char_per_line'],
                           mean_color=kwargs['mean_color'], label_dict=kwargs['label_dict'], mode=kwargs['mode'])
    train_reader = DataLoader(dataset, places=kwargs['place'], num_workers=0, batch_size=kwargs['batch_size'], drop_last=True, shuffle=True)
    return train_reader


def pd_loader(**kwargs):
    import paddle
    from dataset import reader
    temp_reader = reader.custom_reader(file_list=kwargs['file_list'], input_size=kwargs['input_size'], mode=kwargs['mode'])
    train_reader = paddle.batch(paddle.reader.shuffle(temp_reader, buf_size=1), batch_size=kwargs['batch_size'])
    return train_reader
