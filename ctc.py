# -*- coding: utf-8 -*-
# @Time    : 2020/4/23 17:05
# @Author  : zhoujun

import paddle.fluid as fluid
import numpy as np
import paddle
print(paddle.__version__)
# length of the longest logit sequence
max_seq_length = 5
# length of the longest label sequence
max_label_length = 3
# number of logit sequences
batch_size = 16
# class num
class_num = 5

place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()
with fluid.dygraph.guard(place):
    logits = np.random.rand(max_seq_length, batch_size, class_num + 1).astype("float32")
    logits_length = np.array([max_seq_length] * batch_size).astype("int64")
    label = np.random.randint(0, class_num, [batch_size, max_label_length]).astype("int32")
    label_length = np.array([max_label_length] * batch_size).astype("int64")


    label = fluid.dygraph.to_variable(label)
    label_length = fluid.dygraph.to_variable(label_length)
    logits = fluid.dygraph.to_variable(logits)
    logits_length = fluid.dygraph.to_variable(logits_length)

    label.stop_gradient = True
    label_length.stop_gradient = True
    logits_length.stop_gradient = True

    cost = fluid.layers.warpctc(input=logits, label=label, input_length=logits_length, label_length=label_length)
    print(cost)
