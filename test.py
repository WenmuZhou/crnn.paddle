import paddle.fluid as fluid
import numpy as np

# length of the longest logit sequence
max_seq_length = 32
# number of logit sequences
batch_size = 128
place = fluid.CPUPlace()
with fluid.dygraph.guard(place):
    x = np.random.rand(max_seq_length, batch_size, 64).astype("float32")
    y = np.random.randint(0, 64, [max_seq_length * batch_size, 1]).astype("int32")
    x = fluid.dygraph.to_variable(x)
    y = fluid.dygraph.to_variable(y)
    label_len = fluid.dygraph.to_variable(np.array([32] * batch_size).astype("int64"))
    pred_len = fluid.dygraph.to_variable(np.array([32] * batch_size).astype("int64"))
    cost = fluid.layers.warpctc(input=x, label=y,
                                input_length=pred_len,
                                label_length=label_len, blank=63)
    print(cost)