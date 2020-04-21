# -*- coding: UTF-8 -*-
"""
一些常用的工具函数，比如日志
"""
import os
import logging
import functools
import math
from collections import namedtuple

import numpy as np
from PIL import Image

MIN_FLOAT = 0.000001
MAX_FLOAT = 20

def setup_logger(log_file_path: str = None):
    import logging
    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', )
    logger = logging.getLogger('crnn.gluon')
    if log_file_path is not None:
        file_handle = logging.FileHandler(log_file_path)
        file_handle.setFormatter(logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s'))
        logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    logger.info('logger init finished')
    return logger


def greedy_decode(input, blank, input_length):
    ret = []
    for idx, length in enumerate(input_length):
        current_input = input[idx]
        pred_idx = np.argmax(current_input, axis=1)[:length]
        current_decoded = []
        for c_i, c in enumerate(pred_idx):
            if c != blank:
                if len(current_decoded) == 0 or c != current_decoded[-1] or pred_idx[c_i - 1] == blank:
                    current_decoded.append(c)
        ret.append(current_decoded)
    return ret
