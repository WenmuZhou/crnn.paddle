# -*- coding: UTF-8 -*-
"""
一些常用的工具函数，比如日志
"""
import pathlib
import json
import numpy as np

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


def greedy_decode(input, blank):
    ret = []
    for current_input in input:
        pred_idx = np.argmax(current_input, axis=1)
        current_decoded = []
        for c_i, c in enumerate(pred_idx):
            if c != blank:
                if len(current_decoded) == 0 or c != current_decoded[-1] or pred_idx[c_i - 1] == blank:
                    current_decoded.append(c)
        ret.append(current_decoded)
    return ret


def load(file_path: str):
    file_path = pathlib.Path(file_path)
    func_dict = {'.txt': _load_txt, '.json': _load_json, '.list': _load_txt}
    assert file_path.suffix in func_dict
    return func_dict[file_path.suffix](file_path)


def _load_txt(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = [x.strip().strip('\ufeff').strip('\xef\xbb\xbf') for x in f.readlines()]
    return content


def _load_json(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = json.load(f)
    return content


def save(data, file_path):
    file_path = pathlib.Path(file_path)
    func_dict = {'.txt': _save_txt, '.json': _save_json}
    assert file_path.suffix in func_dict
    return func_dict[file_path.suffix](data, file_path)


def _save_txt(data, file_path):
    """
    将一个list的数组写入txt文件里
    :param data:
    :param file_path:
    :return:
    """
    if not isinstance(data, list):
        data = [data]
    with open(file_path, mode='w', encoding='utf8') as f:
        f.write('\n'.join(data))


def _save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


def punctuation_mend(string):
    # 输入字符串或者txt文件路径
    import unicodedata
    import pathlib

    table = {ord(f): ord(t) for f, t in zip(
        u'，。！？【】（）％＃＠＆１２３４５６７８９０“”‘’',
        u',.!?[]()%#@&1234567890""\'\'')}  # 其他自定义需要修改的符号可以加到这里
    res = unicodedata.normalize('NFKC', string)
    res = res.translate(table)
    return res
