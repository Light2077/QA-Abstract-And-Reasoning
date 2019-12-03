import pandas as pd
import numpy as np
from utils.config import *


def save_vocab(path, vocab_index_):
    with open(path, mode="w", encoding="utf-8") as f:
        for key, value in vocab_index_.items():
            f.write(str(key)+" ")
            f.write(str(value)+"\n")

def load_vocab(path):
    # path:
    vocab_index_ = {}
    index_vocab_ = {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f.readlines():
            [vocab, index] = line.strip("\n").split(" ")
            vocab_index_[vocab] = int(index)
            index_vocab_[int(index)] = vocab
    return vocab_index_, index_vocab_


def load_dataset():
    """
    :return: 加载处理好的数据集
    """
    train_x = np.loadtxt(TRAIN_X)
    train_y = np.loadtxt(TRAIN_Y)
    test_x = np.loadtxt(TEST_X)
    train_x.dtype = 'float64'
    train_y.dtype = 'float64'
    test_x.dtype = 'float64'
    return train_x, train_y, test_x
