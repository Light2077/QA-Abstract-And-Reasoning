# -*- coding:utf-8 -*-
# Created by LuoJie at 11/29/19
import tensorflow as tf

from seq2seq.model import Seq2Seq
from seq2seq.train_helper import train_model
from utils.config_gpu import config_gpu
from utils.params import get_params
from utils.saveLoader import Vocab

def train(params):
    # GPU资源配置
    config_gpu()
    # 读取vocab训练
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    params['vocab_size'] = vocab.count
    # 构建模型
    print("Building the model ...")
    model = Seq2Seq(params)
    # 获取保存管理者
    checkpoint = tf.train.Checkpoint(Seq2Seq=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, params['checkpoint_dir'], max_to_keep=5)
    # 训练模型
    train_model(model, vocab, params, checkpoint_manager)


if __name__ == '__main__':
    # 获得参数
    params = get_params()
    # 训练模型
    train(params)
