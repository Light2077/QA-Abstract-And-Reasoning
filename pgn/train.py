import tensorflow as tf

from pgn.model import PGN
from pgn.train_helper import train_model, get_train_msg
from utils.config_gpu import config_gpu
from utils.params import get_params
from utils.saveLoader import Vocab
from utils.config import PGN_CKPT
import numpy as np
import sys

def train(params):
    # GPU资源配置
    config_gpu()
    # 读取vocab训练
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    params['vocab_size'] = vocab.count
    params["trained_epoch"] = get_train_msg(PGN_CKPT)
    # 学习率衰减
    params["learning_rate"] *= np.power(0.95, params["trained_epoch"])

    # 构建模型
    print("Building the model ...")
    model = PGN(params)
    # 获取保存管理者
    checkpoint = tf.train.Checkpoint(PGN=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, PGN_CKPT, max_to_keep=5)

    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")


    # 训练模型
    print("开始训练模型..")
    print("trained_epoch:", params["trained_epoch"])
    print("mode:", params["mode"])
    print("epochs:", params["epochs"])
    print("batch_size:", params["batch_size"])
    print("max_enc_len:", params["max_enc_len"])
    print("max_dec_len:", params["max_dec_len"])
    print("learning_rate:", params["learning_rate"])

    train_model(model, vocab, params, checkpoint_manager)


if __name__ == '__main__':
    # 获得参数
    params = get_params()
    # params["debug_mode"] = True
    # print(params["debug_mode"])
    # 训练模型
    train(params)
