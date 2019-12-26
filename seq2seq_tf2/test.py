# -*- coding:utf-8 -*-
# Created by LuoJie at 12/12/19
import tensorflow as tf
from seq2seq_tf2.batcher import beam_test_batch_generator
from seq2seq_tf2.model import Seq2Seq

from seq2seq_tf2.test_helper import beam_decode, greedy_decode

from utils.config import CKPT_DIR, TEST_X, SEQ2SEQ_CKPT
from utils.saveLoader import load_test_dataset
from utils.config_gpu import config_gpu
from utils.params import get_params
from utils.saveLoader import Vocab
import pandas as pd


def test(params):
    assert params["mode"].lower() in ["test", "eval"], "change training mode to 'test' or 'eval'"
    assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"
    # GPU资源配置
    config_gpu()

    print("Building the model ...")
    model = Seq2Seq(params)

    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])

    print("Creating the checkpoint manager")
    print("Creating the checkpoint manager")
    checkpoint = tf.train.Checkpoint(Seq2Seq=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, SEQ2SEQ_CKPT, max_to_keep=5)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    print("Model restored")

    if params['greedy_decode']:
        params['batch_size'] = 256
        results = predict_result(model, params, vocab, params['test_save_dir'])
    else:
        b = beam_test_batch_generator(params["beam_size"])
        results = []
        for batch in b:
            best_hyp = beam_decode(model, batch, vocab, params)
            results.append(best_hyp.abstract)
        save_predict_result(results, params['result_save_path'])
        print('save result to :{}'.format(params['result_save_path']))

    return results


def predict_result(model, params, vocab, result_save_path):
    test_X = load_test_dataset()
    # 预测结果
    results = greedy_decode(model, test_X, params['batch_size'], vocab, params)
    # 保存结果
    # save_predict_result(results, result_save_path)

    return results


def save_predict_result(results, result_save_path):
    # 读取结果
    test_df = pd.read_csv(TEST_X)
    # 填充结果
    test_df['Prediction'] = results
    # 　提取ID和预测结果两列
    test_df = test_df[['QID', 'Prediction']]
    # 保存结果.
    test_df.to_csv(result_save_path, index=None, sep=',')


if __name__ == '__main__':
    # 获得参数
    params = get_params()
    # 获得参数
    results = test(params)
