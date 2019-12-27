import tensorflow as tf

from seq2seq_tf2.model import Seq2Seq
from seq2seq_tf2.train_helper import train_model
from utils.config_gpu import config_gpu
from utils.params import get_params
from utils.saveLoader import Vocab
from utils.config import SEQ2SEQ_CKPT

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
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, SEQ2SEQ_CKPT, max_to_keep=5)

    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    # 训练模型
    print("开始训练模型..")
    print("mode:", params["mode"])
    print("epochs:", params["epochs"])
    print("batch_size:", params["batch_size"])
    print("max_enc_len:", params["max_enc_len"])
    print("max_dec_len:", params["max_dec_len"])

    train_model(model, vocab, params, checkpoint_manager)


if __name__ == '__main__':
    # 获得参数
    params = get_params()
    # 训练模型
    train(params)
