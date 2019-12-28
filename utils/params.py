import argparse

from utils.config import *
from utils.saveLoader import Vocab, load_train_dataset, load_embedding_matrix

EPOCH = 15
BATCH_SIZE = 64
NUM_SAMPLES = 81625

def get_params():
    vocab = Vocab(VOCAB_PAD)
    steps_per_epoch = NUM_SAMPLES//BATCH_SIZE  # 不算多余的
    parser = argparse.ArgumentParser()

    # 调试选项
    parser.add_argument("--mode", default='train', help="run mode", type=str)
    parser.add_argument("--decode_mode", default='greedy', help="decode mode greedy/beam", type=str)
    parser.add_argument("--greedy_decode", default=True, help="if greedy decode", type=bool)
    parser.add_argument("--debug_mode", default=False, help="debug mode", type=bool)
    parser.add_argument("--beam_size", default=3,
                        help="beam size for beam search decoding (must be equal to batch size in decode mode)",
                        type=int)

    # 预处理后的参数
    parser.add_argument("--max_enc_len",
                        default=200,
                        help="Encoder input max sequence length",
                        type=int)
    parser.add_argument("--max_dec_len",
                        default=40,
                        help="Decoder input max sequence length",
                        type=int)
    parser.add_argument("--vocab_size", default=vocab.count, help="max vocab size , None-> Max ", type=int)

    # 训练参数设置
    parser.add_argument("--batch_size", default=BATCH_SIZE, help="batch size", type=int)
    parser.add_argument("--epochs", default=EPOCH, help="train epochs", type=int)
    parser.add_argument("--steps_per_epoch", default=steps_per_epoch, help="max_train_steps", type=int)
    parser.add_argument("--checkpoints_save_steps", default=2, help="Save checkpoints every N steps", type=int)
    parser.add_argument("--trained_epoch", default=0, help="trained epoch", type=int)

    # 优化器
    parser.add_argument("--learning_rate", default=0.01, help="Learning rate", type=float)
    parser.add_argument("--adagrad_init_acc", default=0.1,
                        help="Adagrad optimizer initial accumulator value. "
                             "Please refer to the Adagrad optimizer API documentation "
                             "on tensorflow site for more details.",
                        type=float)
    parser.add_argument("--max_grad_norm", default=0.8, help="Gradient norm above which gradients must be clipped",
                        type=float)

    # 模型参数
    parser.add_argument("--embed_size",
                        default=300,
                        help="Words embeddings dimension",
                        type=int)
    parser.add_argument("--enc_units", default=256, help="Encoder GRU cell units number", type=int)
    parser.add_argument("--dec_units", default=256, help="Decoder GRU cell units number", type=int)
    parser.add_argument("--attn_units", default=128, help="[context vector, decoder state, decoder input] feedforward \
                            result dimension - this result is used to compute the attention weights",
                        type=int)

    # 相关文件路径
    parser.add_argument("--vocab_path", default=VOCAB_PAD, help="vocab path", type=str)
    parser.add_argument("--train_seg_x_dir", default=TRAIN_SEG_X, help="train_seg_x_dir", type=str)
    parser.add_argument("--train_seg_y_dir", default=TRAIN_SEG_Y, help="train_seg_y_dir", type=str)
    parser.add_argument("--test_seg_x_dir", default=TEST_SEG_X, help="train_seg_x_dir", type=str)
    parser.add_argument("--test_save_dir", default=TEST_SAVE_DIR, help="load batch train data from pickle", type=str)
    parser.add_argument("--result_save_path", default=os.path.join(TEST_SAVE_DIR, "test_res.csv"),
                        help='result save path', type=str)

    # 暂时不确定有何用
    # parser.add_argument("--min_dec_steps", default=4, help="min_dec_steps", type=int)

    args = parser.parse_args()
    _params = vars(args)

    return _params


""" 使用方法：
新建一个文件xx.py
from utils.params import get_params
params = get_params()

在jupyter notebook里
%run xx.py
就能得到以 params 为名的字典
"""

if __name__ == "__main__":
    # get_params_from_dataset(check=True)
    params = get_params()
