import argparse

from utils.config import *
from utils.saveLoader import Vocab, load_train_dataset, load_embedding_matrix

EPOCH = 2
BATCH_SIZE = 64


def get_params_from_dataset(check=False):
    """
    :param check: bool 是否重新检查参数
    :return:
    """
    if check:
        print("开始check数据集参数")
        params_from_dataset = {}
        train_x, train_y, test_x = load_train_dataset()
        embedding = load_embedding_matrix()

        params_from_dataset['n_samples'] = train_x.shape[0]
        params_from_dataset['max_enc_len'] = train_x.shape[1]
        params_from_dataset['max_dec_len'] = train_y.shape[1]
        params_from_dataset['embed_size'] = embedding.shape[1]
        print("开始写入参数")
        with open(PARAMS_FROM_DATASET, mode="w", encoding="utf-8") as f:
            for k, v in params_from_dataset.items():
                f.write(k + "," + str(v) + "\n")
    else:
        params_from_dataset = {}
        for line in open(PARAMS_FROM_DATASET, "r", encoding='utf-8').readlines():
            k, v = line.strip().split(",")
            params_from_dataset[k] = int(v)

    return params_from_dataset


def get_params():
    vocab = Vocab(VOCAB_PAD)

    params_from_dataset = get_params_from_dataset()
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default='train', help="run mode", type=str)
    parser.add_argument("--decode_mode", default='greedy', help="decode mode greedy/beam", type=str)

    parser.add_argument("--max_enc_len",
                        default=params_from_dataset['max_enc_len'],
                        help="Encoder input max sequence length",
                        type=int)

    parser.add_argument("--max_dec_len",
                        default=params_from_dataset['max_dec_len'],
                        help="Decoder input max sequence length",
                        type=int)

    parser.add_argument("--batch_size", default=BATCH_SIZE, help="batch size", type=int)
    parser.add_argument("--epochs", default=EPOCH, help="train epochs", type=int)
    parser.add_argument("--vocab_path", default=VOCAB_PAD, help="vocab path", type=str)
    parser.add_argument("--learning_rate", default=0.01, help="Learning rate", type=float)
    parser.add_argument("--adagrad_init_acc", default=0.1,
                        help="Adagrad optimizer initial accumulator value. "
                             "Please refer to the Adagrad optimizer API documentation "
                             "on tensorflow site for more details.",
                        type=float)
    parser.add_argument("--max_grad_norm", default=0.8, help="Gradient norm above which gradients must be clipped",
                        type=float)

    parser.add_argument("--vocab_size", default=vocab.count, help="max vocab size , None-> Max ", type=int)

    parser.add_argument("--beam_size", default=3,
                        help="beam size for beam search decoding (must be equal to batch size in decode mode)",
                        type=int)

    parser.add_argument("--embed_size",
                        default=params_from_dataset['embed_size'],
                        help="Words embeddings dimension",
                        type=int)

    parser.add_argument("--enc_units", default=256, help="Encoder GRU cell units number", type=int)
    parser.add_argument("--dec_units", default=256, help="Decoder GRU cell units number", type=int)
    parser.add_argument("--attn_units", default=10, help="[context vector, decoder state, decoder input] feedforward \
                            result dimension - this result is used to compute the attention weights",
                        type=int)

    parser.add_argument("--train_seg_x_dir", default=TRAIN_SEG_X, help="train_seg_x_dir", type=str)
    parser.add_argument("--train_seg_y_dir", default=TRAIN_SEG_Y, help="train_seg_y_dir", type=str)
    parser.add_argument("--test_seg_x_dir", default=TEST_SEG_X, help="train_seg_x_dir", type=str)

    parser.add_argument("--checkpoints_save_steps", default=2, help="Save checkpoints every N steps", type=int)

    # 这个该如何确定？
    parser.add_argument("--min_dec_steps", default=4, help="min_dec_steps", type=int)

    max_train_steps = params_from_dataset['n_samples']//BATCH_SIZE + 1
    parser.add_argument("--max_train_steps", default=max_train_steps, help="max_train_steps", type=int)

    parser.add_argument("--train_pickle_dir", default=TRAIN_PICKLE_DIR, help="train_pickle_dir", type=str)
    parser.add_argument("--save_batch_train_data", default=False, help="save batch train data to pickle", type=bool)
    parser.add_argument("--load_batch_train_data", default=False, help="load batch train data from pickle", type=bool)

    parser.add_argument("--test_save_dir", default=TEST_SAVE_DIR, help="load batch train data from pickle", type=bool)

    # new
    parser.add_argument("--test_run", default=False, help="test trainable", type=bool)
    parser.add_argument("--greedy_decode", default=True, help="if greedy decode", type=bool)
    parser.add_argument("--result_save_path", default=os.path.join(TEST_SAVE_DIR, "test_res.csv"),
                        help='result_save_path', type=str)
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


def get_default_params():
    vocab = Vocab(VOCAB_PAD)

    params_from_dataset = get_params_from_dataset()
    max_train_steps = params_from_dataset['n_samples'] // BATCH_SIZE + 1

    _params = {'mode': 'train',
               'decode_mode': "greedy",
               'max_enc_len': params_from_dataset['max_enc_len'],
               'max_dec_len': params_from_dataset['max_dec_len'],
               'batch_size': BATCH_SIZE,
               'epochs': EPOCH,
               'vocab_path': VOCAB_PAD,
               'learning_rate': 0.01,
               'adagrad_init_acc': 0.1,
               'max_grad_norm': 0.8,
               'vocab_size': vocab.count,
               'beam_size': 3,
               'embed_size': params_from_dataset['embed_size'],
               'enc_units': 256,
               'dec_units': 256,
               'attn_units': 64,
               'train_seg_x_dir': TRAIN_SEG_X,
               'train_seg_y_dir': TRAIN_SEG_Y,

               'max_train_steps': max_train_steps,
               'train_pickle_dir': TRAIN_PICKLE_DIR,
               'save_batch_train_data': False,
               'load_batch_train_data': False,
               'test_seg_x_dir': TEST_SEG_X,
               'min_dec_steps': 4,
               'checkpoints_save_steps': 2,
               'test_save_dir': TEST_SAVE_DIR,
               'test_run': False,
               'greedy_decode': True}


    return _params


if __name__ == "__main__":
    # get_params_from_dataset(check=True)
    params = get_params()
