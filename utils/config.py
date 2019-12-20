import os
import pathlib
# ctrl + shift + u 一键大写
root = pathlib.Path(os.path.abspath(__file__)).parent.parent

# 原有数据

TRAIN_DATA = os.path.join(root, 'data', 'AutoMaster_TrainSet.csv')
TEST_DATA = os.path.join(root, 'data', 'AutoMaster_TestSet.csv')
STOP_WORDS = os.path.join(root, 'data', 'stopwords', 'my_stop_words.txt')

# 预处理过程中生成的数据
RAW_TEXT = os.path.join(root, 'data', 'raw_text.txt')  # 原始文本
PROC_TEXT = os.path.join(root, 'data', 'proc_text.txt')  # 预处理后的文本
USER_DICT = os.path.join(root, 'data', 'user_dict_new.txt')  # 自定义词典

TRAIN_SEG = os.path.join(root, 'data', 'train_seg.csv')  # 预处理后的csv文件
TEST_SEG = os.path.join(root, 'data', 'test_seg.csv')

# 2. pad oov处理后的数据
TRAIN_X_PAD = os.path.join(root, 'data', 'train_X_pad.csv')
TRAIN_Y_PAD = os.path.join(root, 'data', 'train_Y_pad.csv')
TEST_X_PAD = os.path.join(root, 'data', 'test_X_pad.csv')

# 训练数据
TRAIN_SEG_X = os.path.join(root, 'data', 'train_seg_x.csv')
TRAIN_SEG_Y = os.path.join(root, 'data', 'train_seg_y.csv')
TEST_SEG_X = os.path.join(root, 'data', 'test_seg_x.csv')

TRAIN_X = os.path.join(root, 'data', 'train_x.txt')
TRAIN_Y = os.path.join(root, 'data', 'train_y.txt')
TEST_X = os.path.join(root, 'data', 'test_x.txt')

# 词向量模型
WV_MODEL = os.path.join(root, 'data', 'wv', 'word2vec.model')
VOCAB = os.path.join(root, 'data', 'wv', 'vocab_index.txt')
EMBEDDING_MATRIX = os.path.join(root, 'data', 'wv', 'embedding_matrix.txt')
VOCAB_PAD = os.path.join(root, 'data', 'wv', 'vocab_index_pad.txt')
EMBEDDING_MATRIX_PAD = os.path.join(root, 'data', 'wv', 'embedding_matrix_pad.txt')
WV_MODEL_PAD = os.path.join(root, 'data', 'wv', 'word2vec_pad.model')

# seq2seq模型
CKPT_DIR = os.path.join(root, 'data', 'checkpoints', 'training_checkpoints')
CKPT_PREFIX = os.path.join(CKPT_DIR, "ckpt")

#其他
FONT = os.path.join(root, 'data', 'TrueType', 'simhei.ttf')
PARAMS_FROM_DATASET = os.path.join(root, 'data', 'params_from_dataset.txt')

# 结果
RESULT_PATH = os.path.join(root, 'data', 'result')
TRAIN_PICKLE_DIR = os.path.join(root, 'data', 'dataset')
TEST_SAVE_DIR = os.path.join(root, 'data', 'result')