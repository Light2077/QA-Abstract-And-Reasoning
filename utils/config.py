import os
import pathlib

root = pathlib.Path(os.path.abspath(__file__)).parent.parent

# 原有数据

TRAIN_DATA = os.path.join(root, 'data', 'AutoMaster_TrainSet.csv')
TEST_DATA = os.path.join(root, 'data', 'AutoMaster_TestSet.csv')
STOP_WORDS = os.path.join(root, 'data', 'stopwords', '哈工大停用词表.txt')

# 预处理过程中生成的数据
RAW_TEXT = os.path.join(root, 'data', 'raw_text.txt')  # 原始文本
PROC_TEXT = os.path.join(root, 'data', 'proc_text.txt')  # 预处理后的文本
USER_DICT = os.path.join(root, 'data', 'user_dict_new.txt')  # 自定义词典

TRAIN_SEG = os.path.join(root, 'data', 'train_seg.csv')  # 预处理后的csv文件
TEST_SEG = os.path.join(root, 'data', 'test_seg.csv')

# 词向量模型
WV_MODEL = os.path.join(root, 'data', 'wv', 'word2vec.model')
