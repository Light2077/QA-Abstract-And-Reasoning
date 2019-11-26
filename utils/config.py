import os
import pathlib

root = pathlib.Path(os.path.abspath(__file__)).parent.parent

# 原有数据
train_data_path = os.path.join(root, 'data', 'AutoMaster_TrainSet.csv')
test_data_path = os.path.join(root, 'data', 'AutoMaster_TestSet.csv')
stop_words_path = os.path.join(root, 'data', 'stopwords', '哈工大停用词表.txt')

# 预处理过程中生成的数据
raw_text_path = os.path.join(root, 'data', 'raw_text.txt')  # 原始文本
proc_text_path = os.path.join(root, 'data', 'proc_text.txt')  # 预处理后的文本
user_dict_path = os.path.join(root, 'data', 'user_dict_new.txt')  # 自定义词典

train_seg_path = os.path.join(root, 'data', 'train_seg.csv')  # 预处理后的csv文件
test_seg_path = os.path.join(root, 'data', 'test_seg.csv')

# 词向量模型
save_model_path = os.path.join(root, 'data', 'wv', 'word2vec.model')
