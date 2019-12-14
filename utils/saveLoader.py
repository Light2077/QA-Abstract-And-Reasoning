import pandas as pd
import numpy as np
from utils.config import *
from utils.decorator import *

@count_time
def get_text(*dataframe, columns=["Question", "Dialogue", "Report"], concater = " "):
    """
    把训练集，测试集的文本拼接在一起

    :param dataframe: 传入一个包含数个df的元组
    :param columns: 要拼接的列
    :param concater: 怎么拼接列，默认用空格拼接
    :return:
    """
    text = ""
    for df in dataframe:
        # 过滤掉数据集没有的特征
        proc_columns = []
        for col in columns:
            if col in df.columns:
                proc_columns.append(col)

        # 把从第三列(包括)开始的数据拼在一起
        text += "\n".join(df[proc_columns].apply(lambda x: concater.join(x), axis=1))
        # text += "<end>\n".join(df.iloc[:, 3:].apply(lambda x: " ".join(["<start>"] + x.to_list()), axis=1))

    return text


def save_text(text, file):
    with open(file, mode="w", encoding="utf-8") as f:
        f.write(text)


def load_text(file):
    with open(file, mode="r", encoding="utf-8") as f:
        text = f.read()
    return text


def save_user_dict(user_dict, file):
    """
    user_dict
    :param user_dict:
    :param file:
    """
    with open(file, mode="w", encoding="utf-8") as f:
        f.write("\n".join(user_dict))



def load_dataset(train_data_path_, test_data_path_):
    """
    数据数据集
    :param train_data_path_:训练集路径
    :param test_data_path_: 测试集路径
    :return:
    """
    # 读取数据集
    train_data = pd.read_csv(train_data_path_)
    test_data = pd.read_csv(test_data_path_)

    # 空值处理
    # train_data.dropna(subset=['Question', 'Dialogue', 'Report'], how='any', inplace=True)
    # test_data.dropna(subset=['Question', 'Dialogue'], how='any', inplace=True)

    train_data = train_data.fillna('')
    test_data = test_data.fillna('')
    return train_data, test_data


def save_vocab(path, vocab_index):
    """

    :param path: 要保存的vocab文件路径
    :param vocab_index: vocab
    """
    with open(path, mode="w", encoding="utf-8") as f:
        for key, value in vocab_index.items():
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


def load_embedding_matrix(embed_path):
    return np.loadtxt(embed_path)


def load_train_dataset():
    """
    :return: 加载处理好的数据集
    """
    train_x = np.load(TRAIN_X)
    train_y = np.load(TRAIN_Y)
    test_x = np.load(TEST_X)

    return train_x, train_y, test_x


def del_all_files_of_dir(path):
    """
    删除文件夹下的所有文件
    :param path: 文件夹路径
    :return:
    """
    if os.listdir(path)==[]:
        print("there no files in this path")
        return None
    for file_name in os.listdir(path):
        file = os.path.join(path, file_name)
        print("remove file:", file_name)
        os.remove(file)
