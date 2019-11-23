import re
import os
import jieba
import pandas as pd
import numpy as np
import time
from functools import wraps


# 装饰器
# 计算函数消耗时间的
def count_time(func):
    @wraps(func)
    def int_time(*args, **kwargs):
        start_time = time.time()  # 程序开始时间
        res = func(*args)
        over_time = time.time()   # 程序结束时间
        total_time = (over_time-start_time)
        print('程序{}()共耗时{:.2f}秒'.format(func.__name__, total_time))
        return res
    return int_time


def load_dataset(train_data_path, test_data_path):
    """
    数据数据集
    :param train_data_path:训练集路径
    :param test_data_path: 测试集路径
    :return:
    """
    # 读取数据集
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    # 空值处理
    # train_data.dropna(subset=['Question', 'Dialogue', 'Report'], how='any', inplace=True)
    # test_data.dropna(subset=['Question', 'Dialogue'], how='any', inplace=True)

    train_data = train_data.fillna('')
    test_data = test_data.fillna('')
    return train_data, test_data


def clean_sentence(sentence):
    '''
    这里我觉得用我的比较好
    特殊符号去除
    :param sentence: 待处理的字符串
    :return: 过滤特殊字符后的字符串
    '''
    if isinstance(sentence, str):

        res = re.sub(r"\(进口\)", "", sentence)

        res = re.sub(
            r'[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好|您好',
            '',
            res)

        return res
    else:
        return ''


@count_time
def get_text(file, *dataframe):
    """
    把训练集，测试集的文本拼接在一起
    :param DataFrame: 传入一个包含数个df的元组
    :return:
    """
    text = ""
    for df in dataframe:
        # 把从第三列(包括)开始的数据拼在一起
        text += "\n".join(df.iloc[:,3:].apply(lambda x:" ".join(x.to_list()), axis=1))
        # text += "<end>\n".join(df.iloc[:, 3:].apply(lambda x: " ".join(["<start>"] + x.to_list()), axis=1))

    with open(file, mode="w", encoding="utf-8") as f:
        f.write(text)

    return text

def load_stop_words(file):
    stop_words = [line.strip() for line in open(file, encoding='UTF-8').readlines()]
    return stop_words


@count_time
def create_user_dict(file, *dataframe):
    """
    创建自定义用户词典
    :param file: 存储位置
    :param DataFrame: 传入的数据集
    :return:
    """
    def process(string):

        r = re.compile("[^\u4e00-\u9fa5_a-zA-Z0-9]+|进口|海外|")
        return r.sub("", string)

    user_dict = pd.Series()
    for df in dataframe:
        user_dict = pd.concat([user_dict, df.Model, df.Brand])
    user_dict = user_dict.apply(process).unique()
    user_dict = np.delete(user_dict, np.argwhere(user_dict == ""))
    with open(file, mode="w", encoding="utf-8") as f:
        f.write("\n".join(user_dict))

    return user_dict
    # return user_dict


if __name__ == '__main__':
    # 相关已存在数据路径
    train_data_path = '../data/AutoMaster_TrainSet.csv'
    test_data_path = '../data/AutoMaster_TestSet.csv'
    stop_words_path = '../data/stopwords/哈工大停用词表.txt'

    # 生成数据路径
    raw_text_path = '../data/raw_text.txt'  # 原始文本
    user_dict_path = '../data/user_dict_new.txt'  # 自定义词典

    # 载入数据(包含了空值的处理)
    train_df, test_df = load_dataset(train_data_path, test_data_path)
    # 载入停用词
    stop_words = load_stop_words(stop_words_path)
    stop_words.extend(["(进口)"])
    # 获得原始的数据文本
    raw_text = get_text(raw_text_path, train_df, test_df)

    # todo: 创建自定义的用户词典
    user_dict = create_user_dict(user_dict_path, train_df, test_df)
    # user_dict = create_user_dict()

# todo: 完善数据预处理，如删掉(进口)
"""
    # r = re.compile(r"<start>.*\(进口\).*<end>")
    # s = r.findall(raw_text)
"""
