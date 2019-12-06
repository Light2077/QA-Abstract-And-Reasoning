import re
import jieba
import pandas as pd
import numpy as np
import time
from functools import wraps
from multiprocessing import Pool, cpu_count
from utils.config import *


def count_time(func):
    @wraps(func)
    def int_time(*args, **kwargs):
        start_time = time.time()  # 程序开始时间
        res = func(*args, **kwargs)
        over_time = time.time()  # 程序结束时间
        total_time = (over_time - start_time)
        print('程序{}()共耗时{:.2f}秒'.format(func.__name__, total_time))
        return res

    return int_time


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


@count_time
def get_text(*dataframe, file=""):
    """
    把训练集，测试集的文本拼接在一起
    :param file: 若为空则不保存文件
    :param dataframe: 传入一个包含数个df的元组
    :return:
    """
    text = ""
    for df in dataframe:
        # 把从第三列(包括)开始的数据拼在一起
        text += "\n".join(df.iloc[:, 3:].apply(lambda x: " ".join(x.to_list()), axis=1))
        # text += "<end>\n".join(df.iloc[:, 3:].apply(lambda x: " ".join(["<start>"] + x.to_list()), axis=1))

    if file is not "":
        with open(file, mode="w", encoding="utf-8") as f:
            f.write(text)

    return text


@count_time
def create_user_dict(file, *dataframe):
    """
    创建自定义用户词典
    :param file: 存储位置
    :param dataframe: 传入的数据集
    :return:
    """

    def process(string):
        """
        预处理sentence
        :param string:
        :return:
        """
        r = re.compile(r"[(（]进口[)）]|\(海外\)|[^\u4e00-\u9fa5_a-zA-Z0-9]")
        return r.sub("", string)

    user_dict_ = pd.Series()
    for df in dataframe:
        user_dict_ = pd.concat([user_dict_, df.Model, df.Brand])
    user_dict_ = user_dict_.apply(process).unique()
    user_dict_ = np.delete(user_dict_, np.argwhere(user_dict_ == ""))

    # 保存user_dict
    with open(file, mode="w", encoding="utf-8") as f:
        f.write("\n".join(user_dict_))

    return user_dict_


class Preprocess:
    def __init__(self):
        self.stop_words_path = '../data/stopwords/哈工大停用词表.txt'
        self.stop_words = self.load_stop_words(self.stop_words_path)

    @staticmethod
    def load_stop_words(file):
        stop_words_ = [line.strip() for line in open(file, encoding='UTF-8').readlines()]
        return stop_words_

    @staticmethod
    def clean_sentence(sentence):
        """
        特殊符号去除
        :param sentence: 待处理的字符串
        :return: 过滤特殊字符后的字符串
        """
        if isinstance(sentence, str):
            r = re.compile(r"[(（]进口[)）]|\(海外\)|[^\u4e00-\u9fa5_a-zA-Z0-9]")
            res = r.sub("", sentence)

            r = re.compile(r"车主说|技师说|语音|图片|你好|您好")
            res = r.sub("", res)

            # res = re.sub(
            #     r'[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好|您好',
            #     '',
            #     res)

            return res
        else:
            return ''

    # 过滤停用词
    def filter_stopwords(self, words):
        """
        过滤停用词
        :param words: 切好词的列表 [word1 ,word2 .......]
        :return: 过滤后的停用词
        """
        return [word for word in words if word not in self.stop_words]

    def sentence_proc(self, sentence):
        """
        预处理模块
        :param sentence:待处理字符串
        :return: 处理后的字符串
        """
        sentence = sentence.upper()
        # 清除无用词
        sentence = self.clean_sentence(sentence)
        # 切词，默认精确模式，全模式cut参数cut_all=True
        words = jieba.cut(sentence)
        # 过滤停用词
        words = self.filter_stopwords(words)

        return ' '.join(words)

    @count_time
    def data_frame_proc(self, df):
        """
        数据集批量处理方法
        :param df: 数据集
        :return:处理好的数据集
        """
        # 批量预处理 训练集和测试集
        jieba.load_userdict(USER_DICT)
        for col_name in ['Brand', 'Model', 'Question', 'Dialogue']:
            df[col_name] = df[col_name].apply(self.sentence_proc)

        if 'Report' in df.columns:
            # 训练集 Report 预处理
            df['Report'] = df['Report'].apply(self.sentence_proc)
        return df

    @count_time
    def parallelize(self, df):
        """
        多核并行处理模块
        有个问题： 多线程处理时，jieba的载入自定义词典失效
        :param df: DataFrame数据
        :return: 处理后的数据
        """
        func = self.data_frame_proc
        cores = cpu_count()

        print("开始并行处理，核心数{}".format(cores))

        with Pool(cores) as p:
            # 数据切分
            data_split = np.array_split(df, cores)
            # 数据分发 合并
            data = pd.concat(p.map(func, data_split))
        return data

    # 为训练，测试，评估服务的预处理
    def pad(self, sentence, max_len, vocab_index_):
        """
        给句子加上<START><PAD><UNK><END>
        example:
        sentence: "方向机 重 助力 泵..."

        :param sentence: 一句话
        :param max_len: 句子的最大长度
        :param vocab_index_: key:词 value:index

        :return:"<START> 方向机 重 助力 泵...<END>"
        """

        # 0.按空格统计切分出词
        words = sentence.strip().split(' ')
        # 1. [截取]规定长度的词数
        words = words[:max_len]
        # 2. 填充< unk > ,判断是否在vocab中, 不在填充 < unk >
        sentence = [word if word in vocab_index_ else '<UNK>' for word in words]
        # 3. 填充< start > < end >
        sentence = ['<START>'] + sentence + ['<STOP>']
        # 4. 判断长度，填充　< pad >
        sentence = sentence + ['<PAD>'] * (max_len - len(words))
        return ' '.join(sentence)

    def transform_data(self, sentence, vocab):
        # 字符串切分成词
        words = sentence.split(' ')
        # 按照vocab的index进行转换
        ids = [vocab[word] if word in vocab else vocab['<UNK>'] for word in words]
        return ids

    def sentence_proc_eval(self, sentence, max_len, vocab):
        """
        单句话处理 ,方便测试
        """
        # 1. 切词处理
        sentence = self.sentence_proc(sentence)
        # 2. 填充
        sentence = self.pad(sentence, max_len, vocab)
        # 3. 转换index
        sentence = self.transform_data(sentence, vocab)
        return np.array([sentence])
