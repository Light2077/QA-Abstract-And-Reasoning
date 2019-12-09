"""
运行此代码可以获得：
- raw_text.txt 原始文本
- user_dict_new.txt 用户自定义词典
- test_seg.csv 预处理分词后的测试集
- train_seg.csv 预处理分词后的训练集
- proc_text.csc 预处理后的文本
"""
from utils.saveLoader import *
from utils.decorator import *
from utils.config import *

import re
import jieba
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count


def create_user_dict(*dataframe):
    """
    创建自定义用户词典
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

    _user_dict = pd.Series()
    for df in dataframe:
        _user_dict = pd.concat([_user_dict, df.Model, df.Brand])
    _user_dict = _user_dict.apply(process).unique()
    _user_dict = np.delete(_user_dict, np.argwhere(_user_dict == ""))

    return _user_dict


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


if __name__ == '__main__':

    # 初始化
    train_df, test_df = load_dataset(TRAIN_DATA, TEST_DATA)  # 载入数据(包含了空值的处理)
    raw_text = get_text(train_df, test_df)  # 获得原始的数据文本
    save_text(raw_text, RAW_TEXT)

    user_dict = create_user_dict(train_df, test_df)  # 创建用户自定义词典
    save_user_dict(user_dict, USER_DICT)

    # 预处理阶段
    reprocess = False  # 是否重新预处理
    if not os.path.isfile(TRAIN_SEG) or reprocess:
        proc = Preprocess()  # 创建个预处理类
        print("多进程处理数据")
        train_seg = proc.parallelize(train_df)
        test_seg = proc.parallelize(test_df)

        train_seg.to_csv(TRAIN_SEG, index=None)
        test_seg.to_csv(TEST_SEG, index=None)
    else:
        train_seg, test_seg = load_dataset(TRAIN_SEG, TEST_SEG)

    # 保存预处理后的文本，作为word2vec的训练材料
    proc_text = get_text(train_seg, test_seg)
    save_text(raw_text, PROC_TEXT)

# todo: 完善数据预处理，如删掉(进口)
"""
# r = re.compile(r"<start>.*(进口).*<end>")
# s = r.findall(raw_text)
a = train_df.loc[2, "Question"]
"""
# todo: 第一次运行跟最后一次运行应该有所区别
# todo: 优化user_dict 优化clean 优化stop_words
"""
2019.11.26
修复了多线程运行时，jieba用户自定义词典无效的bug
2019.12.09
重构preprocess的代码
"""
