import pandas as pd
import numpy as np


class Preprocess:
    """
    用于对原始数据集进行预处理
    """

    def __init__(self):
        self.train_data_path = '../data/AutoMaster_TrainSet.csv'
        self.test_data_path = '../data/AutoMaster_TestSet.csv'
        self.stop_word_path = '../data/stopwords/哈工大停用词表.txt'

        # 载入停用词和数据集
        self.stop_words = self.load_stop_words(self.stop_word_path)
        self.train_df, self.test_df = self.load_dataset(self.train_data_path, self.test_data_path)

    @staticmethod
    def load_stop_words(stop_word_path):
        """
        加载停用词
        :param stop_word_path:停用词路径
        :return: 停用词表 list
        """
        # 打开文件
        file = open(stop_word_path, 'r', encoding='utf-8')
        # 读取所有行
        stop_words = file.readlines()
        # 去除每一个停用词前后 空格 换行符
        stop_words = [stop_word.strip() for stop_word in stop_words]
        return stop_words

    @staticmethod
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
        return train_data, test_data


pre = Preprocess()