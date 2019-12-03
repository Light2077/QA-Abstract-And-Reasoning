"""
运行此代码可以获得：
- word2vec.model
"""
from utils.config import *
import pandas as pd
from gensim.models.word2vec import LineSentence
from gensim.models import word2vec
import numpy as np
import os


def save_vocab(path, vocab_index_):
    with open(path, mode="w", encoding="utf-8") as f:
        for key, value in vocab_index_.items():
            f.write(str(key)+" ")
            f.write(str(value)+"\n")


def load_vocab(path):
    vocab_index_ = {}
    index_vocab_ = {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f.readlines():
            [vocab, index] = line.strip("\n").split(" ")
            vocab_index_[vocab] = index
            index_vocab_[index] = vocab
    return vocab_index_, index_vocab_


if __name__ == "__main__":

    proc_text = pd.read_csv(PROC_TEXT, header=None)

    retrain = False  # 是否重新训练

    if not os.path.isfile(WV_MODEL) or retrain:
        # 如果词向量模型没有保存，则开始训练词向量
        print("开始训练词向量")
        wv_model = word2vec.Word2Vec(LineSentence(PROC_TEXT), workers=12, min_count=5, size=300)

        # 建立词表
        vocab_index = {word: index for index, word in enumerate(wv_model.wv.index2word)}
        index_vocab = {index: word for index, word in enumerate(wv_model.wv.index2word)}

        # 获取词向量矩阵
        embedding_matrix = wv_model.wv.vectors

        # 保存
        wv_model.save(WV_MODEL)  # 词向量模型
        save_vocab(VOCAB_INDEX, vocab_index)  # vocab
        np.savetxt(EMBEDDING_MATRIX, embedding_matrix)  # embedding

    else:
        print("读取已训练好的词向量")
        wv_model = word2vec.Word2Vec.load(WV_MODEL)
        vocab_index, index_vocab = load_vocab(VOCAB_INDEX)
        embedding_matrix = np.loadtxt(EMBEDDING_MATRIX)

    # wv_model.wv.most_similar(['奇瑞'],topn=10)



