from utils.config import *
import pandas as pd
from gensim.models.word2vec import LineSentence
from gensim.models import word2vec
# import gensim
import os

if __name__ == "__main__":

    proc_text = pd.read_csv(PROC_TEXT, header=None)

    retrain = False  # 是否重新训练

    if not os.path.isfile(WV_MODEL) or retrain:
        # 如果词向量模型没有保存，则开始训练词向量
        print("开始训练词向量")
        wv_model = word2vec.Word2Vec(LineSentence(PROC_TEXT), workers=12, min_count=5, size=300)
        wv_model.save(WV_MODEL)
    else:
        print("读取已训练好的词向量")
        wv_model = word2vec.Word2Vec.load(WV_MODEL)

    # wv_model.wv.most_similar(['奇瑞'],topn=10)
    embedding_matrix = wv_model.wv.vectors  # 获取词向量矩阵

    # 建立词表
    vocab_index = {word: index for index, word in enumerate(wv_model.wv.index2word)}
    index_vocab = {index: word for index, word in enumerate(wv_model.wv.index2word)}
