"""
训练词向量
"""
from utils.saveLoader import *
from utils.config import *
from gensim.models.word2vec import LineSentence
from gensim.models import word2vec
from multiprocessing import cpu_count
import numpy as np
import os


def get_wv_model(retrain=False):

    # retrain 是否重新训练
    if not os.path.isfile(WV_MODEL) or retrain:
        if retrain:
            print("重新", end="")
        # 如果词向量模型没有保存，则开始训练词向量
        print("开始训练词向量")
        _wv_model = word2vec.Word2Vec(LineSentence(PROC_TEXT),
                                      workers=cpu_count(),
                                      min_count=5,
                                      sg=1,  # skip-gram
                                      size=300,
                                      iter=10,
                                      seed=1)

        # 建立词表
        _vocab = {word: index for index, word in enumerate(_wv_model.wv.index2word)}

        # 获取词向量矩阵
        _embedding_matrix = _wv_model.wv.vectors

        # 保存
        print("词向量训练完毕，保存词向量模型、Embedding matrix和vocab")
        _wv_model.save(WV_MODEL)  # 词向量模型
        save_vocab(VOCAB, _vocab)  # vocab
        np.savetxt(EMBEDDING_MATRIX, _embedding_matrix, fmt='%.9e')  # embedding

    else:
        print("读取已训练好的词向量")
        _wv_model = word2vec.Word2Vec.load(WV_MODEL)

    return _wv_model


if __name__ == "__main__":
    wv_model = get_wv_model(retrain=False)
    vocab, vocab_reversed = load_vocab(VOCAB)
    embedding_matrix = np.loadtxt(EMBEDDING_MATRIX)

    # wv_model.wv.most_similar(['奇瑞'],topn=10)
