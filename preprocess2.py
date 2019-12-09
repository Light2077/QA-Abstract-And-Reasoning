"""
结合preprocess获得的预处理好的csv和word2vec的词向量模型
进行第二次数据预处理，获取能用于训练seq2seq模型的数据集

"""
import pandas as pd
import numpy as np
from gensim.models.word2vec import LineSentence
from gensim.models import word2vec
from utils.config import *
from utils.saveLoader import *


def pad(sentence, max_len, vocab_index_):
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


def get_max_len(data):
    """
    获得合适的最大长度值
    :param data: 待统计的数据  train_df['Question']
    :return: 最大长度值
    """
    max_lens = data.apply(lambda x: x.count(' '))
    return int(np.mean(max_lens) + 2 * np.std(max_lens))


def transform_data(sentence, vocab):
    # 字符串切分成词
    words = sentence.split(' ')
    # 按照vocab的index进行转换
    ids = [vocab[word] if word in vocab else unk_index for word in words]
    return ids


if __name__ == '__main__':

    vocab_index, index_vocab = load_vocab(VOCAB)

    # 预处理数据载入
    train_seg = pd.read_csv(TRAIN_SEG).fillna("")
    test_seg = pd.read_csv(TEST_SEG).fillna("")

    train_seg['X'] = train_seg[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    test_seg['X'] = test_seg[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)

    # 获取输入数据 适当的最大长度
    train_x_max_len = get_max_len(train_seg['X'])
    test_x_max_len = get_max_len(test_seg['X'])

    x_max_len = max(train_x_max_len, test_x_max_len)

    # 获取标签数据 适当的最大长度
    train_y_max_len = get_max_len(train_seg['Report'])

    # 训练集、测试集pad
    train_seg['X'] = train_seg['X'].apply(lambda x: pad(x, x_max_len, vocab_index))
    train_seg['Y'] = train_seg['Report'].apply(lambda x: pad(x, train_y_max_len, vocab_index))
    test_seg['X'] = test_seg['X'].apply(lambda x: pad(x, x_max_len, vocab_index))

    # 保存中间结果数据
    train_seg['X'].to_csv(TRAIN_X_PAD, index=None, header=False)
    train_seg['Y'].to_csv(TRAIN_Y_PAD, index=None, header=False)
    test_seg['X'].to_csv(TEST_X_PAD, index=None, header=False)

    # add retrain词向量
    if not os.path.isfile(WV_MODEL_PAD):
        print("开始retrain")

        # 词向量的载入
        wv_model = word2vec.Word2Vec.load(WV_MODEL)

        print('start retrain w2v model')
        wv_model.build_vocab(LineSentence(TRAIN_X_PAD), update=True)
        wv_model.train(LineSentence(TRAIN_X_PAD), epochs=5, total_examples=wv_model.corpus_count)
        print('1/3')
        wv_model.build_vocab(LineSentence(TRAIN_Y_PAD), update=True)
        wv_model.train(LineSentence(TRAIN_Y_PAD), epochs=5, total_examples=wv_model.corpus_count)
        print('2/3')
        wv_model.build_vocab(LineSentence(TEST_X_PAD), update=True)
        wv_model.train(LineSentence(TEST_X_PAD), epochs=5, total_examples=wv_model.corpus_count)
        print("retrain finished.")

        # 保存新的词向量模型
        wv_model.save(WV_MODEL_PAD)
    else:
        print("读取retrained的词向量模型")
        wv_model = word2vec.Word2Vec.load(WV_MODEL_PAD)

    # 更新vocab和embedding
    vocab_index = {word: index for index, word in enumerate(wv_model.wv.index2word)}
    index_vocab = {index: word for index, word in enumerate(wv_model.wv.index2word)}
    embedding_matrix = wv_model.wv.vectors

    # 保存更新后的vocab和embedding
    save_vocab(VOCAB_INDEX_PAD, vocab_index)  # vocab
    np.savetxt(EMBEDDING_MATRIX_PAD, embedding_matrix)  # embedding

    unk_index = vocab_index['<UNK>']
    # 将词转换成索引  [<START> 方向机 重 ...] -> [32800, 403, 986, 246, 231...]
    train_x = train_seg['X'].apply(lambda x: transform_data(x, vocab_index))
    train_y = train_seg['Y'].apply(lambda x: transform_data(x, vocab_index))
    test_x = test_seg['X'].apply(lambda x: transform_data(x, vocab_index))

    train_x= np.array(train_x.tolist())
    train_y = np.array(train_y.tolist())
    test_x = np.array(test_x.tolist())

    # 保存数据
    np.savetxt(TRAIN_X, train_x, fmt='%0.8f')
    np.savetxt(TRAIN_Y, train_y, fmt='%0.8f')
    np.savetxt(TEST_X, test_x, fmt='%0.8f')

