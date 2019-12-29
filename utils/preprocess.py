"""
运行此代码可以获得：
"""
import time
from functools import wraps
from utils.word2vec import *
from utils.saveLoader import *
from utils.config import *

import re
import jieba
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from utils.config import DATASET_MSG


def count_time(func):
    """
    用于计算函数运行时间的装饰器
    :param func:
    :return:
    """
    @wraps(func)
    def int_time(*args, **kwargs):

        start_time = time.time()  # 程序开始时间
        res = func(*args, **kwargs)
        over_time = time.time()  # 程序结束时间

        total_time = (over_time - start_time)
        print('程序{}()共耗时{:.2f}秒'.format(func.__name__, total_time))
        return res

    return int_time


def create_user_dict(*dataframe):
    """
    创建自定义用户词典
    :param dataframe: 传入的数据集
    :return:
    """

    def process(sentence):
        """
        预处理sentence
        """
        r = re.compile(r"[(（]进口[)）]|\(海外\)|[^\u4e00-\u9fa5_a-zA-Z0-9]")
        return r.sub("", sentence)

    _user_dict = pd.Series()
    for df in dataframe:
        _user_dict = pd.concat([_user_dict, df.Model, df.Brand])
    _user_dict = _user_dict.apply(process).unique()
    _user_dict = np.delete(_user_dict, np.argwhere(_user_dict == ""))

    return _user_dict


# def sentence_to_words(sentence, vocab):
#     # jieba.cut(sentence)
#
#     return words
#
# def sentence_to_ids(sentence, vocab):
#
#     return ids


def words_to_ids(words, vocab):
    """
    :param words: list ["方向机", "重", "助力", "泵", "谷丙转氨酶"]
    :param vocab:
    :return: list [480, 1111, 308, 288, 14713]
    """
    unk_id = vocab.word_to_id(Vocab.UNKNOWN_TOKEN)
    ids = [unk_id if vocab.word_to_id(w)==unk_id else vocab.word_to_id(w) for w in words]
    return ids


def words_to_sentence(words, vocab):
    """

    :param words: list ["方向机", "重", "助力", "泵", "谷丙转氨酶"]
    :param vocab:
    :return: 方向机重助力泵
    """
    miss = [vocab.word_to_id(Vocab.UNKNOWN_TOKEN),
            vocab.word_to_id(Vocab.PAD_TOKEN),
            vocab.word_to_id(Vocab.START_DECODING),
            vocab.word_to_id(Vocab.STOP_DECODING)]

    sentence = ["" if vocab.word_to_id(w) in miss else w for w in words]
    sentence ="".join(sentence)

    return sentence


def ids_to_words(ids, vocab):
    return [vocab.id2word[i] for i in ids]


def ids_to_sentence(ids, vocab):
    """

    :param ids: list [480, 1111, 14713, 288, 14714, 14715, 14715]
    :param vocab:
    :return: str
    """
    start_id = vocab.word_to_id(Vocab.START_DECODING)
    stop_id = vocab.word_to_id(Vocab.STOP_DECODING)
    pad_id = vocab.word_to_id(Vocab.PAD_TOKEN)
    unk_id = vocab.word_to_id(Vocab.UNKNOWN_TOKEN)

    sentence = ""
    for i in ids:
        if i not in [start_id, stop_id, pad_id, unk_id]:
            sentence+=vocab.id2word[i]
    return sentence


def load_stop_words(file=STOP_WORDS):
    stop_words_ = [line.strip() for line in open(file, encoding='UTF-8').readlines()]
    return stop_words_


class Preprocess:
    def __init__(self):
        self.stop_words = load_stop_words(STOP_WORDS)
        print("创建一个预处理器")

    @staticmethod
    def clean_sentence(sentence):
        """
        特殊符号去除
        :param sentence: 待处理的字符串
        :return: 过滤特殊字符后的字符串
        """
        if isinstance(sentence, str):

            # 删除1. 2. 3. 这些标题
            r = re.compile("\D(\d\.)\D")
            sentence = r.sub("", sentence)

            # 删除带括号的 进口 海外
            r = re.compile(r"[(（]进口[)）]|\(海外\)")
            sentence = r.sub("", sentence)

            # 删除除了汉字数字字母和，！？。.- 以外的字符
            r = re.compile("[^，！？。\.\-\u4e00-\u9fa5_a-zA-Z0-9]")
            sentence = sentence.replace(",", "，")
            sentence = sentence.replace("!", "！")
            sentence = sentence.replace("?", "？")
            # r = re.compile("[^\.\-\u4e00-\u9fa5_a-zA-Z0-9]")

            sentence = r.sub("", sentence)

            # 删除 车主说 技师说 语音 图片
            r = re.compile(r"车主说|技师说|语音|图片")
            sentence = r.sub("", sentence)

            # r = re.compile(r"[(（]进口[)）]|\(海外\)|[^\u4e00-\u9fa5_a-zA-Z0-9]")
            # res = r.sub("", sentence)
            #

            # res = re.sub(
            #     r'[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好|您好',
            #     '',
            #     res)

            return sentence
        else:
            return ''

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
        cores = cpu_count() // 2

        print("开始并行处理，核心数{}".format(cores))

        with Pool(cores) as p:
            # 数据切分
            data_split = np.array_split(df, cores)
            # 数据分发 合并
            data = pd.concat(p.map(func, data_split))
        return data

    def get_seg_data(self, _train_df, _test_df, _reprocess=False):
        # 合并上面的操作流程，获取预处理数据集
        if not os.path.isfile(TRAIN_SEG) or _reprocess:
            print("多进程处理数据")
            _train_seg = self.parallelize(_train_df)
            _test_seg = self.parallelize(_test_df)

            print("保存数据")
            print(TRAIN_SEG, "\n", TEST_SEG)
            _train_seg.to_csv(TRAIN_SEG, index=None)
            _test_seg.to_csv(TEST_SEG, index=None)
        else:
            _train_seg, _test_seg = load_dataset(TRAIN_SEG, TEST_SEG)
        return _train_seg, _test_seg

    # 下半部分为训练，测试，评估服务的预处理
    def pad(self, sentence, max_len, _vocab):
        """
        给句子加上<START><PAD><UNK><END>
        example: "方向机 重 助力 泵..." -> "<START> 方向机 重 助力 泵...<END>"
        :param sentence: 一句话
        :param max_len: 句子的最大长度
        :param _vocab: key:词 value:index
        :return: sentence
        """

        # 0.按空格统计切分出词
        words = sentence.strip().split(' ')
        # 1. [截取]规定长度的词数
        words = words[:max_len]
        # 2. 填充< unk > ,判断是否在vocab中, 不在填充 < unk >
        sentence = [word if word in _vocab else '<UNK>' for word in words]
        # 3. 填充< start > < end >
        sentence = ['<START>'] + sentence + ['<STOP>']
        # 4. 判断长度，填充　< pad >
        sentence = sentence + ['<PAD>'] * (max_len - len(words))
        return ' '.join(sentence)

    def transform_data(self, sentence, _vocab):
        # 字符串切分成词
        words = sentence.split(' ')
        # 按照vocab的index进行转换
        ids = [_vocab[word] if word in _vocab else _vocab['<UNK>'] for word in words]
        return ids

    def get_max_len(self, data):
        """
        获得合适的最大长度值
        :param data: 待统计的数据  train_df['Question']
        :return: 最大长度值
        """
        max_lens = data.apply(lambda x: x.count(' '))
        return int(np.mean(max_lens) + 2 * np.std(max_lens))

    def sentence_proc_eval(self, sentence, max_len, _vocab):
        """
        单句话处理 ,方便测试
        """
        # 1. 切词处理
        sentence = self.sentence_proc(sentence)
        # 2. 填充
        sentence = self.pad(sentence, max_len, _vocab)
        # 3. 转换index
        sentence = self.transform_data(sentence, _vocab)
        return np.array([sentence])

    def get_train_data(self):
        print("开始为seq2seq模型进行数据预处理..")
        _vocab, _vocab_reversed = load_vocab(VOCAB)

        # 预处理数据载入
        _train_seg = pd.read_csv(TRAIN_SEG).fillna("")
        _test_seg = pd.read_csv(TEST_SEG).fillna("")

        _train_seg['X'] = _train_seg[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
        _test_seg['X'] = _test_seg[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)

        # 获取输入数据 适当的最大长度
        #train_x_max_len = self.get_max_len(_train_seg['X'])
        #test_x_max_len = self.get_max_len(_test_seg['X'])

        #x_max_len = max(train_x_max_len, test_x_max_len)
        x_max_len = 198
        # 获取标签数据 适当的最大长度
        train_y_max_len = 38
        print("输入句子最大长度：", x_max_len)
        print("输出句子最大长度：", train_y_max_len)

        # 训练集、测试集pad
        print("训练集、测试集pad")
        _train_seg['X'] = _train_seg['X'].apply(lambda x: self.pad(x, x_max_len, _vocab))
        _train_seg['Y'] = _train_seg['Report'].apply(lambda x: self.pad(x, train_y_max_len, _vocab))
        _test_seg['X'] = _test_seg['X'].apply(lambda x: self.pad(x, x_max_len, _vocab))

        # 保存中间结果数据
        print("保存pad中间结果数据")
        _train_seg['X'].to_csv(TRAIN_X_PAD, index=None, header=False)
        _train_seg['Y'].to_csv(TRAIN_Y_PAD, index=None, header=False)
        _test_seg['X'].to_csv(TEST_X_PAD, index=None, header=False)

        # add retrain词向量
        add_train = True
        if not os.path.isfile(WV_MODEL_PAD) or add_train:
            print("开始增量训练词向量")

            # 词向量的载入
            _wv_model = word2vec.Word2Vec.load(WV_MODEL)

            print('start retrain w2v model')
            _wv_model.build_vocab(LineSentence(TRAIN_X_PAD), update=True)
            _wv_model.train(LineSentence(TRAIN_X_PAD), epochs=5, total_examples=_wv_model.corpus_count)
            print('1/3')
            _wv_model.build_vocab(LineSentence(TRAIN_Y_PAD), update=True)
            _wv_model.train(LineSentence(TRAIN_Y_PAD), epochs=5, total_examples=_wv_model.corpus_count)
            print('2/3')
            _wv_model.build_vocab(LineSentence(TEST_X_PAD), update=True)
            _wv_model.train(LineSentence(TEST_X_PAD), epochs=5, total_examples=_wv_model.corpus_count)
            print("retrain finished.")

            # 保存新的词向量模型
            print("保存PAD后的词向量模型")
            _wv_model.save(WV_MODEL_PAD)
        else:
            print("读取retrained的词向量模型")
            _wv_model = word2vec.Word2Vec.load(WV_MODEL_PAD)

        # 更新vocab和embedding
        _vocab = {word: index for index, word in enumerate(_wv_model.wv.index2word)}
        _embedding_matrix = _wv_model.wv.vectors

        # 保存更新后的vocab和embedding
        print("保存更新后的vocab和embedding")
        save_vocab(VOCAB_PAD, _vocab)  # vocab
        np.savetxt(EMBEDDING_MATRIX_PAD, _embedding_matrix)  # embedding

        # 将词转换成索引  [<START> 方向机 重 ...] -> [32800, 403, 986, 246, 231...]
        train_x = _train_seg['X'].apply(lambda x: self.transform_data(x, _vocab))
        train_y = _train_seg['Y'].apply(lambda x: self.transform_data(x, _vocab))
        test_x = _test_seg['X'].apply(lambda x: self.transform_data(x, _vocab))

        train_x = np.array(train_x.tolist())
        train_y = np.array(train_y.tolist())
        test_x = np.array(test_x.tolist())

        # 保存数据
        print("保存seq2seq训练数据")
        np.savetxt(TRAIN_X, train_x, fmt="%d", delimiter=",")
        np.savetxt(TRAIN_Y, train_y, fmt="%d", delimiter=",")
        np.savetxt(TEST_X, test_x, fmt="%d", delimiter=",")

        get_seg_data()


def del_bad_sample(df):
    """
    删除低质量样本

    :param df:
    :return:
    """

    def detect_bad_words(x):
        for bad in bad_words:
            if (bad in x and len(x) <= 6):
                return True
        return False

    train = pd.read_csv(TRAIN_DATA).fillna("")
    train["QD_nstr"] = train["Question"].apply(lambda x: len(x)) + train["Dialogue"].apply(lambda x: len(x))
    train["Rp_nstr"] = train["Report"].apply(lambda x: len(x))
    bad_words = ['参照下图', '参照图片', '参照图文', '参照图文',
                 '详见图片', '长时间不回复', '如图', '按图',
                 '看图', '见图', '随时联系', '已解决', '已经提供图片',
                 '已经发图片', '还在吗', '匹配']
    train["bad_words"] = train["Report"].apply(lambda x: detect_bad_words(x))

    train["car_master"] = train["Report"].apply(lambda x: "建议您下载汽车大师APP" in x)

    bad_sample_index = train[((train["QD_nstr"] >= 400) &  # Quesetion Dialogue 字符数>=400，且
                              (train["Rp_nstr"] <= 8)) |  # Report字符数<=8(882)，或
                             train["bad_words"] |  # 回复包括bad词(643)，或
                             (train["Rp_nstr"] < 2) |  # Report字符数<2(84)，或
                             train["car_master"]  # 回复推销汽车大师app(31)
                             ].index  # 共1482

    good_df = df.copy().drop(bad_sample_index, axis=0)
    print("共删除{}个低质量样本".format(len(bad_sample_index)))
    return good_df


def get_seg_data():
    print("get_seg_data")

    # 读取数据
    _train_seg = pd.read_csv(TRAIN_SEG).fillna("")
    _test_seg = pd.read_csv(TEST_SEG).fillna("")

    # 删除低质量样本
    _train_seg = del_bad_sample(_train_seg)

    # 删除Report为空的样本
    na_idx = _train_seg[(_train_seg["Report"]=="") | (_train_seg["Report"]==" ")].index
    _train_seg.drop(na_idx, axis=0, inplace=True)

    # 构建训练集x,y 和测试集x
    _train_seg['train_seg_x'] = _train_seg[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    _test_seg['test_seg_x'] = _test_seg[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    _train_seg['train_seg_y'] = _train_seg['Report']

    # 保存
    _train_seg['train_seg_x'].to_csv(TRAIN_SEG_X, index=None)
    _train_seg['train_seg_y'].to_csv(TRAIN_SEG_Y, index=None)
    _test_seg['test_seg_x'].to_csv(TEST_SEG_X, index=None)

    print("create: ", TRAIN_SEG_X)
    print("create: ", TRAIN_SEG_Y)
    print("create: ", TEST_SEG_X)

    # 保存信息
    print("样本数量为:", _train_seg.shape[0])
    with open(file=DATASET_MSG, mode="w", encoding="utf8") as f:
        f.write("n_samples:"+str(_train_seg.shape[0])+"\n")


if __name__ == '__main__':

    """
    分三步
    - step1 进行词向量训练的预处理
    - step2 训练词向量
    - step3 进行模型训练的预处理
    """
    step1 = True  # 是否进行第一步
    reprocess = True  # 是否重新进行预处理

    step2 = True  # 是否进行第二步
    retrain = True  # 是否重新训练词向量

    step3 = True  # 是否进行第三步

    proc = Preprocess()  # 不管怎样先创建一个预处理器
    start_time = time.time()  # 计时开始
    # ------step1 进行词向量训练的预处理------
    if step1:
        print("step1 进行词向量训练的预处理")
        train_df, test_df = load_dataset(TRAIN_DATA, TEST_DATA)  # 载入数据(包含了空值的处理)
        raw_text = get_text(train_df, test_df)  # 获得原始的数据文本
        user_dict = create_user_dict(train_df, test_df)  # 创建用户自定义词典
        save_user_dict(user_dict, USER_DICT)  # 保存用户自定义词典

        # 预处理阶段

        train_seg, test_seg = proc.get_seg_data(train_df, test_df, reprocess)

        # 获取预处理后的文本，作为word2vec的训练材料 按行保存
        proc_text = get_text(train_seg, test_seg, concater="\n")

        # 保存生成的数据
        save_text(raw_text, RAW_TEXT)  # 保存原始文本
        save_text(proc_text, PROC_TEXT)  # 保存处理后的文本

    # -----step2 训练词向量-----
    if step2:
        print("step2 训练词向量")
        wv_model = get_wv_model(retrain)
        vocab, vocab_reversed = load_vocab(VOCAB)
        embedding_matrix = np.loadtxt(EMBEDDING_MATRIX)

    # -----step3 进行模型训练的预处理-----
    if step3:
        print("step3 进行模型训练的预处理")
        proc.get_train_data()

    print("共耗时{:.2f}s".format(time.time()-start_time))

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
