import tensorflow as tf
from utils.saveLoader import load_train_dataset, load_test_dataset, Vocab
from utils.config import VOCAB_PAD
from utils.config_gpu import config_gpu
from utils.params import get_params
def beam_test_batch_generator(beam_size):
    # 加载数据集
    test_x = load_test_dataset()
    for row in test_x:
        beam_search_data = tf.convert_to_tensor([row for _ in range(beam_size)])
        yield beam_search_data

if __name__ == '__main__':
    gen = beam_test_batch_generator(3)

def article_to_ids(article_words, vocab):
    ids = []
    oovs = []
    unk_id = vocab.word_to_id(Vocab.UNKNOWN_TOKEN)
    for w in article_words:
        # 正常按照流程加入词
        i = vocab.word_to_id(w)
        if i == unk_id:  # 如果发现oov词
            if w not in oovs:  # 且oov列表还没有该oov词
                oovs.append(w)  # 该oov词加入oov列表
            oov_num = oovs.index(w)  # 该句第一个oov词 oov_num=0, 第二个oov词 oov_num=1
            ids.append(vocab.size() + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs


def abstract_to_ids(abstract_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.word_to_id(Vocab.UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word_to_id(w)
        if i == unk_id:  # If w is an OOV word
            if w in article_oovs:  # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w)  # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else:  # If w is an out-of-article OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids


def output_to_words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id_to_word(i)  # might be [UNK]
        except ValueError as e:  # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. \
            This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e:  # i doesn't correspond to an article oov
                raise ValueError(
                    'Error: model produced word ID %i which corresponds \
                     to article OOV %i but this example only has %i article OOVs' % (
                        i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words


def get_dec_inp_targ_seqs(sequence, max_len, start_id, stop_id):
    """
    Given the reference summary as a sequence of tokens, return the input sequence for the decoder, and the target sequence which we will use to calculate loss. The sequence will be truncated if it is longer than max_len. The input sequence must start with the start_id and the target sequence must end with the stop_id (but not if it's been truncated).
    Args:
      sequence: List of ids (integers)
      max_len: integer
      start_id: integer
      stop_id: integer
    Returns:
      inp: sequence length <=max_len starting with start_id
      target: sequence same length as input, ending with stop_id only if there was no truncation
    """
    inp = [start_id] + sequence[:]
    target = sequence[:]
    if len(inp) > max_len:  # truncate
        inp = inp[:max_len]
        target = target[:max_len]  # no end_token
    elif len(inp) == max_len:
        target.append(stop_id)
    else:
        target.append(stop_id)  # end token
        inp.append(stop_id)  # end token
    # assert len(inp) == len(target)
    return inp, target


def get_enc_inp_targ_seqs(sequence, max_len, start_id, stop_id):
    """
    Given the reference summary as a sequence of tokens, return the input sequence for the decoder, and the target sequence which we will use to calculate loss. The sequence will be truncated if it is longer than max_len. The input sequence must start with the start_id and the target sequence must end with the stop_id (but not if it's been truncated).
    Args:
      sequence: List of ids (integers)
      max_len: integer
      start_id: integer
      stop_id: integer
    Returns:
      inp: sequence length <=max_len starting with start_id
      target: sequence same length as input, ending with stop_id only if there was no truncation
    """

    if len(sequence) >= max_len:
        inp = sequence[:max_len]
    else:
        inp = sequence[:]

    inp = [start_id] + inp + [stop_id]
    return inp

def example_generator(params, vocab):
    start_decoding = vocab.word_to_id(Vocab.START_DECODING)  # 14700
    stop_decoding = vocab.word_to_id(Vocab.STOP_DECODING)  # 14702

    if params["mode"] == "train":
        # 载入训练集的特征x和标签y
        ds_train_x = tf.data.TextLineDataset(params["train_seg_x_dir"])
        ds_train_y = tf.data.TextLineDataset(params["train_seg_y_dir"])

        # 合并训练数据
        train_dataset = tf.data.Dataset.zip((ds_train_x, ds_train_y))
        train_dataset = train_dataset.shuffle(params["batch_size"]*2+1, reshuffle_each_iteration=True).repeat()

        for raw_record in train_dataset:
            # train_x的处理
            article = raw_record[0].numpy().decode("utf-8")  # '新能源 车 最大 短板 '
            article_words = article.split()[:params["max_enc_len"]-2]  # ['新能源', '车', '最大', '短板']

            enc_input = [vocab.word_to_id(w) for w in article_words]  # [6080, 14, 1250, 14701]
            enc_input = [start_decoding] + enc_input + [stop_decoding]  # [14700, 6080, 14, 1250, 14701, 14702]

            enc_len = len(enc_input)  # 6
            enc_mask = [1 for _ in range(enc_len)]  # [1, 1, 1, 1, 1, 1]

            enc_input_extend_vocab, article_oovs = article_to_ids(article_words,
                                                                  vocab)
            enc_input_extend_vocab = [start_decoding] + enc_input_extend_vocab + [stop_decoding]


            # train_y的处理
            abstract = raw_record[1].numpy().decode("utf-8")  # '在于 充电 还有 一个 续航 里程'
            abstract_words = abstract.split()[:params["max_dec_len"]-2]  # ['在于', '充电', '还有', '一个', '续航', '里程']

            abs_ids = [vocab.word_to_id(w) for w in abstract_words]  # [4980, 939, 41, 27, 4013, 815]
            dec_input = [start_decoding] + abs_ids  # 输入讲道理不加结尾
            # dec_input = [start_decoding] + abs_ids + [stop_decoding]

            abs_ids_extend_vocab = abstract_to_ids(abstract_words, vocab, article_oovs)
            target = abs_ids_extend_vocab + [stop_decoding]  # 用来算loss讲道理不加开头
            # target = [start_decoding] + abs_ids_extend_vocab + [stop_decoding]

            # if params['pointer_gen']:
            #     abs_ids_extend_vocab = abstract_to_ids(abstract_words, vocab, article_oovs)
            #     _, target = get_dec_inp_targ_seqs(abs_ids_extend_vocab, max_dec_len, start_decoding, stop_decoding)

            dec_len = len(target)  # 7
            dec_mask = [1 for _ in range(dec_len)]  # [1, 1, 1, 1, 1, 1, 1]

            assert len(enc_input) == len(enc_input_extend_vocab), "ERROR: your code has something wrong!"

            output = {
                "enc_len": enc_len,
                "enc_input": enc_input,
                "enc_input_extend_vocab": enc_input_extend_vocab,
                "article_oovs": article_oovs,
                "dec_input": dec_input,
                "target": target,
                "dec_len": dec_len,
                "article": article,
                "abstract": abstract,
                "abstract_sents": abstract,
                "dec_mask": dec_mask,
                "enc_mask": enc_mask
            }

            yield output

    else:
        test_dataset = tf.data.TextLineDataset(params["test_seg_x_dir"])

        for raw_record in test_dataset:

            article = raw_record.numpy().decode("utf-8")  # '新能源 车 最大 短板 '
            article_words = article.split()[:params["max_enc_len"]]  # ['新能源', '车', '最大', '短板']


            enc_input = [vocab.word_to_id(w) for w in article_words]  # [6080, 14, 1250, 14701]
            enc_input = [start_decoding] + enc_input + [stop_decoding]  # [14700, 6080, 14, 1250, 14701, 14702]

            enc_len = len(enc_input)  # 6
            enc_mask = [1 for _ in range(enc_len)]  # [1, 1, 1, 1, 1, 1]

            enc_input_extend_vocab, article_oovs = article_to_ids(article_words, vocab)
            enc_input_extend_vocab = [start_decoding] + enc_input_extend_vocab + [stop_decoding]
            assert len(enc_input) == len(enc_input_extend_vocab), "ERROR: your code has something wrong!"
            output = {
                "enc_len": enc_len,
                "enc_input": enc_input,
                "enc_input_extend_vocab": enc_input_extend_vocab,
                "article_oovs": article_oovs,
                "dec_input": [],
                "target": [],
                "dec_len": params['max_dec_len']-1,  # 少个开头或结尾
                "article": article,
                "abstract": '',
                "abstract_sents": '',
                "dec_mask": [],
                "enc_mask": enc_mask
            }


            # 每一批的数据都一样阿, 是的是为了beam search
            if params["decode_mode"]=="beam":
                for _ in range(params["batch_size"]):
                    yield output
            elif params["decode_mode"]=="greedy":
                yield output
            else:
                print("shit")


def batch_generator(generator, params, vocab):
    output_types = {
        "enc_len": tf.int32,
        "enc_input": tf.int32,
        "enc_input_extend_vocab": tf.int32,
        "article_oovs": tf.string,
        "dec_input": tf.int32,
        "target": tf.int32,
        "dec_len": tf.int32,
        "article": tf.string,
        "abstract": tf.string,
        "abstract_sents": tf.string,
        "dec_mask": tf.float32,
        "enc_mask": tf.float32,
    }

    output_shapes = {
        "enc_len": [],
        "enc_input": [None],
        "enc_input_extend_vocab": [None],
        "article_oovs": [None],
        "dec_input": [None],
        "target": [None],
        "dec_len": [],
        "article": [],
        "abstract": [],
        "abstract_sents": [],
        "dec_mask": [None],
        "enc_mask": [None]
    }

    padded_shapes = {"enc_len": [],
                      "enc_input": [None],
                      "enc_input_extend_vocab": [None],
                      "article_oovs": [None],
                      "dec_input": [params['max_dec_len']-1],
                      "target": [params['max_dec_len']-1],
                      "dec_len": [],
                      "article": [],
                      "abstract": [],
                      "abstract_sents": [],
                      "dec_mask": [params['max_dec_len']-1],
                      "enc_mask": [None]
                      }

    padding_values = {"enc_len": -1,
                      "enc_input": vocab.word2id[Vocab.PAD_TOKEN],
                      "enc_input_extend_vocab": vocab.word2id[Vocab.PAD_TOKEN],
                      "article_oovs": b'',
                      "dec_input" : vocab.word2id[Vocab.PAD_TOKEN],
                      "target": vocab.word2id[Vocab.PAD_TOKEN],
                      "dec_len": -1,
                      "article": b"",
                      "abstract": b"",
                      "abstract_sents": b'',
                      "dec_mask": 0.,
                      "enc_mask": 0.
                      }

    dataset =  tf.data.Dataset.from_generator(lambda: generator(params, vocab),
                                              output_types=output_types,
                                              output_shapes=output_shapes)
    dataset = dataset.padded_batch(params["batch_size"],
                                   padded_shapes=padded_shapes,
                                   padding_values=padding_values)
    def update(entry):

        # 输出分成2个字典一个是enc的输入，一个是dec的输入
                # enc部分
        # if
        # max_oov_len
        return ({"enc_input": entry["enc_input"],
                 "extended_enc_input": entry["enc_input_extend_vocab"],
                 "article_oovs": entry["article_oovs"],
                 "enc_len": entry["enc_len"],
                 "article": entry["article"],
                 # "max_oov_len": tf.shape(entry["article_oovs"])[1],
                 "enc_mask": entry["enc_mask"]},

                {
                 "dec_input" : entry["dec_input"],
                 "dec_target": entry["target"],
                 "dec_len": entry["dec_len"],
                 "abstract": entry["abstract"],
                 "dec_mask": entry["dec_mask"]})
    dataset = dataset.map(update)
    return dataset


def batcher(vocab, params):
    # if params['mode'] == 'train' and params['load_batch_train_data']:
    #     dataset = load_batch_generator(params)
    # else:
    dataset = batch_generator(example_generator, params, vocab)
    return dataset


if __name__ == "__main__":
    # GPU资源配置
    config_gpu()
    # 获取参数
    params = get_params()
    params['mode'] = 'train'
    # vocab 对象
    vocab = Vocab(VOCAB_PAD)
    ds = batcher(vocab, params)

    batch = next(iter(ds.take(1)))

