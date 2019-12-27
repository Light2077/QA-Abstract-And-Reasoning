import tensorflow as tf
from utils.saveLoader import load_train_dataset, load_test_dataset, Vocab
from utils.config import VOCAB_PAD

def train_batch_generator(batch_size, sample_sum=None):
    # 加载数据集
    # train_X, train_Y = load_train_dataset(max_enc_len, max_dec_len)
    train_x, train_y = load_train_dataset()

    # 只选部分样本来测试代码
    if sample_sum:
        train_x = train_x[:sample_sum]
        train_y = train_y[:sample_sum]

    dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(batch_size*2 + 1)
    dataset = dataset.batch(batch_size)  # drop_remainder=True

    steps_per_epoch = len(train_x) // batch_size
    return dataset, steps_per_epoch


def beam_test_batch_generator(beam_size):
    # 加载数据集
    test_x = load_test_dataset()
    for row in test_x:
        beam_search_data = tf.convert_to_tensor([row for _ in range(beam_size)])
        yield beam_search_data

if __name__ == '__main__':
    gen = beam_test_batch_generator(3)


# todo: 预处理数据时不应该截断句子，而是在载入数据集的时候截断
def get_dec_inp_targ_seqs(sequence, max_len, start_id, stop_id):
    """
    Given the reference summary as a sequence of tokens, return the input sequence for the decoder,
    and the target sequence which we will use to calculate loss.
    The sequence will be truncated if it is longer than max_len.
    The input sequence must start with the start_id and the target sequence must end with the stop_id (but not if it's been truncated).
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
    else:  # no truncation
        target.append(stop_id)  # end token
    assert len(inp) == len(target)
    return inp, target


def example_generator(params, vocab):
    start_decoding = vocab.word_to_id(Vocab.START_DECODING)  # 14700
    stop_decoding = vocab.word_to_id(Vocab.STOP_DECODING)  # 14702

    if params["mode"] == "train":
        # 载入训练集的特征x和标签y
        ds_train_x = tf.data.TextLineDataset(params["train_seg_x_dir"])
        ds_train_y = tf.data.TextLineDataset(params["train_seg_y_dir"])

        # 合并训练数据
        train_dataset = tf.data.Dataset.zip((ds_train_x, ds_train_y))
        train_dataset = train_dataset.shuffle(params["batch_size"]*10+1, reshuffle_each_iteration=True).repeat()

        for raw_record in train_dataset:
            # train_x的处理
            article = raw_record[0].numpy().decode("utf-8")  # '新能源 车 最大 短板 '
            article_words = article.split()[:params["max_enc_len"]]  # ['新能源', '车', '最大', '短板']

            enc_input = [vocab.word_to_id(w) for w in article_words]  # [6080, 14, 1250, 14701]
            enc_input = [start_decoding]+ enc_input + [stop_decoding]  # [14700, 6080, 14, 1250, 14701, 14702]
            enc_len = len(enc_input)  # 6
            sample_encoder_pad_mask = [1 for _ in range(enc_len)]  # [1, 1, 1, 1, 1, 1]


            # train_y的处理
            abstract = raw_record[1].numpy().decode("utf-8")  # '在于 充电 还有 一个 续航 里程'
            abstract_words = abstract.split()  # ['在于', '充电', '还有', '一个', '续航', '里程']
            abs_ids = [vocab.word_to_id(w) for w in abstract_words]  # [4980, 939, 41, 27, 4013, 815]


            dec_input, target = get_dec_inp_targ_seqs(abs_ids, params["max_dec_len"], start_decoding, stop_decoding)
            # dec_input [14700, 4980, 939, 41, 27, 4013, 815]
            # target [4980, 939, 41, 27, 4013, 815, 14702]

            dec_len = len(dec_input)  # 7
            sample_decoder_pad_mask = [1 for _ in range(dec_len)]  # [1, 1, 1, 1, 1, 1, 1]


            output = {
                "article": article,  # '新能源 车 最大 短板 '
                "enc_input": enc_input,  # [14700, 6080, 14, 1250, 14701, 14702]
                "sample_encoder_pad_mask": sample_encoder_pad_mask,  # [1, 1, 1, 1, 1, 1]
                "enc_len": enc_len,  # 6

                "abstract": abstract,  # '在于 充电 还有 一个 续航 里程'
                "dec_input": dec_input,  # [14700, 4980, 939, 41, 27, 4013, 815]
                "target": target,  # [4980, 939, 41, 27, 4013, 815, 14702]
                "sample_decoder_pad_mask": sample_decoder_pad_mask,  # [1, 1, 1, 1, 1, 1, 1]
                "dec_len": dec_len,  # 7
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
            sample_encoder_pad_mask = [1 for _ in range(enc_len)]  # [1, 1, 1, 1, 1, 1]

            output = {"article": article,  # '新能源 车 最大 短板 '
                      "enc_input": enc_input,  # [14700, 6080, 14, 1250, 14701, 14702]
                      "sample_encoder_pad_mask": sample_encoder_pad_mask,  # [1, 1, 1, 1, 1, 1]
                      "enc_len": enc_len,  # 6

                      "abstract": "",
                      "dec_input": [],
                      "target": [],
                      "sample_decoder_pad_mask": [],
                      "dec_len": params["max_dec_len"],  # 7
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
    output_types={
        "article": tf.string,  # '新能源 车 最大 短板 '
        "enc_input": tf.int32,  # [14700, 6080, 14, 1250, 14701, 14702]
        "sample_encoder_pad_mask": tf.int32,  # [1, 1, 1, 1, 1, 1]
        "enc_len": tf.int32,  # 6

        "abstract": tf.string,  # '在于 充电 还有 一个 续航 里程'
        "dec_input": tf.int32,  # [14700, 4980, 939, 41, 27, 4013, 815]
        "target": tf.int32,  # [4980, 939, 41, 27, 4013, 815, 14702]
        "sample_decoder_pad_mask": tf.int32,  # [1, 1, 1, 1, 1, 1, 1]
        "dec_len": tf.int32,  # 7
    }

    output_shapes={
        "article": [],  # 不限
        "enc_input": [None],  # 长度不限
        "sample_encoder_pad_mask": [None],  # 不限
        "enc_len": [],  # 不限

        "abstract": [],  # 不限
        "dec_input": [None],  # 长度不限
        "target": [None],  # 不限
        "sample_decoder_pad_mask": [None],  # 不限
        "dec_len": [],
    }

    padded_shapes={
        "article": [],  # 不限
        "enc_input": [None],  # 以最长的为准
        "sample_encoder_pad_mask": [None],  # 以最长的为准
        "enc_len": [],  # 不限

        "abstract": [],  # 不限
        "dec_input": [params['max_dec_len']],  # 统一dec长度
        "target": [params['max_dec_len']],  # 统一dec长度
        "sample_decoder_pad_mask": [params['max_dec_len']],  # 统一dec长度
        "dec_len": [],
    }

    padding_values={
        "article": b"",  #
        "enc_input": vocab.word2id[Vocab.PAD_TOKEN],  # 用pad的id来填充
        "sample_encoder_pad_mask": 0,  # 多余的用0填充
        "enc_len": -1,  # 不限

        "abstract": b"",  # 不限
        "dec_input": vocab.word2id[Vocab.PAD_TOKEN],  # 用pad的id来填充
        "target": vocab.word2id[Vocab.PAD_TOKEN],  # 用pad的id来填充
        "sample_decoder_pad_mask": 0,  # 多余的用0填充
        "dec_len": -1,
    }
    dataset =  tf.data.Dataset.from_generator(lambda: generator(params, vocab),
                                              output_types=output_types,
                                              output_shapes=output_shapes)
    dataset = dataset.padded_batch(params["batch_size"],
                                   padded_shapes=padded_shapes,
                                   padding_values=padding_values,
                                  drop_remainder=True)
    def update(entry):
        return ({"article": entry["article"],
                 "enc_input": entry["enc_input"],
                 "sample_encoder_pad_mask": entry["sample_encoder_pad_mask"],
                 "enc_len": entry["enc_len"],},

                {"abstract": entry["abstract"],
                 "dec_input": entry["dec_input"],
                 "target": entry["target"],
                 "dec_len": entry["dec_len"],
                 "sample_decoder_pad_mask": entry["sample_decoder_pad_mask"]})
    dataset = dataset.map(update)
    return dataset


def batcher(vocab, params):
    # if params['mode'] == 'train' and params['load_batch_train_data']:
    #     dataset = load_batch_generator(params)
    # else:
    dataset = batch_generator(example_generator, params, vocab)
    return dataset


if __name__ == "__main__":
    pass
    # "<START>" 14712
    # "<UNK>" 14713
    # "<STOP>" 14714
    # "<PAD>" 14715
    # words = ["方向机", "重", "助力", "泵", "谷丙转氨酶"]
    # ids = [480, 1111, 14713, 288, 14714, 14715, 14715]
    # vocab = Vocab(VOCAB_PAD)
    #
    # # print("sentence:", sentence)
    # # print("ids_to_words: ", )
    # # print("ids_to_words: ", )
    # print("words:", words)
    # print("words_to_ids: ", words_to_ids(words, vocab))
    # print("words_to_sentence: ", words_to_sentence(words, vocab))
    #
    #
    # print("ids:", ids)
    # print("ids_to_words: ", ids_to_words(ids, vocab))
    # print("ids_to_sentence: ", ids_to_sentence(ids, vocab))
