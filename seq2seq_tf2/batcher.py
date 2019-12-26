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


if __name__ == "__main__":
    # "<START>" 14712
    # "<UNK>" 14713
    # "<STOP>" 14714
    # "<PAD>" 14715
    words = ["方向机", "重", "助力", "泵", "谷丙转氨酶"]
    ids = [480, 1111, 14713, 288, 14714, 14715, 14715]
    vocab = Vocab(VOCAB_PAD)

    # print("sentence:", sentence)
    # print("ids_to_words: ", )
    # print("ids_to_words: ", )
    print("words:", words)
    print("words_to_ids: ", words_to_ids(words, vocab))
    print("words_to_sentence: ", words_to_sentence(words, vocab))


    print("ids:", ids)
    print("ids_to_words: ", ids_to_words(ids, vocab))
    print("ids_to_sentence: ", ids_to_sentence(ids, vocab))
