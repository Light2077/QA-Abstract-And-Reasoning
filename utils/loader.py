def load_vocab(path):
    vocab_index_ = {}
    index_vocab_ = {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f.readlines():
            [vocab, index] = line.strip("\n").split(" ")
            vocab_index_[vocab] = index
            index_vocab_[index] = vocab
    return vocab_index_, index_vocab_
