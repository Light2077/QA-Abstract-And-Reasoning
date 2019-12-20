import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim ,embedding_matrix , enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units # whats this
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,weights=[embedding_matrix],trainable=False)

        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        # x shape (, )
        x = self.embedding(x)
        # after embedding x shape (, )
        output, state = self.gru(x, initial_state = hidden)
        # output shape ()
        # state shape ()
        return output, state

    def initialize_hidden_state(self):
        # (一批数据大小, 隐层的维数)
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W_s = tf.keras.layers.Dense(units)
        self.W_h = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        self.W_c = tf.keras.layers.Dense(units)  # New


    def call(self, dec_hidden, enc_output, enc_pad_mask, use_coverage, prev_coverage):
        # query为上次的GRU隐藏层
        # values为编码器的编码结果enc_output
        # 在seq2seq模型中，St是后面的query向量，而编码过程的隐藏状态hi是values。
        hidden_with_time_axis = tf.expand_dims(dec_hidden, 1)

        if use_coverage and prev_coverage is not None:
            # self.W_s(values) [batch_sz, max_len, units] self.W_h(hidden_with_time_axis) [batch_sz, 1, units]
            # self.W_c(prev_coverage) [batch_sz, max_len, units]  score [batch_sz, max_len, 1]
            score = self.V(tf.nn.tanh(self.W_s(enc_output) + self.W_h(hidden_with_time_axis) + self.W_c(prev_coverage)))
            # attention_weights shape (batch_size, max_len, 1)

            mask = tf.cast(enc_pad_mask, dtype=score.dtype)
            masked_score = tf.squeeze(score, axis=-1) * mask
            masked_score = tf.expand_dims(masked_score, axis=2)

            attention_weights = tf.nn.softmax(masked_score, axis=1)
            coverage = attention_weights + prev_coverage

        else:
            # score shape == (batch_size, max_length, 1)
            # we get 1 at the last axis because we are applying score to self.V
            # the shape of the tensor before applying self.V is (batch_size, max_length, units)
            # 计算注意力权重值
            score = self.V(tf.nn.tanh(
                self.W_s(enc_output) + self.W_h(hidden_with_time_axis)))

            mask = tf.cast(enc_pad_mask, dtype=score.dtype)
            masked_score = tf.squeeze(score, axis=-1) * mask
            masked_score = tf.expand_dims(masked_score, axis=2)

            attention_weights = tf.nn.softmax(masked_score, axis=1)
            # attention_weights = masked_attention(attention_weights)
            if use_coverage:
                coverage = attention_weights

            # attention_weights sha== (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # # 使用注意力权重*编码器输出作为返回值，将来会作为解码器的输入
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, tf.squeeze(attention_weights, -1), coverage


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                                                   trainable=False)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        # 之前好像没加softmax 不需要加softmax
        # self.fc = tf.keras.layers.Dense(vocab_size, activation="softmax")  # 为了softmax层数要保持一致
        self.fc = tf.keras.layers.Dense(vocab_size, ) 

        # used for attention
        # self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output, context_vector):
        # attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)

        dec_x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)

        # output shape (batch_size, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape (batch_size, vocab)
        pred = self.fc(output)

        return dec_x, pred, state


class Pointer(tf.keras.layers.Layer):

    def __init__(self):
        super(Pointer, self).__init__()
        self.w_s_reduce = tf.keras.layers.Dense(1)  # 隐层的权重
        self.w_i_reduce = tf.keras.layers.Dense(1)  # 上下文的权重
        self.w_c_reduce = tf.keras.layers.Dense(1)  # decoder输入的权重

    def call(self, context_vector, dec_hidden, dec_inp):
        return tf.nn.sigmoid(self.w_s_reduce(dec_hidden) + self.w_c_reduce(context_vector) + self.w_i_reduce(dec_inp))