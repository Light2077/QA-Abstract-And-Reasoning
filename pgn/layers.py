import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.use_bi_gru = True
        # 双向
        if self.use_bi_gru:
            self.enc_units = self.enc_units // 2

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                                                   trainable=False)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

        self.bi_gru = tf.keras.layers.Bidirectional(self.gru)

    def call(self, enc_input):
        # (batch_size, enc_len, embedding_dim)
        enc_input_embedded = self.embedding(enc_input)
        initial_state = self.gru.get_initial_state(enc_input_embedded)

        if self.use_bi_gru:
            # 是否使用双向GRU
            output, forward_state, backward_state = self.bi_gru(enc_input_embedded, initial_state=initial_state * 2)
            enc_hidden = tf.keras.layers.concatenate([forward_state, backward_state], axis=-1)

        else:
            # 单向GRU
            output, enc_hidden = self.gru(enc_input_embedded, initial_state=initial_state)

        return output, enc_hidden


def masked_attention(enc_padding_mask, attn_dist):
    """Take softmax of e then apply enc_padding_mask and re-normalize"""
    attn_dist = tf.squeeze(attn_dist, axis=2)
    mask = tf.cast(enc_padding_mask, dtype=attn_dist.dtype)
    attn_dist *= mask  # apply mask
    masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
    attn_dist = attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize
    attn_dist = tf.expand_dims(attn_dist, axis=2)
    return attn_dist


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W_s = tf.keras.layers.Dense(units)
        self.W_h = tf.keras.layers.Dense(units)
        self.W_c = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, dec_hidden, enc_output, enc_pad_mask, use_coverage, prev_coverage=None):
        # dec_hidden shape == (batch_size, hidden size)
        # enc_output (batch_size, enc_len, enc_units)

        # hidden_with_time_axis shape == (batch_size, 1, dec_units)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(dec_hidden, 1)

        if use_coverage and prev_coverage is not None:
            # Multiply coverage vector by w_c to get coverage_features.
            # self.W_s(values) [batch_sz, max_len, units] self.W_h(hidden_with_time_axis) [batch_sz, 1, units]
            # self.W_c(prev_coverage) [batch_sz, max_len, units]  score [batch_sz, max_len, 1]
            score = self.V(tf.nn.tanh(
                                      self.W_s(enc_output) +
                                      self.W_h(hidden_with_time_axis) +
                                      self.W_c(prev_coverage)
                                      )
                           )
            # attention_weights shape (batch_size, max_len, 1)

            # attention_weights shape (batch_size, max_length, 1)
            attention_weights = tf.nn.softmax(score, axis=1)

            # attention_weights = masked_attention(enc_pad_mask, attention_weights)
            coverage = attention_weights + prev_coverage
        else:
            # score shape == (batch_size, max_length, 1)
            # we get 1 at the last axis because we are applying score to self.V
            # the shape of the tensor before applying self.V is (batch_size, max_length, units)
            # 计算注意力权重值
            score = self.V(tf.nn.tanh(self.W_s(enc_output) + self.W_h(hidden_with_time_axis)))

            attention_weights = tf.nn.softmax(score, axis=1)
            # attention_weights = masked_attention(enc_pad_mask, attention_weights)
            if use_coverage:
                coverage = attention_weights
            else:
                coverage = []

            # # 使用注意力权重*编码器输出作为返回值，将来会作为解码器的输入
            # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, tf.squeeze(attention_weights, -1), coverage


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_sz = batch_size
        self.dec_units = dec_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                                                   trainable=False)

        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, dec_input, prev_dec_hidden, enc_output, context_vector):
        # 使用上次的隐藏层（第一次使用编码器隐藏层）、编码器输出计算注意力权重
        # enc_output shape == (batch_size, max_length, hidden_size)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        dec_input = self.embedding(dec_input)

        # 将上一循环的预测结果跟注意力权重值结合在一起作为本次的GRU网络输入
        # dec_input (batch_size, 1, embedding_dim + hidden_size)
        dec_input = tf.concat([tf.expand_dims(context_vector, 1), dec_input], axis=-1)

        # passing the concatenated vector to the GRU
        dec_output, dec_hidden = self.gru(dec_input)

        # dec_output shape == (batch_size * 1, hidden_size)
        dec_output = tf.reshape(dec_output, (-1, dec_output.shape[2]))

        # pred shape == (batch_size, vocab)
        pred = self.fc(dec_output)
        return dec_input, pred, dec_hidden

class Pointer(tf.keras.layers.Layer):

    def __init__(self):
        super(Pointer, self).__init__()
        self.w_s_reduce = tf.keras.layers.Dense(1)
        self.w_i_reduce = tf.keras.layers.Dense(1)
        self.w_c_reduce = tf.keras.layers.Dense(1)

    def __call__(self, context_vector, dec_hidden, dec_inp):
        return tf.nn.sigmoid(self.w_s_reduce(dec_hidden) +
                             self.w_c_reduce(context_vector) +
                             self.w_i_reduce(dec_inp))

