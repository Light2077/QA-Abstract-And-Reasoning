import tensorflow as tf
from pgn.layers import Encoder, Decoder, Pointer, BahdanauAttention
from utils.saveLoader import load_embedding_matrix


class PGN(tf.keras.Model):
    def __init__(self, params):
        super(PGN, self).__init__()
        self.embedding_matrix = load_embedding_matrix()
        self.params = params
        self.encoder = Encoder(params["vocab_size"],
                               params["embed_size"],
                               self.embedding_matrix,
                               params["enc_units"],
                               params["batch_size"])

        self.attention = BahdanauAttention(units=params["attn_units"])

        self.decoder = Decoder(params["vocab_size"],
                               params["embed_size"],
                               self.embedding_matrix,
                               params["dec_units"],
                               params["batch_size"],
                               self.attention)

        self.pointer = Pointer()

    #     def call_decoder_one_step(self, dec_input, dec_hidden,
    #                               enc_output, enc_extended_inp,
    #                               batch_oov_len, enc_pad_mask,
    #                               use_coverage, prev_coverage):

    #         context_vector, attentions, coverage_ret = self.attention(dec_hidden,
    #                                                                   enc_output,
    #                                                                   enc_pad_mask,
    #                                                                   use_coverage,
    #                                                                   prev_coverage)
    #         dec_x, pred, dec_hidden = self.decoder(dec_input,
    #                                                dec_hidden,
    #                                                enc_output,
    #                                                context_vector)

    #         if self.params["pointer_gen"]:
    #             p_gen = self.pointer(context_vector, dec_hidden, tf.squeeze(dec_x, axis=1))
    #             final_dists = _calc_final_dist(enc_extended_inp,
    #                                            [pred],
    #                                            [attentions],
    #                                            [p_gen],
    #                                            batch_oov_len,
    #                                            self.params["vocab_size"],
    #                                            self.params["batch_size"])
    #             return tf.stack(final_dists, 1), dec_hidden, context_vector, attentions, p_gen, coverage_ret
    #         else:
    #             return pred, dec_hidden, context_vector, attentions, None, coverage_ret
    #         # return pred, dec_hidden, context_vector, attention_weights

    def call(self, enc_inp, dec_inp,
             enc_extended_inp, batch_oov_len,
             enc_pad_mask, use_coverage=True):
        '''
        :param enc_inp:
        :param dec_inp:  tf.expand_dims(dec_inp[:, t], 1)
        :param enc_extended_inp:
        :param batch_oov_len:
        '''
        # # 用tf.TensorArray代替list
        # predictions = tf.TensorArray(tf.float32, size=dec_inp.shape[1])
        # attentions = tf.TensorArray(tf.float32, size=dec_inp.shape[1])
        # p_gens = tf.TensorArray(tf.float32, size=dec_inp.shape[1])
        # coverages = tf.TensorArray(tf.float32, size=dec_inp.shape[1])

        predictions = []
        attentions = []
        p_gens = []
        coverages = []
        # 计算encoder的输出
        enc_output, enc_hidden = self.encoder(enc_inp)
        dec_hidden = enc_hidden

        # (batch_size, enc_len, 1)
        prev_coverage = tf.zeros((enc_output.shape[0],enc_output.shape[1],1))
        for t in tf.range(dec_inp.shape[1]):
            context_vector, dec_hidden, \
            dec_x, pred, attn, prev_coverage = self.decoder(dec_inp[:, t],  # (batch_size, )
                                                       dec_hidden,  # (batch_size, dec_units)
                                                       enc_output,  # (batch_size, enc_len, enc_units)
                                                       enc_pad_mask,  # (batch_size, enc_len)
                                                       prev_coverage,
                                                       use_coverage
                                                       )

            p_gen = self.pointer(context_vector, dec_hidden, dec_x)

            # 每轮迭代后把相应数据写入TensorArray
            predictions.append(pred)
            attentions.append(attn)
            p_gens.append(p_gen)
            coverages.append(prev_coverage)

        #     predictions.write(t, pred)
        #     attentions.write(t, attn)
        #     p_gens.write(t, p_gen)
        #     coverages.write(t, prev_coverage)
        #
        # predictions = tf.transpose(predictions.stack(), perm=[1, 0, 2])
        # attentions = tf.transpose(attentions.stack(), perm=[1, 0, 2])
        # p_gens =  tf.transpose(p_gens.stack(), perm=[1, 0, 2])
        # coverages =  tf.transpose(coverages.stack(), perm=[1, 0, 2, 3])
        #
        predictions = tf.stack(predictions, axis=1)
        attentions = tf.stack(attentions, axis=1)
        p_gens = tf.stack(p_gens, axis=1)
        coverages = tf.stack(coverages, axis=1)

        coverages = tf.squeeze(coverages, -1)
        # 计算final_dist
        # 注tf.transpose()的作用是调整坐标轴顺序
        # predictions.stack() 的 shape == (dec_len, batch_size, vocab_size)
        # 执行了tf.transpose 后 shape == (batch_size, dec_len, vocab_size)
        final_dist = _calc_final_dist(enc_extended_inp,
                                      predictions,
                                      attentions,
                                      p_gens,
                                      batch_oov_len,
                                      self.params["vocab_size"],
                                      self.params["batch_size"])

        # final_dist (batch_size, dec_len, vocab_size+batch_oov_len)
        # (batch_size, dec_len, enc_len)
        return final_dist, attentions, coverages
        # return final_dist, dec_hidden, context_vector, attentions, p_gen, prev_coverage


def _calc_final_dist(_enc_batch_extend_vocab, vocab_dists, attn_dists, p_gens, batch_oov_len, vocab_size,
                     batch_size):

    # 确定的修改代码
    # 先计算公式的左半部分
    # _vocab_dists_pgn (batch_size, dec_len, vocab_size)
    _vocab_dists_pgn = vocab_dists * p_gens
    # 根据oov表的长度补齐原词表
    # _extra_zeros (batch_size, dec_len, batch_oov_len)
    if batch_oov_len !=0:
        _extra_zeros = tf.zeros((batch_size, p_gens.shape[1], batch_oov_len))
        # 拼接后公式的左半部分完成了
        # _vocab_dists_extended (batch_size, dec_len, vocab_size+batch_oov_len)
        _vocab_dists_extended = tf.concat([_vocab_dists_pgn, _extra_zeros], axis=-1)

    # 公式右半部分
    # 乘以权重后的注意力
    # _attn_dists_pgn (batch_size, dec_len, enc_len)
    _attn_dists_pgn = attn_dists * (1 - p_gens)
    # 拓展后的长度
    _extended_vocab_size = vocab_size + batch_oov_len

    # 要更新的数组 _attn_dists_pgn
    # 更新之后数组的形状与 公式左半部分一致
    # shape=[batch_size, dec_len, vocab_size+batch_oov_len]
    shape = _vocab_dists_extended.shape

    enc_len = tf.shape(_enc_batch_extend_vocab)[1]
    dec_len = tf.shape(_vocab_dists_extended)[1]

    # batch_nums (batch_size, )
    batch_nums = tf.range(0, limit=batch_size)
    # batch_nums (batch_size, 1)
    batch_nums = tf.expand_dims(batch_nums, 1)
    # batch_nums (batch_size, 1, 1)
    batch_nums = tf.expand_dims(batch_nums, 2)

    # tile 在第1,2个维度上分别复制batch_nums dec_len,enc_len次
    # batch_nums (batch_size, dec_len, enc_len)
    batch_nums = tf.tile(batch_nums, [1, dec_len, enc_len])

    # (dec_len, )
    dec_len_nums = tf.range(0, limit=dec_len)
    # (1, dec_len)
    dec_len_nums = tf.expand_dims(dec_len_nums, 0)
    # (1, dec_len, 1)
    dec_len_nums = tf.expand_dims(dec_len_nums, 2)
    # tile是用来在不同维度上复制张量的
    # dec_len_nums (batch_size, dec_len, enc_len)
    dec_len_nums = tf.tile(dec_len_nums, [batch_size, 1, enc_len])

    # _enc_batch_extend_vocab_expand (batch_size, 1, enc_len)
    _enc_batch_extend_vocab_expand = tf.expand_dims(_enc_batch_extend_vocab, 1)
    # _enc_batch_extend_vocab_expand (batch_size, dec_len, enc_len)
    _enc_batch_extend_vocab_expand = tf.tile(_enc_batch_extend_vocab_expand, [1, dec_len, 1])

    # 因为要scatter到一个3D tensor上，所以最后一维是3
    # indices (batch_size, dec_len, enc_len, 3)
    indices = tf.stack((batch_nums,
                        dec_len_nums,
                        _enc_batch_extend_vocab_expand),
                       axis=3)

    # 开始更新
    attn_dists_projected = tf.scatter_nd(indices, _attn_dists_pgn, shape)
    # 至此完成了公式的右半边

    # 计算最终分布
    final_dists = _vocab_dists_extended + attn_dists_projected

    return final_dists