# -*- coding:utf-8 -*-
# Created by LuoJie at 12/7/19
import tensorflow as tf

from seq2seq_tf2.model_layers import Encoder, BahdanauAttention, Decoder, Pointer
from seq2seq_tf2.seq2seq_model import Seq2Seq
from utils.config import save_wv_model_path
from utils.gpu_utils import config_gpu
from utils.wv_loader import load_embedding_matrix, Vocab


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

        self.attention = BahdanauAttention(params["attn_units"])

        self.decoder = Decoder(params["vocab_size"],
                               params["embed_size"],
                               self.embedding_matrix,
                               params["dec_units"],
                               params["batch_size"])

        self.pointer = Pointer()

    def call_encoder(self, enc_inp):

        enc_hidden = self.encoder.initialize_hidden_state()

        enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)
        return enc_output, enc_hidden

    def call_decoder_onestep(self, dec_input, dec_hidden, enc_output, enc_extended_inp, batch_oov_len):

        context_vector, attention_weights = self.attention(dec_hidden, enc_output)

        dec_x, pred, dec_hidden = self.decoder(dec_input,
                                               dec_hidden,
                                               enc_output,
                                               context_vector)
        p_gen = self.pointer(context_vector, dec_hidden, tf.squeeze(dec_x, axis=1))

        final_dists = _calc_final_dist(enc_extended_inp, [pred], [attention_weights], [p_gen], batch_oov_len,
                                       self.params["vocab_size"], self.params["batch_size"])

        return tf.stack(final_dists, 1), dec_hidden, context_vector, attention_weights, p_gen
        # return pred, dec_hidden, context_vector, attention_weights

    def call(self, dec_input, dec_hidden, enc_output,
             dec_target, enc_extended_inp, batch_oov_len,
             enc_pad_mask, use_coverage=True, prev_coverage=None):
        '''
        :param dec_input:  tf.expand_dims(dec_inp[:, t], 1)
        :param dec_hidden:
        :param enc_output:
        :param dec_target:
        :param enc_extended_inp:
        :param batch_oov_len:
        '''
        predictions = []
        attentions = []
        p_gens = []
        coverages = []

        context_vector, attention_weights, coverage_ret = self.attention(dec_hidden,
                                                                         enc_output,
                                                                         enc_pad_mask,
                                                                         use_coverage,
                                                                         prev_coverage)
        for t in range(dec_target.shape[1]):
            dec_x, pred, dec_hidden = self.decoder(tf.expand_dims(dec_input[:, t], 1),
                                                   dec_hidden,
                                                   enc_output,
                                                   context_vector)

            context_vector, attention_weights, coverage_ret = self.attention(dec_hidden,
                                                                             enc_output,
                                                                             enc_pad_mask,
                                                                             use_coverage,
                                                                             coverage_ret)

            p_gen = self.pointer(context_vector, dec_hidden, tf.squeeze(dec_x, axis=1))
            # using teacher forcing
            # dec_input = tf.expand_dims(dec_target[:, t], 1)
            coverages.append(coverage_ret)
            predictions.append(pred)
            attentions.append(attention_weights)
            p_gens.append(p_gen)

        final_dists = _calc_final_dist(enc_extended_inp,
                                       predictions,
                                       attentions,
                                       p_gens,
                                       batch_oov_len,
                                       self.params["vocab_size"],
                                       self.params["batch_size"])
        if self.params["mode"] == "train":
            # predictions_shape = (batch_size, dec_len, vocab_size)
            # with dec_len = 1 in pred mode
            return tf.stack(final_dists, 1), dec_hidden, attentions, coverages
        else:
            return tf.stack(final_dists, 1), dec_hidden, context_vector, tf.stack(attentions, 1), tf.stack(p_gens, 1)


def _calc_final_dist(_enc_batch_extend_vocab, vocab_dists, attn_dists, p_gens, batch_oov_len, vocab_size, batch_size):
    """
    Calculate the final distribution, for the pointer-generator model
    Args:
    vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays.
                The words are in the order they appear in the vocabulary file.
    attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays
    Returns:
    final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
    """
    # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
    vocab_dists = [p_gen * dist for (p_gen, dist) in zip(p_gens, vocab_dists)]
    attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(p_gens, attn_dists)]

    # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
    extended_vsize = vocab_size + batch_oov_len  # the maximum (over the batch) size of the extended vocabulary
    extra_zeros = tf.zeros((batch_size, batch_oov_len))
    # list length max_dec_steps of shape (batch_size, extended_vsize)
    vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists]

    # Project the values in the attention distributions onto the appropriate entries in the final distributions
    # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary,
    # then we add 0.1 onto the 500th entry of the final distribution
    # This is done for each decoder timestep.
    # This is fiddly; we use tf.scatter_nd to do the projection
    batch_nums = tf.range(0, limit=batch_size)  # shape (batch_size)
    batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
    attn_len = tf.shape(_enc_batch_extend_vocab)[1]  # number of states we attend over
    batch_nums = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)
    indices = tf.stack((batch_nums, _enc_batch_extend_vocab), axis=2)  # shape (batch_size, enc_t, 2)
    shape = [batch_size, extended_vsize]
    # list length max_dec_steps (batch_size, extended_vsize)
    attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists]

    # Add the vocab distributions and the copy distributions together to get the final distributions
    # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving
    # the final distribution for that decoder timestep
    # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
    final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in
                   zip(vocab_dists_extended, attn_dists_projected)]

    return final_dists


if __name__ == '__main__':
    # GPU资源配置
    config_gpu()
    # 读取vocab训练
    vocab, reverse_vocab = Vocab.load_vocab(save_wv_model_path)
    # 计算vocab size
    vocab_size = len(vocab)
    batch_size = 128
    input_sequence_len = 200

    params = {}
    params["vocab_size"] = vocab_size
    params["embed_size"] = 500
    params["enc_units"] = 512
    params["attn_units"] = 512
    params["dec_units"] = 512
    params["batch_size"] = batch_size

    model = Seq2Seq(params)

    # example_input
    example_input_batch = tf.ones(shape=(batch_size, input_sequence_len), dtype=tf.int32)

    # sample input
    sample_hidden = model.encoder.initialize_hidden_state()

    sample_output, sample_hidden = model.encoder(example_input_batch, sample_hidden)
    # 打印结果
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

    attention_layer = BahdanauAttention(10)
    context_vector, attention_weights = attention_layer(sample_hidden, sample_output)

    print("Attention context_vector shape: (batch size, units) {}".format(context_vector.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

    sample_decoder_output, _, = model.decoder(tf.random.uniform((batch_size, 1)),
                                              sample_hidden, sample_output, context_vector)

    print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
