import tensorflow as tf
from seq2seq_tf2.layers import Encoder, Decoder, BahdanauAttention
from utils.saveLoader import load_embedding_matrix


class Seq2Seq(tf.keras.Model):
    def __init__(self, params):
        super(Seq2Seq, self).__init__()
        self.embedding_matrix = load_embedding_matrix()
        self.params = params
        self.encoder = Encoder(vocab_size = params["vocab_size"],
                               embedding_dim = params["embed_size"],
                               embedding_matrix = self.embedding_matrix,
                               enc_units = params["enc_units"],
                               batch_size = params["batch_size"])

        self.attention = BahdanauAttention(units = params["attn_units"])

        self.decoder = Decoder(vocab_size =  params["vocab_size"],
                               embedding_dim = params["embed_size"],
                               embedding_matrix = self.embedding_matrix,
                               dec_units = params["dec_units"],
                               batch_size = params["batch_size"])

    def call_decoder_onestep(self, dec_input, dec_hidden, enc_output):
        # context_vector ()
        # attention_weights ()
        context_vector, attention_weights = self.attention(dec_hidden, enc_output)

        # pred ()
        pred, dec_hidden = self.decoder(dec_input,
                                        None,
                                        None,
                                        context_vector)
        return pred, dec_hidden, context_vector, attention_weights

    def call(self, dec_input, dec_hidden, enc_output, dec_target):
        predictions = []
        attentions = []

        context_vector, _ = self.attention(dec_hidden, enc_output)

        for t in range(1, dec_target.shape[1]):
            pred, dec_hidden = self.decoder(dec_input,
                                            dec_hidden,
                                            enc_output,
                                            context_vector)

            context_vector, attn = self.attention(dec_hidden, enc_output)
            # using teacher forcing
            dec_input = tf.expand_dims(dec_target[:, t], 1)

            predictions.append(pred)
            attentions.append(attn)
        return tf.stack(predictions, 1), dec_hidden