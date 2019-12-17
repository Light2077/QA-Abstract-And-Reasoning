# -*- coding:utf-8 -*-
import tensorflow as tf

from batcher import batcher
from loss import coverage_loss
from seq2seq import Seq2Seq
from utils.config import VOCAB_PAD
from utils.config_gpu import config_gpu
import time
import gc
import pickle


def train_model(model, vocab, params, checkpoint_manager):
    epochs = params['epochs']
    # batch_size = params['batch_size']
    # max_dec_len = params['max_dec_len']
    # max_enc_len = params['max_enc_len']

    optimizer = tf.keras.optimizers.Adagrad(params['learning_rate'],
                                            initial_accumulator_value=params['adagrad_init_acc'],
                                            clipnorm=params['max_grad_norm'])
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    # 定义损失函数
    def loss_function(real, pred, padding_mask):
        loss = 0
        for t in range(real.shape[1]):
            if padding_mask:
                loss_ = loss_object(real[:, t], pred[:, t, :])
                mask = tf.cast(padding_mask[:, t], dtype=loss_.dtype)
                loss_ *= mask
                loss_ = tf.reduce_mean(loss_, axis=0)  # batch-wise
                loss += loss_
            else:
                loss_ = loss_object(real[:, t], pred[:, t, :])
                loss_ = tf.reduce_mean(loss_, axis=0)  # batch-wise
                loss += loss_
        return tf.reduce_mean(loss)

    # 训练
    @tf.function
    def train_step(enc_inp, extended_enc_input, max_oov_len,
                   dec_input, dec_target, cov_loss_wt,
                   enc_pad_mask, padding_mask=None):
        batch_loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = model.call_encoder(enc_inp)

            # 第一个隐藏层输入
            dec_hidden = enc_hidden
            # 逐个预测序列
            predictions, _, attentions, coverages = model(dec_input,
                                                          dec_hidden,
                                                          enc_output,
                                                          dec_target,
                                                          extended_enc_input,
                                                          max_oov_len,
                                                          enc_pad_mask=enc_pad_mask,
                                                          use_coverage=True,
                                                          prev_coverage=None)

            # print('dec_target is :{}'.format(dec_target))
            # print('predictions is :{}'.format(predictions.shape))
            # print('dec_target is :{}'.format(dec_target.shape))
            # print('padding_mask is :{}'.format(padding_mask.shape))
            # # [max_y_len,batch size ,max_x_len]
            # print('attentions is :{}'.format(attentions))
            # # [max_y_len,batch size ,max_x_len,1]
            # print('coverages is :{}'.format(coverages))
            # batch_loss = loss_function(dec_target, predictions, padding_mask)

            # l_loss = loss_function(dec_target, predictions, padding_mask)
            # print('l_loss :{}'.format(l_loss))
            # c_loss = coverage_loss(attentions, coverages, padding_mask)
            # print('c_loss :{}'.format(c_loss))

            batch_loss = loss_function(dec_target, predictions, padding_mask) + \
                         cov_loss_wt * coverage_loss(attentions, coverages, padding_mask)

            variables = model.encoder.trainable_variables + model.decoder.trainable_variables + \
                        model.attention.trainable_variables + model.pointer.trainable_variables

            gradients = tape.gradient(batch_loss, variables)

            optimizer.apply_gradients(zip(gradients, variables))

            return batch_loss

    for epoch in range(epochs):
        start = time.time()
        dataset = batcher(vocab, params)
        total_loss = 0
        step = 0
        for encoder_batch_data, decoder_batch_data in dataset:
            # print('batch[0]["enc_input"] is ', batch[0]["enc_input"])
            # print('batch[0]["extended_enc_input"] is ', batch[0]["extended_enc_input"])
            # print('batch[1]["dec_input"] is ', batch[1]["dec_input"])
            # print('batch[1]["dec_target"] is ', batch[1]["dec_target"])
            # print('batch[0]["max_oov_len"] is ', batch[0]["max_oov_len"])
            batch_loss = train_step(encoder_batch_data["enc_input"],
                                    encoder_batch_data["extended_enc_input"],
                                    encoder_batch_data["max_oov_len"],
                                    decoder_batch_data["dec_input"],
                                    decoder_batch_data["dec_target"],
                                    cov_loss_wt=0.5,
                                    enc_pad_mask=encoder_batch_data["sample_encoder_pad_mask"],
                                    padding_mask=decoder_batch_data["sample_decoder_pad_mask"])

            # batch_loss = train_step(inputs, target)
            total_loss += batch_loss

            step += 1

            if step % 1 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             step,
                                                             batch_loss.numpy()))

        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            ckpt_save_path = checkpoint_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))
            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                total_loss / step))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

            if step > params['max_train_steps']:
                break
