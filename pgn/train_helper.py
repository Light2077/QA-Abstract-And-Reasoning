# -*- coding:utf-8 -*-
import tensorflow as tf
import time
import os
import numpy as np
from utils.config import PGN_CKPT
from pgn.batcher import batcher
from pgn.loss import calc_loss

def get_train_msg(ckpt):
    # 获得已训练的轮次
    path = os.path.join(ckpt, "trained_epoch.txt")
    with open(path, mode="r", encoding="utf-8") as f:
        trained_epoch = int(f.read())
    return trained_epoch


def save_train_msg(ckpt, trained_epoch):
    # 保存训练信息（已训练的轮数）
    path = os.path.join(ckpt, "trained_epoch.txt")
    with open(path, mode="w", encoding="utf-8") as f:
        f.write(str(trained_epoch))


def train_model(model, vocab, params, checkpoint_manager):

    epochs = params['epochs']

    optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])

    # @tf.function(input_signature=(tf.TensorSpec(shape=[params["batch_size"], None], dtype=tf.int32),
    #                               tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]-1], dtype=tf.int32),
    #                               tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]-1], dtype=tf.int32),
    #                               tf.TensorSpec(shape=[params["batch_size"], None], dtype=tf.int32),
    #                               tf.TensorSpec(shape=[], dtype=tf.int32),
    #                               tf.TensorSpec(shape=[params["batch_size"], None], dtype=tf.float32),
    #                               tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]-1], dtype=tf.float32),
    #                               tf.TensorSpec(shape=[], dtype=tf.float32)))
    def train_step(target, enc_inp, dec_inp, enc_extended_inp,
                   batch_oov_len, enc_mask, dec_mask, cov_loss_wt):

        with tf.GradientTape() as tape:

            final_dist, attentions, coverages = model(enc_inp, dec_inp, enc_extended_inp,
                                                        batch_oov_len, enc_mask)

            batch_loss, log_loss, cov_loss = calc_loss(target, final_dist,
                                                       dec_mask, attentions,
                                                       coverages, cov_loss_wt)
        variables = model.trainable_variables
        gradients = tape.gradient(batch_loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss, log_loss, cov_loss

    dataset = batcher(vocab, params)
    steps_per_epoch =params["steps_per_epoch"]

    for epoch in range(epochs):
        start = time.time()
        total_loss = total_log_loss = total_cov_loss = 0

        for (batch, (enc_data, dec_data)) in enumerate(dataset.take(steps_per_epoch)):

            # 以防万一，传进去的参数全为tensor
            cov_loss_wt = tf.cast(params["cov_loss_wt"], dtype=tf.float32)

            try:
                batch_oov_len =  tf.shape(enc_data["article_oovs"])[1]
            except:
                batch_oov_len = tf.constant(0)

            batch_loss, log_loss, cov_loss = train_step(dec_data["dec_target"],
                                                        enc_data["enc_input"],
                                                        dec_data["dec_input"],
                                                        enc_data["extended_enc_input"],
                                                        batch_oov_len,
                                                        enc_data["enc_mask"],
                                                        dec_data["dec_mask"],
                                                        cov_loss_wt)

            total_loss += batch_loss
            total_log_loss += log_loss
            total_cov_loss += cov_loss
            if (batch+1) % 1 == 0:
                print('Epoch {} Batch {} batch_loss {:.4f} log_loss {:.4f} cov_loss {:.4f}'.format(
                                                params["trained_epoch"] + epoch+1,
                                                batch+1,
                                                batch_loss.numpy(),
                                                log_loss.numpy(),
                                                cov_loss.numpy())
                      )

        # ----------------------------------debug mode-----------------------------
            if params["debug_mode"]:
                if batch >= 10:
                    break
        if params["debug_mode"]:
            break
        # -------------------------------------------------------------------------

        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 1 == 0:
            ckpt_save_path = checkpoint_manager.save()

            try:
                record_file = os.path.join(PGN_CKPT, "record.txt")
                with open(record_file, mode="a", encoding="utf-8") as f:
                    f.write('Epoch {} Loss {:.4f}\n'.format(params["trained_epoch"] + epoch + 1,
                                                total_loss / steps_per_epoch))
            except:
                pass

            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

            # ---------------学习率衰减---------------------------------------------
            lr = params["learning_rate"] * np.power(0.95, epoch+1)
            # 更新优化器的学习率
            optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=lr)
            assert lr == optimizer.get_config()["learning_rate"]
            print("learning_rate=", optimizer.get_config()["learning_rate"])
            # ---------------------------------------------------------------------

            save_train_msg(PGN_CKPT, params["trained_epoch"]+epoch+1)  # 保存已训练的轮数

        # 打印信息
        print('Epoch {} Loss {:.4f} log_loss {:.4f} cov_loss {:.4f}'.format(
                                            params["trained_epoch"] + epoch + 1,
                                            total_loss / steps_per_epoch,
                                        total_log_loss / steps_per_epoch,
                                        total_cov_loss / steps_per_epoch )
                )
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
