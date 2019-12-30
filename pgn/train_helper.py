# -*- coding:utf-8 -*-
import tensorflow as tf

from seq2seq_tf2.batcher import train_batch_generator, batcher
import time
import os
import numpy as np
from utils.config import SEQ2SEQ_CKPT

def get_train_msg():
    # 获得已训练的轮次
    path = os.path.join(SEQ2SEQ_CKPT, "trained_epoch.txt")
    with open(path, mode="r", encoding="utf-8") as f:
        trained_epoch = int(f.read())
    return trained_epoch


def save_train_msg(trained_epoch):
    # 保存训练信息（已训练的轮数）
    path = os.path.join(SEQ2SEQ_CKPT, "trained_epoch.txt")
    with open(path, mode="w", encoding="utf-8") as f:
        f.write(str(trained_epoch))



def train_model(model, vocab, params, checkpoint_manager):

    epochs = params['epochs']
    batch_size = params['batch_size']

    pad_index = vocab.word2id[vocab.PAD_TOKEN]
    start_index = vocab.word2id[vocab.START_DECODING]

    # 计算vocab size
    # params['vocab_size'] = vocab.count

    optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=params["learning_rate"])
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    # 定义损失函数
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, pad_index))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)

        # loss_,mask (batch_size, dec_len-1)
        loss_ *= mask
        return tf.reduce_mean(loss_)
        # return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

        # 训练
    # @tf.function(input_signature=(tf.TensorSpec(shape=[params["batch_size"], params["max_enc_len"]], dtype=tf.int64),
    #                               tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]], dtype=tf.int64)))
    def train_step(enc_input, dec_target):
        # dec_target [4980, 939, 41, 27, 4013, 815, 14702]

        with tf.GradientTape() as tape:

            # enc_output (batch_size, enc_len, enc_unit)
            # enc_hidden (batch_size, enc_unit)
            enc_output, enc_hidden = model.encoder(enc_input)

            # 第一个decoder输入 开始标签
            # dec_input (batch_size, 1)
            dec_input = tf.expand_dims([start_index] * batch_size, 1)

            # 第一个隐藏层输入
            # dec_hidden (batch_size, enc_unit)
            dec_hidden = enc_hidden
            # 逐个预测序列
            # predictions (batch_size, dec_len-1, vocab_size)
            predictions, _ = model(dec_input, dec_hidden, enc_output, dec_target)

            _batch_loss = loss_function(dec_target[:, 1:], predictions)

        variables = model.trainable_variables
        gradients = tape.gradient(_batch_loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return _batch_loss

    # dataset, steps_per_epoch = train_batch_generator(batch_size)

    dataset = batcher(vocab, params)
    steps_per_epoch =params["steps_per_epoch"]

    for epoch in range(epochs):
        start = time.time()
        total_loss = 0

       # for (batch, (inputs, target)) in enumerate(dataset.take(steps_per_epoch)):
        for (batch, enc_dec_inputs) in enumerate(dataset.take(steps_per_epoch)):
            inputs = enc_dec_inputs["enc_input"]
            target = enc_dec_inputs["target"]

            batch_loss = train_step(inputs, target)

            total_loss += batch_loss
            if (batch+1) % 1 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(params["trained_epoch"] + epoch+1,
                                                             batch+1,
                                                             batch_loss.numpy())
                      )

            if params["debug_mode"]:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch+1,
                                                             batch,
                                                             batch_loss.numpy()))
                if batch >= 10:
                    break

        if params["debug_mode"]:
            break

        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 1 == 0:
            ckpt_save_path = checkpoint_manager.save()

            try:
                record_file = os.path.join(SEQ2SEQ_CKPT, "record.txt")
                with open(record_file, mode="a", encoding="utf-8") as f:
                    f.write('Epoch {} Loss {:.4f}\n'.format(params["trained_epoch"] + epoch + 1,
                                                total_loss / steps_per_epoch))
            except:
                pass

            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

            # ---学习率衰减---
            lr = params["learning_rate"] * np.power(0.9, epoch+1)

            # 更新优化器的学习率
            optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=lr)

            assert lr == optimizer.get_config()["learning_rate"]

            print("learning_rate=", optimizer.get_config()["learning_rate"])
            save_train_msg(params["trained_epoch"]+epoch+1)  # 保存已训练的轮数

        print('Epoch {} Loss {:.4f}'.format(params["trained_epoch"] + epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
