# -*- coding:utf-8 -*-
import tensorflow as tf

from seq2seq_tf2.batcher import train_batch_generator, batcher
import time
# from utils.config import WV_MODEL_PAD
# from utils.config_gpu import config_gpu

def train_model(model, vocab, params, checkpoint_manager):
    epochs = params['epochs']
    batch_size = params['batch_size']

    pad_index = vocab.word2id[vocab.PAD_TOKEN]
    start_index = vocab.word2id[vocab.START_DECODING]

    # 计算vocab size
    params['vocab_size'] = vocab.count

    optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=0.01)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    # 定义损失函数
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, pad_index))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    # 训练
    # @tf.function(input_signature=(tf.TensorSpec(shape=[params["batch_size"], params["max_enc_len"]], dtype=tf.int64),
    #                               tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]], dtype=tf.int64)))
    def train_step(enc_inp, dec_target):

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = model.call_encoder(enc_inp)
            # 第一个decoder输入 开始标签
            dec_input = tf.expand_dims([start_index] * batch_size, 1)
            # 第一个隐藏层输入
            dec_hidden = enc_hidden
            # 逐个预测序列
            predictions, _ = model(dec_input, dec_hidden, enc_output, dec_target)
            batch_loss = loss_function(dec_target[:, 1:], predictions)

        variables = model.trainable_variables
        gradients = tape.gradient(batch_loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    # dataset, steps_per_epoch = train_batch_generator(batch_size)
    dataset = batcher(vocab, params)
    # steps_per_epoch =
    for epoch in range(epochs):
        start = time.time()
        total_loss = 0

        for (batch, (inputs, target)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inputs, target)
            total_loss += batch_loss

            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            ckpt_save_path = checkpoint_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
