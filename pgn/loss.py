# -*- coding:utf-8 -*-
# Created by LuoJie at 12/14/19
import tensorflow as tf

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
def _log_loss(target, pred, dec_mask):
    """
    计算log_loss
    :param target: shape (batch_size, dec_len)
    :param pred:  shape (batch_size, dec_len, vocab_size)
    :param dec_mask: shape (batch_size, dec_len)
    :return: log loss
    """
    loss_ = loss_object(target, pred)
    # 注batcher产生padding_mask时，数据类型需要指定成tf.float32可以少下面这行代码
    # mask = tf.cast(padding_mask, dtype=loss_.dtype)
    loss_ *= dec_mask
    loss_ = tf.reduce_mean(loss_)
    return loss_
    # return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def _coverage_loss(attentions, coverages, dec_mask):
    """
    计算coverage loss
    :param attentions: shape (batch_size, dec_len, enc_len)
    :param coverages: shape (batch_size, dec_len, enc_len)
    :param dec_mask: shape (batch_size, dec_len)
    :return: cov_loss
    """
    # cov_loss (batch_size, dec_len, enc_len)
    cov_loss = tf.minimum(attentions, coverages)
    # mask
    cov_loss = tf.expand_dims(dec_mask, -1) * cov_loss

    # 对enc_len的维度求和
    cov_loss = tf.reduce_sum(cov_loss, axis=2)
    cov_loss = tf.reduce_mean(cov_loss)
    return cov_loss


def calc_loss(target, pred, dec_mask, attentions, coverages, cov_loss_wt=0.5, use_coverage=True):
    if use_coverage:
        log_loss = _log_loss(target, pred, dec_mask)
        cov_loss = _coverage_loss(attentions, coverages, dec_mask)
        return log_loss + cov_loss_wt * cov_loss, log_loss, cov_loss
    else:
        return _log_loss(target, pred, dec_mask), 0, 0


def pgn_log_loss_function(real, final_dists, padding_mask):
    # Calculate the loss per step
    # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
    loss_per_step = []  # will be list length max_dec_steps containing shape (batch_size)
    batch_nums = tf.range(0, limit=real.shape[0])  # shape (batch_size)
    for dec_step, dist in enumerate(final_dists):
        # The indices of the target words. shape (batch_size)
        targets = real[:, dec_step]
        indices = tf.stack((batch_nums, targets), axis=1)  # shape (batch_size, 2)
        gold_probs = tf.gather_nd(dist, indices)  # shape (batch_size). prob of correct words on this step
        losses = -tf.math.log(gold_probs)
        loss_per_step.append(losses)
    # Apply dec_padding_mask and get loss
    _loss = _mask_and_avg(loss_per_step, padding_mask)
    return _loss


def _mask_and_avg(values, padding_mask):
    """Applies mask to values then returns overall average (a scalar)

    Args:
      values: a list length max_dec_steps containing arrays shape (batch_size).
      padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

    Returns:
      a scalar
    """
    padding_mask = tf.cast(padding_mask, dtype=values[0].dtype)
    dec_lens = tf.reduce_sum(padding_mask, axis=1)  # shape batch_size. float32
    values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
    values_per_ex = sum(values_per_step) / dec_lens  # shape (batch_size); normalized value for each batch member
    return tf.reduce_mean(values_per_ex)  # overall average


# 和loss_function一样，但是不如它优美
def loss_function2(real, pred, padding_mask):
    loss = 0
    for t in range(real.shape[1]):
        loss_ = loss_object(real[:, t], pred[:, t, :])
        mask = tf.cast(padding_mask[:, t], dtype=loss_.dtype)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        loss_ = tf.reduce_mean(loss_)  # batch-wise
        loss += loss_
    return loss / real.shape[1]


