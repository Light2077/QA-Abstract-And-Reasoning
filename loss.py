# -*- coding:utf-8 -*-
import tensorflow as tf


def coverage_loss(attn_dists, coverages, padding_mask):
    '''

    :param attn_dists:[max_y_len,batch size ,max_x_len]
    :param coverages: [max_y_len,batch size ,max_x_len,1]
    :param padding_mask:  [batch_size,decoder_max_len ]
    :return:
    '''
    # attn_dists:
    # attn_dists = tf.squeeze(attn_dists, axis=-1)

    # shape (batch_size, max_len_x). Initial coverage is zero.
    coverage = tf.zeros_like(attn_dists[0])
    # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
    cover_losses = []
    for a in attn_dists:
        cover_loss = tf.reduce_sum(tf.minimum(a, coverage), [1])  # calculate the coverage loss for this step
        cover_losses.append(cover_loss)
        # update the coverage vector
        # batch size ,max_x_len
        coverage += a

    # coverage_loss = mask_and_avg(cover_losses, padding_mask)
    loss = tf.reduce_sum(tf.reduce_mean(cover_losses, axis=0))
    # tf.print('coverage loss(batch sum):', loss)
    return loss


def mask_coverage_loss(attn_dists, coverages, padding_mask):
    """
    Calculates the coverage loss from the attention distributions.
      Args:
        attn_dists coverages: [max_len_y, batch_sz, max_len_x, 1]
        padding_mask: shape (batch_size, max_len_y).
      Returns:
        coverage_loss: scalar
    """
    cover_losses = []
    # transfer attn_dists coverages to [max_len_y, batch_sz, max_len_x]
    attn_dists = tf.squeeze(attn_dists, axis=3)
    coverages = tf.squeeze(coverages, axis=3)

    assert attn_dists.shape == coverages.shape

    for t in range(attn_dists.shape[0]):
        cover_loss_ = tf.reduce_sum(tf.minimum(attn_dists[t, :, :], coverages[t, :, :]), axis=-1)  # max_len_x wise
        cover_losses.append(cover_loss_)

    # change from[max_len_y, batch_sz] to [batch_sz, max_len_y]
    cover_losses = tf.stack(cover_losses, 1)
    # cover_loss_ [batch_sz, max_len_y]
    mask = tf.cast(padding_mask, dtype=cover_loss_.dtype)
    cover_losses *= mask
    loss = tf.reduce_sum(tf.reduce_mean(cover_losses, axis=0))  # mean loss of each time step and then sum up
    tf.print('coverage loss(batch sum):', loss)
    return loss
