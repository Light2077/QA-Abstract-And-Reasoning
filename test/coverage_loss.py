# coverage loss

def _coverage_loss(attn_dists, vocab):
    """
    Calculates the coverage loss from the attention distributions.
    attn_dists shape (decoder_len, batch_size, atten_length)
    padding_mask 掩码操作

    return: coverage_loss shape 
    """


    coverage = tf.zeros_like(attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
    covlosses = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
    for a in attn_dists:
        covloss = tf.reduce_sum(tf.minimum(a, coverage), [1]) # calculate the coverage loss for this step
        covlosses.append(covloss)
        coverage += a # update the coverage vector
    coverage_loss = _mask_and_avg(covlosses, vocab)
    return coverage_loss

def _mask_and_avg(covlosses, vocab):
	pad_index=vocab['<PAD>']
	unk_index=vocab['<UNK>']

	pad_mask = tf.math.equal(real, pad_index)
	unk_mask = tf.math.equal(real, unk_index)

    # <PAD> 和 <UNK> 的损失都不算
    mask = tf.math.logical_not(tf.math.logical_or(pad_mask,unk_mask))

    mask = tf.cast(mask, dtype=loss_.dtype)
    covlosses = covlosses * mask

    return tf.reduce_mean(covlosses)


