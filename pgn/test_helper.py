# -*- coding:utf-8 -*-
# Created by LuoJie at 12/8/19
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from pgn.model import _calc_final_dist

def greedy_decode(model, dataset, vocab, params):
    # 存储结果
    batch_size = params["batch_size"]
    results = []

    sample_size = 20000
    # batch 操作轮数 math.ceil向上取整 小数 +1
    # 因为最后一个batch可能不足一个batch size 大小 ,但是依然需要计算
    steps_epoch = sample_size // batch_size + 1
    # [0,steps_epoch)
    ds = iter(dataset)
    for i in tqdm(range(steps_epoch)):
        enc_data, dec_data = next(ds)

        # batch_data = enc_data["enc_input"]

        results += batch_greedy_decode(model, enc_data, vocab, params)
    return results


def decode_one_step(params, model, enc_extended_inp, batch_oov_len, dec_input, dec_hidden, enc_output,
                    enc_pad_mask, prev_coverage, batch_size, use_coverage=True):
    # 开始decoder
    context_vector, dec_hidden, \
    dec_x, pred, attn, coverage = model.decoder(dec_input, dec_hidden, enc_output,
                                                enc_pad_mask, prev_coverage, use_coverage)

    # 计算p_gen
    p_gen = model.pointer(context_vector, dec_hidden, dec_x)

    # 保证pred attn p_gen的参数为3D的
    final_dist = _calc_final_dist(enc_extended_inp,
                                  tf.expand_dims(pred, 1),
                                  tf.expand_dims(attn, 1),
                                  tf.expand_dims(p_gen, 1),
                                  batch_oov_len,
                                  params["vocab_size"],
                                  batch_size)

    return final_dist, dec_hidden, coverage


def batch_greedy_decode(model, enc_data, vocab, params):
    # 判断输入长度
    batch_data = enc_data["enc_input"]
    batch_size = enc_data["enc_input"].shape[0]
    # 开辟结果存储list
    predicts = [''] * batch_size
    inputs = batch_data
    # print(batch_size, batch_data.shape)
    enc_output, enc_hidden = model.encoder(inputs)
    dec_hidden = enc_hidden
    # dec_input = tf.expand_dims([vocab.word_to_id(vocab.START_DECODING)] * batch_size, 1)
    dec_input = tf.constant([vocab.word2id[vocab.START_DECODING]] * batch_size)

    # Teacher forcing - feeding the target as the next input

    try:
        batch_oov_len = tf.shape(enc_data["article_oovs"])[1]
    except:
        batch_oov_len = tf.constant(0)

    coverage = tf.zeros((enc_output.shape[0],enc_output.shape[1],1))
    for t in range(params['max_dec_len']):
        # 单步预测
        # final_dist (batch_size, 1, vocab_size+batch_oov_len)
        final_dist, dec_hidden, coverage = decode_one_step(params, model,
                                                           enc_data["extended_enc_input"],
                                                           batch_oov_len,
                                                           dec_input,
                                                           dec_hidden,
                                                           enc_output,
                                                           enc_data["enc_mask"],
                                                           coverage,
                                                           batch_size,
                                                           use_coverage=True)

        # id转换
        final_dist = tf.squeeze(final_dist, axis=1)
        predicted_ids = tf.argmax(final_dist, axis=1)

        for index, predicted_id in enumerate(predicted_ids.numpy()):
            predicts[index] += vocab.id_to_word(predicted_id) + ' '

        # using teacher forcing
        dec_input = predicted_ids


    results = []
    for predict in predicts:
        # 去掉句子前后空格
        predict = predict.strip()
        # 句子小于max len就结束了 截断
        if vocab.STOP_DECODING in predict:
            # 截断stop
            predict = predict[:predict.index(vocab.STOP_DECODING)]
        # 保存结果
        results.append(predict)
    return results


class Hypothesis:
    """ Class designed to hold hypothesises throughout the beamSearch decoding """

    def __init__(self, tokens, log_probs, hidden, attn_dists):
        self.tokens = tokens  # list of all the tokens from time 0 to the current time step t
        self.log_probs = log_probs  # list of the log probabilities of the tokens of the tokens
        self.hidden = hidden  # decoder hidden state after the last token decoding
        self.attn_dists = attn_dists  # attention dists of all the tokens
        self.abstract = ""

    def extend(self, token, log_prob, hidden, attn_dist):
        """Method to extend the current hypothesis by adding the next decoded token and all the informations associated with it"""
        return Hypothesis(tokens=self.tokens + [token],  # we add the decoded token
                          log_probs=self.log_probs + [log_prob],  # we add the log prob of the decoded token
                          hidden=hidden,  # we update the state
                          attn_dists=self.attn_dists + [attn_dist])

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def tot_log_prob(self):
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        return self.tot_log_prob / len(self.tokens)


def print_top_k(hyp, k, vocab, batch):
    text = " ".join([vocab.id_to_word(int(index)) for index in batch[0]])
    print('\nhyp.text :{}'.format(text))
    for i in range(min(k,len(hyp))):
        k_hyp = hyp[i]
        k_hyp.abstract = " ".join([vocab.id_to_word(index) for index in k_hyp.tokens])
        print('top {} best_hyp.abstract :{}\n'.format(i, k_hyp.abstract))


def beam_decode(model, batch, vocab, params):
    # 初始化mask
    start_index = vocab.word_to_id(vocab.START_DECODING)
    stop_index = vocab.word_to_id(vocab.STOP_DECODING)
    unk_index = vocab.word_to_id(vocab.UNKNOWN_TOKEN)
    batch_size = params['batch_size']

    # 单步decoder
    def decoder_one_step(enc_output, dec_input, dec_hidden):
        # 单个时间步 运行

        final_pred, dec_hidden, context_vector, attention_weights = model.call_decoder_onestep(dec_input,
                                                                                               dec_hidden,
                                                                                               enc_output)

        # 拿到top k个index 和 概率
        top_k_probs, top_k_ids = tf.nn.top_k(tf.squeeze(final_pred), k=params["beam_size"] * 2)
        # 计算log概率
        top_k_log_probs = tf.math.log(top_k_probs)

        results = {
            "dec_hidden": dec_hidden,
            "attention_weights": attention_weights,
            "top_k_ids": top_k_ids,
            "top_k_log_probs": top_k_log_probs}

        # 返回需要保存的中间结果和概率
        return results

    # 测试数据的输入
    enc_input = batch

    # 计算第encoder的输出
    enc_output, enc_hidden = model.encoder(enc_input)

    # 初始化batch size个 假设对象
    hyps = [Hypothesis(tokens=[start_index],
                       log_probs=[0.0],
                       hidden=enc_hidden[0],
                       attn_dists=[]) for _ in range(batch_size)]
    # 初始化结果集
    results = []  # list to hold the top beam_size hypothesises
    # 遍历步数
    steps = 0  # initial step

    # 长度还不够 并且 结果还不够 继续搜索
    while steps < params['max_dec_len'] and len(results) < params['beam_size']:
        # 获取最新待使用的token
        latest_tokens = [h.latest_token for h in hyps]
        # 替换掉 oov token unknown token
        latest_tokens = [t if t in vocab.id2word else unk_index for t in latest_tokens]

        # 获取所以隐藏层状态
        hiddens = [h.hidden for h in hyps]

        dec_input = tf.expand_dims(latest_tokens, axis=1)
        dec_hidden = tf.stack(hiddens, axis=0)

        # 单步运行decoder 计算需要的值
        decoder_results = decoder_one_step(enc_output,
                                           dec_input,
                                           dec_hidden)

        dec_hidden = decoder_results['dec_hidden']
        attention_weights = decoder_results['attention_weights']
        top_k_log_probs = decoder_results['top_k_log_probs']
        top_k_ids = decoder_results['top_k_ids']

        # 现阶段全部可能情况
        all_hyps = []
        # 原有的可能情况数量 TODO
        num_orig_hyps = 1 if steps == 0 else len(hyps)

        # 遍历添加所有可能结果
        for i in range(num_orig_hyps):
            h, new_hidden, attn_dist = hyps[i], dec_hidden[i], attention_weights[i]
            # 分裂 添加 beam size 种可能性
            for j in range(params['beam_size'] * 2):
                # 构造可能的情况
                new_hyp = h.extend(token=top_k_ids[i, j].numpy(),
                                   log_prob=top_k_log_probs[i, j],
                                   hidden=new_hidden,
                                   attn_dist=attn_dist)
                # 添加可能情况
                all_hyps.append(new_hyp)

        # 重置
        hyps = []
        # 按照概率来排序
        sorted_hyps = sorted(all_hyps, key=lambda h: h.avg_log_prob, reverse=True)

        # 筛选top前beam_size句话
        for h in sorted_hyps:
            if h.latest_token == stop_index:
                # 长度符合预期,遇到句尾,添加到结果集
                if steps >= params['min_dec_steps']:
                    results.append(h)
            else:
                # 未到结束 ,添加到假设集
                hyps.append(h)

            # 如果假设句子正好等于beam_size 或者结果集正好等于beam_size 就不在添加
            if len(hyps) == params['beam_size'] or len(results) == params['beam_size']:
                break

        steps += 1

    if len(results) == 0:
        results = hyps

    hyps_sorted = sorted(results, key=lambda h: h.avg_log_prob, reverse=True)
    print_top_k(hyps_sorted, 3, vocab, batch)

    best_hyp = hyps_sorted[0]
    best_hyp.abstract = " ".join([vocab.id_to_word(index) for index in best_hyp.tokens])
    # best_hyp.text = batch[0]["article"].numpy()[0].decode()
    return best_hyp
