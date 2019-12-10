# -*- coding:utf-8 -*-
# Created by LuoJie at 12/4/19

from matplotlib import font_manager

import matplotlib.ticker as ticker
from utils.config import FONT

# 解决中文乱码
font = font_manager.FontProperties(fname=FONT)
import matplotlib.pyplot as plt


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 12, 'fontproperties': font}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
