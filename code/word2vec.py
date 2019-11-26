from utils.config import *
import pandas as pd
from gensim.models.word2vec import LineSentence
from gensim.models import word2vec
import gensim

if __name__ == "__main__":

    proc_text = pd.read_csv(proc_text_path, header=None)

    # model = word2vec.Word2Vec(LineSentence(proc_text_path), workers=12,min_count=5,size=300)