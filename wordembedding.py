from gensim.models import Word2Vec
import pandas as pd
import os
from os import path


def build(docs):
    # docs should be in DataFrame Format for better and faster preprocessing
    corpus_list = docs.str.split().to_list()
    if not os.path.exists("model"):
        # make new directory
        os.mkdir("model")
        model = Word2Vec(corpus_list, size=100,
                         window=5, min_count=2, workers=4, sg=1)
        model.wv.save('model/corpusvectors.wv')
        model.save('model/wordembed.model')
        print("model saved")
    else:
        print("model exists")


def load():
    model = Word2Vec.load('model/wordembed.model')
    return model
