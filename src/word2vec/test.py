from gensim.models import KeyedVectors

import numpy as np
import gensim
import json
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec

# model_path = "model/word2vec/google-300/word2vec-google-news-300.model"
# model = KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors='ignore')

model_path = "model/word2vec/zw-300/zw.model"
model = Word2Vec.load(model_path)


# 假设你的文本数据在一个列表中
texts = [
            "Component is entity.",
            "Sketch reference is entity."
        ]

word = "entity"
similar_words = model.wv.most_similar(word, topn=10)
print(f"Words most similar to '{word}':")
for similar_word, similarity in similar_words:
    print(f"{similar_word}: {similarity}")