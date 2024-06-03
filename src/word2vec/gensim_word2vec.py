from gensim.models import KeyedVectors

import numpy as np
import gensim
import json
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec

model_path = "model/word2vec/google-300-bin/GoogleNews-vectors-negative300.bin"
model = KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors='ignore')


# 假设你的文本数据在一个列表中
texts = [
            "Component is entity.",
            "Sketch reference is entity."
        ]
with open('data/LoRA/train.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)
    for item in data:
        dialog = item["dialog"]
        texts.append(dialog[2]["content"])

with open('data/LoRA/val.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)
    for item in data:
        dialog = item["dialog"]
        texts.append(dialog[2]["content"])

# 分词和预处理
# with open('data/LoRA/texts.txt', 'w', encoding='utf-8') as f:
#     for text in texts:
#         f.write(" ".join(simple_preprocess(text)) + "\n")

# 分词和预处理
processed_texts = [simple_preprocess(text) for text in texts]
# processed_texts = processed_texts[:100]

# 创建gensim模型，初始化词嵌入层为预训练模型的向量
fine_tune_model = Word2Vec(vector_size=model.vector_size, window=5, min_count=1, sg=1)
# fine_tune_model = Word2Vec(vector_size=300, window=5, min_count=1, sg=0)
fine_tune_model.build_vocab(processed_texts)

# 使用预训练的词向量初始化新模型的权重
# fine_tune_model.build_vocab([list(model.key_to_index.keys())], update=False)
fine_tune_model.wv.vectors = model.vectors
fine_tune_model.wv.key_to_index = model.key_to_index
fine_tune_model.wv.index_to_key = model.index_to_key

# 初始化syn1neg属性
# if hasattr(model.wv, 'syn1neg'):
#     fine_tune_model.syn1neg = model.syn1neg

# 更新词汇表
# fine_tune_model.build_vocab(corpus_file='data/LoRA/texts.txt', progress_per=10000, update=True)
fine_tune_model.build_vocab(processed_texts, update=True)
fine_tune_model.syn1neg = np.zeros((len(fine_tune_model.wv.key_to_index), model.vector_size), dtype=np.float32)


# 准备数据迭代器
# def data_iterator(texts, batch_size):
#     for i in range(0, len(texts), batch_size):
#         yield [text for text in texts[i:i + batch_size]]
grouped_texts = [processed_texts[i:i+100] for i in range(0, len(processed_texts), 100)]


# 训练模型
# fine_tune_model.train(corpus_file='data/LoRA/texts.txt', total_words=fine_tune_model.corpus_total_words, epochs=3)
# fine_tune_model.train(processed_texts, total_examples=len(processed_texts), epochs=3)

# 增量训练模型
for batch in grouped_texts:
    # fine_tune_model.build_vocab(batch, update=True)
    fine_tune_model.train(batch, total_examples=len(batch), epochs=3, queue_factor=4)

# 保存微调后的模型
fine_tune_model.save("model/word2vec/zw-300/zw.model")

# 使用微调后的模型
word = "entity"
similar_words = fine_tune_model.wv.most_similar(word, topn=10)
print(f"Words most similar to '{word}':")
for similar_word, similarity in similar_words:
    print(f"{similar_word}: {similarity}")
