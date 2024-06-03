from sentence_transformers import SentenceTransformer

# 下载并加载 BERT-base 模型
# model_bert_base = SentenceTransformer('bert-base-nli-mean-tokens')
# print("BERT-base model loaded.")

# 下载并加载 BERT-large 模型
# model_bert_large = SentenceTransformer('bert-large-nli-mean-tokens')
# print("BERT-large model loaded.")

# 下载并加载 RoBERTa-base 模型
# model_roberta_base = SentenceTransformer('roberta-base-nli-mean-tokens')
# print("RoBERTa-base model loaded.")

# 下载并加载 RoBERTa-large 模型
model_roberta_large = SentenceTransformer('roberta-large-nli-mean-tokens')
print("RoBERTa-large model loaded.")

# 可选：保存模型到本地
# model_bert_base.save('bert-base-nli-mean-tokens')
# model_bert_large.save('model/s2vec/bert-large')
# model_roberta_base.save('model/s2vec/roberta-base')
model_roberta_large.save('model/s2vec/roberta-large')

# 从本地加载模型
# model_bert_base_local = SentenceTransformer('bert-base-nli-mean-tokens')
# model_bert_large_local = SentenceTransformer('bert-large-nli-mean-tokens')
# model_roberta_base_local = SentenceTransformer('roberta-base-nli-mean-tokens')
# model_roberta_large_local = SentenceTransformer('roberta-large-nli-mean-tokens')

print("Models loaded from local storage.")
