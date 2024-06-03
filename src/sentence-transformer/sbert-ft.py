from sentence_transformers import SentenceTransformer, losses, InputExample, LoggingHandler, evaluation
from torch.utils.data import DataLoader
import logging

# 设置日志记录
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, handlers=[LoggingHandler()])

# 加载预训练模型
model = SentenceTransformer('path_to_save_roberta_large')

# 准备训练数据
train_examples = [
    InputExample(texts=['Sentence 1', 'Sentence 2'], label=0.8),
    InputExample(texts=['Another sentence', 'Yet another sentence'], label=0.3)
]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# 定义损失函数
train_loss = losses.CosineSimilarityLoss(model)

# 微调模型
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

# 保存微调后的模型
model.save('path_to_save_model')
