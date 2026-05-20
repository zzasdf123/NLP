import torch.nn as nn
import torch
embed = nn.Embedding(100,3,padding_idx=5)
#padding_idx = 5 表示索引为5的词向量全为0
tensor = torch.tensor([[1,4,5,80,5],[11,24,54,99,5]])
print(embed(tensor))