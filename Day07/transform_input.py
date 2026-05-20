import torch
import torch.nn as nn
import math
# todo:1.定义词嵌入层
class Embeddings(nn.Module):
    def __init__(self,vocab_size,d_model):
        super().__init__()
        # vocab：词表大小
        self.vocab_size = vocab_size
        # d_model：词向量的维度
        self.d_model = d_model
        # 创建词嵌入层
        self.embed = nn.Embedding(self.vocab_size, self.d_model)
    def forward(self,x):
        # x:[batch_size,seq_len]
        # embed_x:[batch_size,seq_len,d_model]
        embed_x = self.embed(x)
        # 将词向量缩放目的：1.符合标准正太分布 2.增强Embedding的影响
        return embed_x * math.sqrt(self.d_model)
# todo:2.定义位置编码层（位置编码层结果需要与词嵌入层相加）
class PostionalEncoding(nn.Module):
    def __init__(self,d_model,dropout_p,max_len=60):
        super().__init__()
        # d_model:代表词嵌入的维度
        # dropout_p:代表随机失活概率
        # max_len:代表最大句子长度
        # 1.初始化位置编码张量-->[max_len,d_model]-->[60,512]
        pe = torch.zeros(max_len,d_model)
        # 定义随机失活层
        self.dropout = nn.Dropout(dropout_p)
        # 2.定义位置矩阵-->[max_len,1]-->[60,1]
        tem_vec =torch.arange(0,max_len).unsqueeze(1)
        # 3.根据公式定义转换矩阵256个值-->[256]
        div_vec = torch.exp(torch.arange(0,d_model,2)* -math.log(10000.0)/d_model)
        # 4.将每个位置先赋值256个值-->[60,256]
        position =tem_vec * div_vec
        # 5.将pe进行赋值，奇数为用sin，偶数位用cos-e
        pe[:,0::2] = torch.sin(position)
        pe[:,1::2] = torch.cos(position)
        # 6.升维度-->[1,max_len,d_model]————》[1,60,512]
        pe = pe.unsqueeze(0)
        # 7.把pe位置编码矩阵 注册成模型的持久缓冲区buffer中，模型保存加载时，可以跟模型参数一样，一同被加载
        # buffer：对模型效果有帮助，但不是模型结构中的参数与超参数，不参与模型训练
        self.register_buffer('pe',pe)
    def forward(self,x):
        # x代表Embedding层之后的结果————>[batch_size,seq_len,d_model]-->[2,4,512]
        # 因为之前pe是最大句子长度，所以这里取x的句子长度
        # self.pe[:,:x.shape[1]]-->[1,4,512]
        position_x = x + self.pe[:,:x.shape[1]]
        return self.dropout(position_x)

if __name__ == '__main__':
    embed = Embeddings(vocab_size=1000,d_model=512)
    x = torch.tensor([[4,5,47,23],[55,45,654,152]])
    embed_x = embed(x)
    position_embed = PostionalEncoding(d_model=512,dropout_p=0.5)
    result = position_embed(embed_x)
    print(result.shape)





