# -*-coding:utf-8-*-
import copy
import torch
import torch.nn as nn
import numpy
import math
import torch.nn.functional as F
from transform_input import *
# todo:1.生成一个下三角矩阵
def sub_mask(size):
    mask_veb = 1-torch.triu(torch.ones((1,size,size),dtype=torch.long),diagonal=1)
    return mask_veb
# todo:2.实现注意力计算公式的代码
def attention(query,key,value,mask=None,dropout=None):
    # 自注意力机制，query= key= value-->[2,4,512]
    # mask如果用到编码器段就是padding_mask,如果用到解码器段就是sentence_mask
    # dropout随机失活对象
    # d_k：词嵌入的维度 512
    d_k = query.size(-1)
    # 根据注意力计算公式计算注意力分数
    # [2,4,512]*[2,512,4]-->[2,4,4]
    scores = torch.matmul(query,torch.transpose(key,-1,-2)) / math.sqrt(d_k)
    # 如果mask不为空，需要对scores掩码
    if mask is not None:
    # if not mask
        scores = scores.masked_fill(mask==0,-1e9)
    # 将scores进行归一化处理
    atten_weight = F.softmax(scores,dim=-1)
    # print(f'atten_weight:{atten_weight}')
    # 对atten_weight进行随机失活
    if dropout is not None:
        atten_weight = dropout(atten_weight)
    return torch.matmul(atten_weight,value),atten_weight

def ceshi_attention():
    # 假设编码器输入
    x0 = torch.tensor([[12,24,51,55],
                       [55,45,65,441]])
    # 实例化编码器对象
    vocab_size = 1000
    d_model = 512
    my_embed = Embeddings(vocab_size=vocab_size,d_model=d_model)
    # 经过词嵌入层
    embed_x = my_embed(x0)
    # 经过位置编码层（注意：融合了embed层）
    dropout_p = 0.1
    my_pe = PostionalEncoding(d_model=d_model,dropout_p=dropout_p)
    # 经过位置编码层
    pe_x = my_pe(embed_x)
    # 因为是自注意力机制，所以query=key=value
    # query = key = value = pe_x
    # attention_result,atten_weight = attention(query,key,value)
    # print(attention_result.shape)
    # print(atten_weight.shape)
    # 加入掩码
    mask = torch.zeros(2,4,4)
    query = key = value = pe_x
    attention_result,atten_weight = attention(query,key,value,mask= mask)

# todo:3.实现多头注意力的计算
def clones(model,N):
    return nn.ModuleList([copy.deepcopy(model) for _ in range(N)])
class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim,head,dropout_p):
        super().__init__()
        # embedding_dim:代表词嵌入的维度d_model
        # head:代表多头注意力的个数
        # 确保embed_dim能被head整除
        assert embed_dim % head ==0
        # 1.获取每个头的嵌入维度
        self.d_k = embed_dim // head
        # 2.定义head属性
        self.head = head
        # 3.定义四个linear层
        self.linears = clones(nn.Linear(embed_dim,embed_dim),N=4)
        # 4.定义属性atten_weight
        self.atten_weight = None
        # 5.定义随机失活层
        self.dropout = nn.Dropout(p = dropout_p)

    def forward(self,query,key,value,mask=None,dropout=None):
        # query= key= value-->[2,4,512]
        # mask————》[head,seq_len,seq_len]--》[8,4,4]
        if mask is not None:
            mask =mask.unsqueeze(0)
        self.batch_size = query.size(0)
        # 将原始的query,key,value进行linear层，进行线性变换，然后再切分成多个头
        # model(x) = [2,4,512]
        # model(x).view(self.batch_size,-1,self.head,self.d_k)-->[2,4,8,64]
        # transpose(1,2)————》[2,8,4,64]
        query,key,value = [model(x).view(self.batch_size,-1,self.head,self.d_k).transpose(1,2)
            for model,x in zip(self.linears,(query,key,value))]
        # 调用attention的方法实现多头注意力的计算
        # atten_result————>[2,8,4,64]
        # atten_weight————>[2,8,4,64] * [2,8,64,4] = [2,8,4,4]
        # atten_weight[2,8,4,4] * value[2,8,4,64] = atten_result[2,8,4,64]
        atten_result,atten_weight = attention(query,key,value,mask=mask,dropout=self.dropout)
        # 将多头进行合并:result--》[2,4,512]
        result = atten_result.transpose(1,2).contiguous().view(self.batch_size,-1,self.head*self.d_k)
        return self.linears[-1](result)
def ceshi_mutiheaf_attention():
    # 假设编码器输入
    x0 = torch.tensor([[12,24,51,55],
                       [55,45,65,441]])
    # 实例化编码器对象
    vocab_size = 1000
    d_model = 512
    my_embed = Embeddings(vocab_size=vocab_size,d_model=d_model)
    # 经过词嵌入层
    embed_x = my_embed(x0)
    # 经过位置编码层（注意：融合了embed层）
    dropout_p = 0.1
    my_pe = PostionalEncoding(d_model=d_model,dropout_p=dropout_p)
    # 经过位置编码层
    pe_x = my_pe(embed_x)
    # 实例化多头注意力对象
    mha = MultiHeadAttention(embed_dim=512,head=8,dropout_p=0.1)
    # 加入掩码
    mask = torch.zeros(8,4,4)
    query = key = value = pe_x
    attention_result = mha(query,key,value,mask=mask)
    # print(attention_result.shape)
    # print(attention_result)
    return attention_result

# todo:4.实现前馈全连接层
class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout_p=0.1):
        # d_model:词嵌入的维度
        # d_ff:feed_forward的维度
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.linear2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(p = dropout_p)
    def forward(self,x):
        # x为第一个子层输出结果（多头注意力机制计算结果）h
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# todo:5.实现规范化层
class layerNorm(nn.Module):
    def __init__(self,feature,eps=1e-6):
        super().__init__()
        # feature_dim:词嵌入维度
        # eps:eps值 防止分母为零
        self.eps = eps
        # 可学习参数a
        self.a = nn.Parameter(torch.ones(feature))
        # 可学习参数b
        self.b = nn.Parameter(torch.zeros(feature))
    def forward(self,x):
        # x可能来自于多头注意力机制的计算结果，也可能来自于前馈全连接层：x————>[2,4,512]
        # 获取张量x的均值
        x_mean = torch.mean(x,dim=-1,keepdim= True)
        # 获取张量x的方差
        x_std = torch.std(x,dim=-1,keepdim= True)
        return self.a * (x-x_mean)/(x_std+self.eps) + self.b

# todo：6.实现子层连接结构
class SubLayerConnection(nn.Module):
    def __init__(self,size,dropout_p):
        super().__init__()
        # size:词嵌入维度
        self.norm = layerNorm(feature=size)
        # dropout_p:随机失活概率
        self.dropout = nn.Dropout(p = dropout_p)
    def forward(self,x,sublayer):
        # x:如果sublayer是多头注意力机制，x代表原始输入；如果sublayer是前馈全连接层，x代表前第一个子层输出结果
        # x：[2,4,512]
        # sublayer:-->lamda x atten(x,x,x,mask)
        # 第一种方式：post_norm
        result = x + self.dropout(self.norm(sublayer(x)))
        # 第二种方式：pre_norm
        # result = x + self.dropout(sublayer(self.norm(x)))
        return result

def ceshi_sublayer():
    # 假设编码器输入
    x0 = torch.tensor([[12, 24, 51, 55],
                       [55, 45, 65, 441]])
    # 实例化编码器对象
    vocab_size = 1000
    d_model = 512
    my_embed = Embeddings(vocab_size=vocab_size, d_model=d_model)
    # 经过词嵌入层
    embed_x = my_embed(x0)
    # 经过位置编码层（注意：融合了embed层）
    dropout_p = 0.1
    my_pe = PostionalEncoding(d_model=d_model, dropout_p=dropout_p)
    # 经过位置编码层
    pe_x = my_pe(embed_x)
    # 实例化多头注意力对象
    mha = MultiHeadAttention(embed_dim=512, head=8, dropout_p=0.1)
    # 加入掩码
    mask = torch.zeros(8, 4, 4)
    # 定义一个匿名函数
    sub_layer = lambda x: mha(x,x,x,mask= mask)
    # 实例化子层连接对象
    my_sublayer_connect = SubLayerConnection(size=d_model,dropout_p=dropout_p)
    reslut = my_sublayer_connect(x=pe_x,sublayer=sub_layer)
    # print(f'第一个子层包含注意力机制的结果：{reslut.shape}')
    # print(f'第一个子层包含注意力机制的结果：{reslut}')

# todo:7.定义编码器层
class EncoderLayer(nn.Module):
    def __init__(self,size,self_attention,feed_forward,dropout_p):
        super().__init__()
        # size:词嵌入维度
        self.size = size
        # self_attention:多头注意力机制对象
        self.self_attention = self_attention
        # feed_forward:前馈全连接层对象
        self.feed_forward = feed_forward
        # clone两个子层连接对象
        self.sublayer = clones(SubLayerConnection(size =self.size,dropout_p =dropout_p),2)

    def forward(self,x,mask):
        # 经过第一个子层连接层
        x = self.sublayer[0](x=x,sublayer= lambda x: self.self_attention(x,x,x,mask=mask))
        # 经过第二个子层连接层
        x = self.sublayer[1](x=x,sublayer=self.feed_forward)
        return x

# todo:8.实现编码器
class Encoder(nn.Module):
    def __init__(self,layer,N):
        # layer:一个编码器层对象  N：编码器层数量
        super().__init__()
        # 实例化多个编码器层对象
        self.layers = clones(model=layer,N=N)
        # 实例化规范化层
        self.layer_norm = layerNorm(feature=layer.size,eps=1e-6)
    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x=x,mask=mask)
        # 返回规范化后的结果
        return self.layer_norm(x)
def ceshi_encoder():
    # 假设编码器输入
    x0 = torch.tensor([[12, 24, 51, 55],
                       [55, 45, 65, 441]])
    # 实例化编码器对象
    vocab_size = 1000
    d_model = 512
    my_embed = Embeddings(vocab_size=vocab_size, d_model=d_model)
    # 经过词嵌入层
    embed_x = my_embed(x0)
    # 经过位置编码层（注意：融合了embed层）
    dropout_p = 0.1
    my_pe = PostionalEncoding(d_model=d_model, dropout_p=dropout_p)
    # 经过位置编码层
    position_x = my_pe(embed_x)
    # 实例化多头注意力对象
    self_attention = MultiHeadAttention(embed_dim=512, head=8, dropout_p=0.1)
    # 加入掩码
    mask = torch.zeros(8, 4, 4)
    # 实例化前馈全连接层
    ff = FeedForward(d_model=512,d_ff=1024)
    # 实例化编码器层对象
    encoder_layer = EncoderLayer(size=512,self_attention=self_attention,feed_forward=ff,dropout_p=dropout_p)
    # 实例化编码器对象
    encoder = Encoder(layer=encoder_layer,N=6)
    # 将经过位置编码的数据送入编码器
    output = encoder(x=position_x,mask= mask)
    # print(f'编码器得到的结果：{output.shape}')
    # print(f'编码器得到的结果：{output}')
    return output

if __name__ == '__main__':
    ceshi_encoder()