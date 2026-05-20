# -*-coding:utf-8-*-
import copy
import torch
import torch.nn as nn
import numpy
import math
import torch.nn.functional as F
from transform_input import *
from transform_encoder import *
# todo:1。定义解码器层（由三个子层连接结构组成）
class DecoderLayer(nn.Module):
    def __init__(self,size,self_attention,src_attention,feed_forward,dropout_p):
        super().__init__()
        # size:词嵌入维度
        self.size = size
        # self_attention:多头自注意力机制对象 q=k=v
        self.self_attention = self_attention
        # src_attention:多头注意力机制对象 q!=k=v
        self.src_attention =src_attention
        # feed_forward:前馈全连接层对象
        self.feed_forward = feed_forward
        # clone三个子层连接对象
        self.sublayers = clones(SubLayerConnection(size =self.size,dropout_p =dropout_p),3)
    def forward(self,y,encoder_output,score_mask,target_mask):
        # y:输入序列-->[2,4,512]
        # encoder_output:编码器输出-->[2,4,512]
        # score_mask:作用在第二个子层连接层的多头注意力机制对象上进行padding mask
        # target_mask:作用在第一个子层连接层的多头自注意力机制对象上进行sentence mask（casual mask）
        # 经过第一个子层连接层 得到多头自注意力机制结果+add+norm
        y1 = self.sublayers[0].forward(x=y,sublayer= lambda y:self.self_attention(y,y,y,mask=target_mask))
        # 经过第二个子层连接层 得到多头注意力机制结果+add+norm
        y2 = self.sublayers[1](x=y1,sublayer= lambda y1:self.src_attention(y1,encoder_output,encoder_output,mask=score_mask))
        # 经过第三个子层连接层 得到前馈全连接层结果+add+norm
        y3 = self.sublayers[2](x=y2,sublayer=self.feed_forward)
        return y3

def ceshi_decoder():
    # 定义一个两行三列的输出作为y
    y = torch.tensor([[51,12,85,415,12,54],
                      [45,64,145,23,555,12]])
    # 1.经过embedding层得到词嵌入结果-->[2,6,512]
    embedding = Embeddings(vocab_size=2000,d_model=512)
    embed_y = embedding(y)
    print(f'词嵌入结果为：{embed_y.shape}')
    # 2.经过position encoder层得到位置编码层结果-->[2,6,512]
    my_pe = PostionalEncoding(d_model=512,dropout_p=0.1)
    position_y = my_pe(embed_y)
    print(f'词嵌入结果为：{position_y.shape}')
    # 3.实例化多头注意力机制对象
    attention = MultiHeadAttention(embed_dim=512,head=8,dropout_p=0.1)
    self_atten = copy.deepcopy(attention)
    src_atten = copy.deepcopy(attention)
    # 4.实例化前馈全连接层
    feed_forward = FeedForward(d_model=512,d_ff=1024)
    # 5.实例化解码器层对象
    decoder_layer = DecoderLayer(size=512,self_attention=self_atten,src_attention=src_atten,feed_forward=feed_forward,dropout_p=0.1)
    # 6. 整理解码器的输入 y,encoder_output,score_mask,target_mask
    encoder_output = ceshi_encoder()
    score_mask = torch.zeros(8,6,4)
    target_mask = torch.zeros(8,6,6)
    output = decoder_layer(y=position_y,encoder_output=encoder_output,score_mask=score_mask,target_mask=target_mask)
    print(f'解码器层得到结果：{output.shape}')

# todo:2.定义解码器
class Decoder(nn.Module):
    def __init__(self,layer,N):
        super().__init__()
        self.layers =clones(model=layer,N=6)
        self.norm = layerNorm(feature=layer.size)

    def forward(self,y,encoder_output,score_mask,target_mask):
        for layer in self.layers:
            y = layer(y,encoder_output=encoder_output,score_mask=score_mask,target_mask=target_mask)
        return self.norm(y)
def ceshi_Decoder():
    # 定义一个两行三列的输出作为y
    y = torch.tensor([[51,12,85,415,12,54],
                      [45,64,145,23,555,12]])
    # 1.经过embedding层得到词嵌入结果-->[2,6,512]
    embedding = Embeddings(vocab_size=2000,d_model=512)
    embed_y = embedding(y)
    # print(f'词嵌入结果为：{embed_y.shape}')
    # 2.经过position encoder层得到位置编码层结果-->[2,6,512]
    my_pe = PostionalEncoding(d_model=512,dropout_p=0.1)
    position_y = my_pe(embed_y)
    # print(f'词嵌入结果为：{position_y.shape}')
    # 3.实例化多头注意力机制对象
    attention = MultiHeadAttention(embed_dim=512,head=8,dropout_p=0.1)
    self_atten = copy.deepcopy(attention)
    src_atten = copy.deepcopy(attention)
    # 4.实例化前馈全连接层
    feed_forward = FeedForward(d_model=512,d_ff=1024)
    # 5.实例化解码器层对象
    decoder_layer = DecoderLayer(size=512,self_attention=self_atten,src_attention=src_atten,feed_forward=feed_forward,dropout_p=0.1)
    # 6. 整理解码器的输入 y,encoder_output,score_mask,target_mask
    encoder_output = ceshi_encoder()
    score_mask = torch.zeros(8,6,4)
    target_mask = torch.zeros(8,6,6)
    # 7.实例化解码器对象
    decoder = Decoder(layer =decoder_layer,N=6)
    output = decoder(y=position_y,encoder_output=encoder_output,score_mask=score_mask,target_mask=target_mask)
    # print(f'解码器层得到结果：{output.shape}')
    # print(f'解码器层得到结果：{output}')
    return output

if __name__ == '__main__':
    ceshi_Decoder()