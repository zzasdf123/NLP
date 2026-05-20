# -*-coding:utf-8-*-
import copy
import torch
import torch.nn as nn
import numpy
import math
import torch.nn.functional as F
from transform_input import *
from transform_encoder import *
from transform_decoder import *

class Generator(nn.Module):
    def __init__(self,d_model,vocab_size):
        # d_model:词嵌入维度
        # vocab_size:解码器词表大小
        super().__init__()
        self.linear =nn.Linear(d_model,vocab_size)
    def forward(self,x):
        output = F.log_softmax(self.linear(x),dim=-1)
        return output

def ceshi_generator():
    decoder_output = ceshi_Decoder()
    # 实例化generator对象
    my_generator = Generator(d_model=512,vocab_size=2000)
    result = my_generator(decoder_output)
    print(f'generator结果为：{result.shape}')
    print(f'generator结果为：{result}')

if __name__ == '__main__':
    ceshi_generator()