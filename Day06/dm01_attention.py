# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.python.eager.polymorphic_function.attributes import ORIGINAL_FUNCTION_NAME


#todo:按照第一种规则实现注意力的计算
class MyAttention(nn.Module):
    def __init__(self,query_size,key_size,value_size1,value_size2,output_size):
        super().__init__()
        #query_size:查询张量的最后一个维度
        self.query_size = query_size
        #key_size:K张量的最后一个维度
        self.key_size = key_size
        #value_size1:V张量的中间维度
        self.value_size1 = value_size1
        #value_size2:V张量的最后一个维度
        self.value_size2 = value_size2
        #output_size:注意力指定最后输出维度
        self.output_size = output_size
        #定义第一个全连接层：计算注意力的权重值
        #输入特征，注意：Q与K要进行拼接，然后再输入：Q-->[1,1,32] ,K-->[1,1,32] -->[1,1,64]
        #输出特征，注意：因为Linear之后的结果要和V——>[1,32,64] 做运算，所以输出维度一定是32
        self.attention_weight = nn.Linear(self.query_size + self.key_size,self.value_size1)
        #定义第二个全连接层：计算最终的注意力结果
        # 输入特征，注意：Q与第一步计算的结果要进行拼接，然后再输入：Q-->[1,1,32] ,步骤1结果-->[1,1,64] -->[1,1,96]
        # 输出特征，注意：指定的输出output_size
        self.out =  nn.Linear(self.query_size + self.value_size2,self.output_size)
    def forward(self,Q,K,V):
        #Q:[1,1,32] K:[1,1,32] V:[1,32,64]
        #1.按照注意力计算的步骤，三步走开始实现注意力的运算
        #第一步：按照第一种注意力计算规则来实现Q,K,V的运算
        #1.1将Q,K进行拼接 经过Linear得到权重
        #Q[0]+K[0]=[1,64]
        #[1,64]经过第一个全连接层，输出维度：[1,32]
        atten_weight = F.softmax(self.attention_weight(torch.cat((Q[0],K[0]),dim=-1)), dim=-1)
        #1.2将得到的权重与V进行矩阵相乘
        #atten_weight [1,32] --> [1,1,32]*V[1,32,64] = [1,1,64]
        temp_result = torch.bmm(atten_weight.unsqueeze(0),V)
        #第二步：将Q与第一步结果temp_result进行拼接
        # [1,32] + [1,64] = cat_tensor [1,96]
        cat_tensor = torch.cat((temp_result[0],Q[0]),dim=-1)
        #第三步：将第二步结果按照指定形状输出
        output = self.out(cat_tensor).unsqueeze(0)
        return output,atten_weight

#todo:按照第一种规则实现注意力的计算(不进行升维降维)
class OrignAttention(nn.Module):
    def __init__(self,query_size,key_size,value_size1,value_size2,output_size):
        super().__init__()
        #query_size:查询张量的最后一个维度
        self.query_size = query_size
        #key_size:K张量的最后一个维度
        self.key_size = key_size
        #value_size1:V张量的中间维度
        self.value_size1 = value_size1
        #value_size2:V张量的最后一个维度
        self.value_size2 = value_size2
        #output_size:注意力指定最后输出维度
        self.output_size = output_size
        #定义第一个全连接层：计算注意力的权重值
        #输入特征，注意：Q与K要进行拼接，然后再输入：Q-->[1,1,32] ,K-->[1,1,32] -->[1,1,64]
        #输出特征，注意：因为Linear之后的结果要和V——>[1,32,64] 做运算，所以输出维度一定是32
        self.attention_weight = nn.Linear(self.query_size + self.key_size,self.value_size1)
        #定义第二个全连接层：计算最终的注意力结果
        # 输入特征，注意：Q与第一步计算的结果要进行拼接，然后再输入：Q-->[1,1,32] ,步骤1结果-->[1,1,64] -->[1,1,96]
        # 输出特征，注意：指定的输出output_size
        self.out =  nn.Linear(self.query_size + self.value_size2,self.output_size)
    def forward(self,Q,K,V):
        #Q:[1,1,32] K:[1,1,32] V:[1,32,64]
        #1.按照注意力计算的步骤，三步走开始实现注意力的运算
        #第一步：按照第一种注意力计算规则来实现Q,K,V的运算
        #1.1将Q,K进行拼接 经过Linear得到权重
        #Q+K=[1,1,64]
        #[1,1,64]经过第一个全连接层，输出维度：[1,1,32]
        atten_weight = F.softmax(self.attention_weight(torch.cat((Q,K),dim=-1)), dim=-1)
        #1.2将得到的权重与V进行矩阵相乘
        #atten_weight [1,1,32] --> [1,1,32]*V[1,32,64] = [1,1,64]
        temp_result = torch.bmm(atten_weight,V)
        #第二步：将Q与第一步结果temp_result进行拼接
        # [1,1,32] + [1,1,64] = cat_tensor [1,1,96]
        cat_tensor = torch.cat((temp_result,Q),dim=-1)
        #第三步：将第二步结果按照指定形状输出
        output = self.out(cat_tensor)
        return output,atten_weight

if __name__ == '__main__':
    query_size= 32
    key_size = 32
    value_size1 = 32
    value_size2 = 64
    output_size = 32
    my_atten1 = MyAttention(query_size,key_size,value_size1,value_size2,output_size)
    Q = torch.randn(1,1,32)
    K = torch.randn(1,1,32)
    V = torch.randn(1,32,64)
    output1,atten_weight1 = my_atten1(Q,K,V)
    print(output1.shape)
    print(atten_weight1.shape)
    my_atten2 = OrignAttention(query_size, key_size, value_size1, value_size2, output_size)
    output2, atten_weight2 = my_atten2(Q, K, V)
    print(output2.shape)
    print(atten_weight2.shape)