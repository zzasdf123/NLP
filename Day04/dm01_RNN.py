import torch
import torch.nn as nn
def dm1_rnn_for_base():
    #1.实例化模型
    #RNN参数说明
    #参1：input_size:输入的词嵌入维度
    #参2：hidden_size:RNN单元输出的隐藏层张量的维度
    #参3：num_layers:有几层RNN单元（有几个隐藏层）
    input_size = 5
    hidden_size = 6
    num_layers = 1
    model = nn.RNN(input_size,hidden_size,num_layers)
    #2.获取x0输入
    #x的参数说明
    #参1：sequence_len:每个样本的长度（单词的个数）（因为RNN模型batch_first=False）
    #参2：batch_size:一个批次送入几个样本
    #参3:input_size:输入的词嵌入维度
    sequence_len = 1
    batch_size = 3
    input_size = 5
    x0 = torch.randn(sequence_len,batch_size,input_size)
    #3.获取h0输入
    # h0的参数说明
    # 参1：num_layers:有几层RNN单元（有几个隐藏层）
    # 参2：batch_size:RNN单元输出的隐藏层张量的维度
    # 参3:hidden_size:RNN单元输出的隐藏层张量的维度
    h0 = torch.randn(num_layers,batch_size,hidden_size)
    #4.将输入送给RNN模型得到下一时间步输出结果
    output,h_0 = model(x0,h0)
    print(output)
    print('*'*20)
    print(h_0)
#修改样本长度
def dm2_rnn_for_len():
    #1.实例化模型
    #RNN参数说明
    #参1：input_size:输入的词嵌入维度
    #参2：hidden_size:RNN单元输出的隐藏层张量的维度
    #参3：num_layers:有几层RNN单元（有几个隐藏层）
    input_size = 5
    hidden_size = 6
    num_layers = 1
    model = nn.RNN(input_size,hidden_size,num_layers)
    #2.获取x0输入
    #x的参数说明
    #参1：sequence_len:每个样本的长度（单词的个数）（因为RNN模型batch_first=False）
    #参2：batch_size:一个批次送入几个样本
    #参3:input_size:输入的词嵌入维度
    sequence_len = 4
    batch_size = 3
    x0 = torch.randn(sequence_len,batch_size,input_size)
    print(f'x0.shape{x0.shape}')
    #3.获取h0输入
    # h0的参数说明
    # 参1：num_layers:有几层RNN单元（有几个隐藏层）
    # 参2：batch_size:RNN单元输出的隐藏层张量的维度
    # 参3:hidden_size:RNN单元输出的隐藏层张量的维度
    h0 = torch.randn(num_layers,batch_size,hidden_size)
    print(f'h0.shape{h0.shape}')
    #4.将输入送给RNN模型得到下一时间步输出结果（一次性送入模型）
    output,h_0 = model(x0,h0)
    # print('一次性送入模型')
    print(output)
    print(h_0)
    # print('*' * 20)
    # #5.将一个token一个送入模型
    # #x0.size(0) = 4 代表sequence_len
    # #x0 --》[4,1,5]
    # print(x0)
    # print('一个一个token输入')
    # for idx in range(x0.size(0)):
    #     #print(x0[idx])
    #     temp = x0[idx].unsqueeze(0)
    #     #print(temp)
    #     output,h0 = model(temp,h0)
    #     print('*' * 20)
    #     print(output)
    #     print(h0)
#batch的位置放在第一位
#batch_first默认为False，但是如果设置为True，那么input第一个参数是batch_size
def dm3_rnn_for_batch():
    #1.实例化模型
    #RNN参数说明
    #参1：input_size:输入的词嵌入维度
    #参2：hidden_size:RNN单元输出的隐藏层张量的维度
    #参3：num_layers:有几层RNN单元（有几个隐藏层）
    input_size = 5
    hidden_size = 6
    num_layers = 1
    model = nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
    #2.获取x0输入
    #x的参数说明
    #参1：sequence_len:每个样本的长度（单词的个数）（因为RNN模型batch_first=False）
    #参2：batch_size:一个批次送入几个样本
    #参3:input_size:输入的词嵌入维度
    sequence_len = 4
    batch_size = 3
    x0 = torch.randn(batch_size,sequence_len,input_size)
    print(f'x0.shape{x0.shape}')
    #3.获取h0输入
    # h0的参数说明
    # 参1：num_layers:有几层RNN单元（有几个隐藏层）
    # 参2：batch_size:RNN单元输出的隐藏层张量的维度
    # 参3:hidden_size:RNN单元输出的隐藏层张量的维度
    h0 = torch.randn(num_layers,batch_size,hidden_size)
    print(f'h0.shape{h0.shape}')
    #4.将输入送给RNN模型得到下一时间步输出结果（一次性送入模型）
    output,h_0 = model(x0,h0)
    # print('一次性送入模型')
    print(output)
    print(h_0)
#多层RNN
def dm4_rnn_for_numlayers():
    #1.实例化模型
    #RNN参数说明
    #参1：input_size:输入的词嵌入维度
    #参2：hidden_size:RNN单元输出的隐藏层张量的维度
    #参3：num_layers:有几层RNN单元（有几个隐藏层）
    input_size = 5
    hidden_size = 6
    num_layers = 2
    model = nn.RNN(input_size,hidden_size,num_layers)
    #2.获取x0输入
    #x的参数说明
    #参1：sequence_len:每个样本的长度（单词的个数）（因为RNN模型batch_first=False）
    #参2：batch_size:一个批次送入几个样本
    #参3:input_size:输入的词嵌入维度
    sequence_len = 4
    batch_size = 3
    x0 = torch.randn(sequence_len,batch_size,input_size)
    print(f'x0.shape{x0.shape}')
    #3.获取h0输入
    # h0的参数说明
    # 参1：num_layers:有几层RNN单元（有几个隐藏层）
    # 参2：batch_size:RNN单元输出的隐藏层张量的维度
    # 参3:hidden_size:RNN单元输出的隐藏层张量的维度
    h0 = torch.randn(num_layers,batch_size,hidden_size)
    print(f'h0.shape{h0.shape}')
    #4.将输入送给RNN模型得到下一时间步输出结果（一次性送入模型）
    output,h_0 = model(x0,h0)
    # print('一次性送入模型')
    print(output)
    print(h_0)

if __name__ == '__main__':
    #dm1_rnn_for_base()
    #dm2_rnn_for_len()
    #dm3_rnn_for_batch()
    dm4_rnn_for_numlayers()