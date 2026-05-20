import torch
import torch.nn as nn
def dm1_lstm():
    #1.实例化模型
    #LSTM参数说明
    #参1：input_size:输入的词嵌入维度
    #参2：hidden_size:LSTM单元输出的隐藏层张量的维度
    #参3：num_layers:有几层LSTM单元（有几个隐藏层）
    input_size = 5
    hidden_size = 6
    num_layers = 2
    model = nn.LSTM(input_size,hidden_size,num_layers)
    #2.获取x0输入
    #x的参数说明
    #参1：sequence_len:每个样本的长度（单词的个数）（因为RNN模型batch_first=False）
    #参2：batch_size:一个批次送入几个样本
    #参3:input_size:输入的词嵌入维度
    sequence_len = 4
    batch_size = 3
    x0 = torch.randn(sequence_len,batch_size,input_size)
    #3.获取h0输入
    # h0的参数说明
    # 参1：num_layers:有几层LSTM单元（有几个隐藏层）
    # 参2：batch_size:LSTM单元输出的隐藏层张量的维度
    # 参3:hidden_size:LSTM单元输出的隐藏层张量的维度
    h0 = torch.randn(num_layers,batch_size,hidden_size)
    c0 = torch.randn(num_layers, batch_size, hidden_size)
    #4.将输入送给LSTM模型得到下一时间步输出结果
    output,(h0,c0) = model(x0,(h0,c0))
    print(f'output:{output}')
    print('*'*20)
    print(f'h0:{h0}')
    print(f'c0:{c0}')
if __name__ == '__main__':
    dm1_lstm()