#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from networkx.generators.harary_graph import hnm_harary_graph
from tensorflow.python.ops.distributions.kullback_leibler import cross_entropy
from torch.utils.data import DataLoader,Dataset        #导入torch 的数据源 数据迭代器工具包
import string                                          #用于获取常见的字母及字符规范化
import time                                            #导入时间工具包
import matplotlib.pyplot as plt                        #引入制图工具
import pandas as pd
from tqdm import tqdm                                   #引入进度条
import json
#todo:1.获取常用的字符数量，也就是one-hot编码的去重后的词汇的总量n
all_letters = string.ascii_letters + " ,;.'"
#print(f'all_letters:{all_letters}')
n_letters = len(all_letters)
print(f'当前字符的总量：{n_letters}')

#todo：2.获取国家名种类数
# df = pd.read_csv(filepath_or_buffer ='./name_classfication.txt',sep = '\t')
# categorys =set(df.iloc[:,1])
# categorys = list(categorys)
categorys = ['Korean', 'Arabic', 'Vietnamese', 'Chinese', 'Scottish', 'Polish', 'English', 'Italian', 'Spanish', 'Czech', 'Portuguese', 'French', 'Dutch', 'Greek', 'German', 'Russian', 'Japanese', 'Irish']
print(f'国家列表：{categorys}')
categorys_num = len(categorys)
print(f'当前国家名种类数：{categorys_num}')

#todo：3.读取数据到内存
def read_data(file_path):
    #1.定义两个空列表存储数据
    my_list_x,my_list_y = [],[]
    with open(file_path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            if len(line) <=5:
                continue
            x,y = line.rstrip().split('\t')   #strip 去除字符串首尾的换行符  rstrip删除字符串右边的换行符
            my_list_x.append(x)
            my_list_y.append(y)
    return my_list_x,my_list_y

#todo:4.构建Dataset类
class NameDataset(Dataset):
    def __init__(self,my_list_x,my_list_y):
        super().__init__()
        #样本x
        self.my_list_x = my_list_x
        #样本y
        self.my_list_y = my_list_y
        #样本数量
        self.sample_len = len(my_list_y)
    #获取样本数量
    def __len__(self):
        return self.sample_len
    #根据索引取出元素item，代表索引
    def __getitem__(self,item):
        #1.对异常索引进行修正
        item = min(max(item,0),self.sample_len - 1)
        #2.根据索引取出样本
        x = self.my_list_x[item]
        y = self.my_list_y[item]
        #3.将人名转换为张量形式
        #3.1初始化全0张量
        tensor_x = torch.zeros(size=(len(x),n_letters))
        #print(tensor_x)
        #3.2遍历人名的每一个字母进行one——hot编码
        for idx,letter in enumerate(x):
            #print(idx)
            #print(letter)
            tensor_x[idx][all_letters.find(letter)] = 1
            #print(tensor_x)
        #print(tensor_x)
        tensor_y = torch.tensor(categorys.index(y),dtype=torch.long)
        return tensor_x,tensor_y

#todo：5.实例化Dataloader对象
def get_dataloader():
    #1.获取文档数据
    my_list_x,my_list_y = read_data(file_path = './name_classfication.txt')
    #2.获取Dataset对象
    name_dataset = NameDataset(my_list_x,my_list_y)
    #3.封装Dataloader对象(会对数据进行增维)
    train_Dataloader = DataLoader(dataset=name_dataset,batch_size=1,shuffle=True)
    return train_Dataloader

#todo:6.定义RNN层
class NameRNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers=1):
        super().__init__()
        #input_size:输入数据的词嵌入维度
        self.input_size = input_size
        #hidden_size:代表RNN模型输出的维度（隐藏层输出维度）
        self.hidden_size = hidden_size
        #output_size:代表输出层类别的总个数（18个国家）
        self.output_size = output_size
        #num_layers:代表RNN模型有多少层
        self.num_layers = num_layers
        #定义rnn层：（默认情况下：batch_first = False）
        self.rnn = nn.RNN(self.input_size,self.hidden_size,num_layers)
        #定义输出层
        self.out = nn.Linear(self.hidden_size,self.output_size)
        #3.定义softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self,x,h0):
        #x:代表输入的原始数据的维度[seq_len,input_size]
        #h0:代表初始化的值，[num_layers,batch_size,hidden_size] ——》[1,1,128]
        #x需要先升维：[seq_len,input_size]——>[seq_len,batch_size,input_size]
        #x1 = x.unsqueeze(1)
        # print(f'h0:{h0.shape}')
        # print(f'x:{x.shape}')
        x1 = torch.unsqueeze(x,1)
        # print(f'x1:{x1.shape}')
        #将x1与h0送入RNN模型
        output,hn = self.rnn(x1,h0)
        # print(f'output.shape{output.shape}')
        # print(f'hn.shape{hn.shape}')
        #获取最后一个单词的隐藏层张量来代表整个句子（人名）的语义
        #这里可以直接用hn代替
        temp = output[-1] #[1,128]
        #将temp送入输出层：result————》[1,18]
        result = self.out(temp)
        return self.softmax(result),hn

    def init_hidden(self):
        return torch.zeros(self.num_layers,1,self.hidden_size)

#todo:7.定义LSTM层
class NameLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        # input_size:输入数据的词嵌入维度
        self.input_size = input_size
        # hidden_size:代表lstm模型输出的维度（隐藏层输出维度）
        self.hidden_size = hidden_size
        # output_size:代表输出层类别的总个数（18个国家）
        self.output_size = output_size
        # num_layers:代表lstm模型有多少层
        self.num_layers = num_layers
        # 定义lstm层：（默认情况下：batch_first = False）
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers)
        # 定义输出层
        self.out = nn.Linear(self.hidden_size, self.output_size)
        # 3.定义softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self,x0,h0,c0):
        # x0:代表输入的原始数据的维度[seq_len,input_size]
        # h0,c0:代表初始化的值，[num_layers,batch_size,hidden_size] ——》[1,1,128]
        # x0需要先升维：[seq_len,input_size]——>[seq_len,batch_size,input_size]
        # x1 = x.unsqueeze(1)
        x0= torch.unsqueeze(x0, 1)
        # 将x0,h0,c0送入lstm模型
        output,(hn,cn) = self.lstm(x0,(h0,c0))
        # 获取最后一个单词的隐藏层张量来代表整个句子（人名）的语义
        # 这里可以直接用hn代替
        temp = output[-1]  # [1,128]
        # 将temp送入输出层：result————》[1,18]
        result = self.out(temp)
        return self.softmax(result),hn,cn

    def init_hidden(self):
        h0 = torch.zeros(self.num_layers,1,self.hidden_size)
        c0 = torch.zeros(self.num_layers,1,self.hidden_size)
        return h0,c0

#todo:8.定义GRU层
class NameGRU(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers=1):
        super().__init__()
        #input_size:输入数据的词嵌入维度
        self.input_size = input_size
        #hidden_size:代表GRU模型输出的维度（隐藏层输出维度）
        self.hidden_size = hidden_size
        #output_size:代表输出层类别的总个数（18个国家）
        self.output_size = output_size
        #num_layers:代表GRU模型有多少层
        self.num_layers = num_layers
        #定义GRU层：（默认情况下：batch_first = False）
        self.gru = nn.GRU(self.input_size,self.hidden_size,num_layers)
        #定义输出层
        self.out = nn.Linear(self.hidden_size,self.output_size)
        #3.定义softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self,x,h0):
        #x:代表输入的原始数据的维度[seq_len,input_size]
        #h0:代表初始化的值，[num_layers,batch_size,hidden_size] ——》[1,1,128]
        #x需要先升维：[seq_len,input_size]——>[seq_len,batch_size,input_size]
        x1 = torch.unsqueeze(x,1)
        output,hn = self.gru(x1,h0)
        #获取最后一个单词的隐藏层张量来代表整个句子（人名）的语义
        #这里可以直接用hn代替
        temp = output[-1] #[1,128]
        #将temp送入输出层：result————》[1,18]
        result = self.out(temp)
        return self.softmax(result),hn

    def init_hidden(self):
        return torch.zeros(self.num_layers,1,self.hidden_size)

#todo:9.定义RNN训练函数
def train_rnn():
    #正常步骤:1.读取文件数据 2.构建dataset对象 以便于构建Dataloader对象 3.构建模型对象 4.构建损失函数对象 5.构建优化器对象 6.训练模型
    my_lr = 1e-3
    epochs = 1
    input_size = 57
    hidden_size = 128
    output_size = 18
    num_layers = 1
    #创建模型对象
    my_model = NameRNN(input_size, hidden_size, output_size)
    #创建损失函数对象
    criterion = nn.NLLLoss()
    #创建优化器对象
    optimizer = optim.Adam(my_model.parameters(),lr=my_lr)
    #定义模型参数
    start_time = time.time() #开始的时间
    total_iter_num = 0 #已经训练的样本总个数
    total_loss = 0.0   #已经训练的样本的损失之和
    total_loss_list = []  #每隔100个样本计算一下平均损失，画图
    total_num_acc = 0  #已经训练的样本中预测正确的个数
    total_acc_list = [] #每隔100个样本计算一下平均准确率，画图
    #开始训练
    for epoch in range(epochs):
        #1.实例化Dataloader对象
        train_dataloader = get_dataloader()
        #2.开始内部数据迭代
        for idx,(x,y) in enumerate(tqdm(train_dataloader)):
            #3.将数据送入模型
            h0 = my_model.init_hidden()
            output,hn = my_model(x[0],h0)
            #4.计算损失  output.shape--》[1,18] y.shape-->[1]
            loss = criterion(output,y)
            #5.梯度清零+反向传播+梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #6.计算总损失
            total_iter_num += 1
            total_loss += loss.item()
            #7.计算总准确率
            total_num_acc += 1 if torch.argmax(output).item() == y.item() else 0
            total_acc = total_num_acc/total_iter_num
            #8.每隔100个样本计算一下平均损失，平均准确率
            if total_iter_num % 100 == 0:
                total_loss_list.append(total_loss /total_iter_num)
                total_acc_list.append(total_num_acc/total_iter_num)
            #9.每隔2000步，打印日志
            if total_iter_num % 2000 == 0:
                print(f'轮数：{epoch+1},迭代次数：{total_iter_num},平均损失：{total_loss/total_iter_num:.6f},总准确率：{total_acc:.3f},时间：{time.time()-start_time:.2f}')
    torch.save(my_model.state_dict(),'C:/Users/Administrator/PycharmProjects/PythonProject/NLP/Day05/name_class_rnn.pth')
    total_time = time.time() - start_time
    print(f'训练完成，总耗时：{total_time:.2f}秒')
    #10.将训练的结果进行保存
    dict1 = {'total_loss_list':total_loss_list,
            'total_acc_list':total_acc_list}
    with open('rnn_result.json','w') as fw:
        fw.write(json.dumps(dict1))
    #return total_loss_list,total_acc_list,total_time

#todo:10.定义LSTM训练函数
def train_lstm():
    my_lr = 1e-3
    epochs = 1
    input_size = 57
    hidden_size = 128
    output_size = 18
    num_layers = 1
    #1.读取数据
    my_list_x,my_list_y = read_data(file_path = './name_classfication.txt')
    #2.构建dataset对象
    my_dataset = NameDataset(my_list_x,my_list_y)
    #3.构建dataloader对象
    train_dataloader = DataLoader(my_dataset,batch_size=1,shuffle=True)
    #4.构建模型对象
    my_model = NameLSTM(input_size, hidden_size, output_size,num_layers)
    #5.构建损失函数对象
    criterion = nn.NLLLoss()
    #6.构建优化器对象
    optimizer = torch.optim.Adam(my_model.parameters(),lr=my_lr)
    #7.定义模型参数
    start_time = time.time()
    total_iter_num = 0
    total_loss = 0.0
    total_loss_list = []
    total_num_acc = 0
    total_acc_list = []
    #8.具体每轮训练
    for epoch in range(epochs):
        for x,y in tqdm(train_dataloader):
            #初始化隐藏层和细胞状态
            h0,c0 = my_model.init_hidden()
            # 前向传播
            output,hn,cn = my_model(x[0],h0,c0)
            # 计算损失
            loss = criterion(output,y)
            #梯度清零 + 反向传播 + 梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 计算总损失
            total_loss += loss.item()
            total_iter_num += 1
            # 计算总准确率
            total_num_acc += 1 if torch.argmax(output).item() == y.item() else 0
            total_acc = total_num_acc/total_iter_num
            # 每隔100个样本计算一下平均损失，平均准确率
            if total_iter_num % 100 == 0:
                total_loss_list.append(total_loss /total_iter_num)
                total_acc_list.append(total_num_acc/total_iter_num)
            if total_iter_num % 2000 == 0:
                print(f'轮数：{epoch+1},迭代次数：{total_iter_num},平均损失：{total_loss/total_iter_num:.6f},总准确率：{total_acc:.3f},时间：{time.time()-start_time:.2f}')
    torch.save(my_model.state_dict(),'C:/Users/Administrator/PycharmProjects/PythonProject/NLP/Day05/name_class_lstm.pth')
    total_time = time.time() - start_time
    print(f'训练完成，总耗时：{total_time:.2f}秒')
    dict1 = {'total_loss_list':total_loss_list,
            'total_acc_list':total_acc_list,
            'total_time':total_time}
    with open('lstm_result.json','w') as fw:
        fw.write(json.dumps(dict1))
        #return total_loss_list,total_acc_list,total_time

#todo:11.定义GRU训练函数
def train_gru():
    my_lr = 1e-3
    epochs = 1
    input_size = 57
    hidden_size = 128
    output_size = 18
    num_layers = 1
    #1.获取Dataloader对象
    train_dataloader = get_dataloader()
    #2.构建模型对象
    my_model = NameGRU(input_size, hidden_size, output_size,num_layers)
    #3.构建损失函数对象
    criterion = nn.NLLLoss()
    #4.构建优化器对象
    optimizer = torch.optim.Adam(my_model.parameters(),lr=my_lr)
    #5.定义模型参数
    start_time = time.time()
    total_iter_num = 0
    total_loss = 0.0
    total_loss_list = []
    total_num_acc = 0
    total_acc_list = []
    #6.开始每轮训练
    for epoch in range(epochs):
        for x,y in tqdm(train_dataloader):
            h0 = my_model.init_hidden()
            # 前向传播
            output,hn = my_model(x[0],h0)
            # 获取损失
            loss = criterion(output,y)
            #梯度清零 + 反向传播 + 梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 获取总损失
            total_loss += loss.item()
            total_iter_num += 1
            # 获取总准确率
            total_num_acc += 1 if torch.argmax(output).item() == y.item() else 0
            total_acc = total_num_acc/total_iter_num
            # 每隔100个样本计算一下平均损失，平均准确率
            if total_iter_num % 100 == 0:
                total_loss_list.append(total_loss/total_iter_num)
                total_acc_list.append(total_num_acc/total_iter_num)
            #每隔2000步，打印日志
            if total_iter_num % 2000 == 0:
                print(f'轮数：{epoch+1},迭代次数：{total_iter_num},平均损失：{total_loss/total_iter_num:.6f},总准确率：{total_acc:.3f},时间：{time.time()-start_time:.2f}')
    #7.保存模型参数
    torch.save(my_model.state_dict(),'C:/Users/Administrator/PycharmProjects/PythonProject/NLP/Day05/name_class_gru.pth')
    total_time = time.time() - start_time
    print(f'训练完成，总耗时：{total_time:.2f}秒')
    dict1 = {'total_loss_list':total_loss_list,
            'total_acc_list':total_acc_list}
    with open('gru.json','w') as fw:
        fw.write(json.dumps(dict1))
        #return total_loss_list,total_acc_list,total_time

#todo:12.对比可视化
def compare_rnns():
    with open('rnn_result.json','r') as fr:
        rnn_dict = json.loads(fr.read())
        # rnn_total_loss_list = rnn_dict['total_loss_list']
        # rnn_total_acc_list = rnn_dict['total_acc_list']
    with open('lstm_result.json','r') as fr:
        lstm_dict = json.loads(fr.read())
        # lstm_total_loss_list = lstm_dict['total_loss_list']
        # lstm_total_acc_list = lstm_dict['total_acc_list']
    with open('gru.json','r') as fr:
        gru_dict = json.loads(fr.read())
        # gru_total_loss_list = gru_dict['total_loss_list']
        # gru_total_acc_list = gru_dict['total_acc_list']
    # 绘制图像
    #1.绘制损失对比曲线图
    plt.figure(0)
    plt.plot(rnn_dict['total_loss_list'],label='RNN')
    plt.plot(lstm_dict['total_loss_list'],label='LSTM')
    plt.plot(gru_dict['total_loss_list'],label='GRU')
    plt.legend(loc='upper right')
    plt.savefig("C:/Users/Administrator/PycharmProjects/PythonProject/NLP/Day05/损失对比图")
    plt.show()
    #2.绘制准确率对比曲线图
    plt.figure(1)
    plt.plot(rnn_dict['total_acc_list'],label='RNN')
    plt.plot(lstm_dict['total_acc_list'],label='LSTM')
    plt.plot(gru_dict['total_acc_list'],label='GRU')
    plt.legend(loc='upper left')
    plt.savefig("C:/Users/Administrator/PycharmProjects/PythonProject/NLP/Day05/准确率对比图")
    plt.show()

#todo:13.定义将人名转换为向量的函数
def name2tensor(x):
    #将x转换为one-hot编码形式
    tensor_x = torch.zeros(len(x),n_letters)
    for idx,letter in enumerate(x):
        tensor_x[idx][all_letters.find(letter)] = 1
    return tensor_x

#todo:14.定义rnn模型的预测
def predict_rnn(x):
    #1.将输入人名转换为向量
    tensor_x = name2tensor(x)
    #2。实例化模型，并加载模型参数
    input_size = 57
    hidden_size = 128
    output_size = 18
    my_model = NameRNN(input_size, hidden_size, output_size)
    my_model.load_state_dict(torch.load('C:/Users/Administrator/PycharmProjects/PythonProject/NLP/Day05/name_class.pth'))
    #3.将向量输入模型，获取预测结果
    with torch.no_grad():
        #将数据送入模型
        h0 = my_model.init_hidden()
        output,hn = my_model(tensor_x,h0)
        #取出预测结果中topk3结果
        topv,topi = torch.topk(output,k=3,dim=1)
        for i in range(3):
            temv = topv[0][i]
            tempi = topi[0][i]
            name_class = categorys[tempi]
            print(f'需要预测的人名为：{x},预测结果为：{temv:.3f},预测的国家为：{name_class}')

#todo:15.定义lstm模型的预测
def predict_lstm(x):
    tensor_x = name2tensor(x)
    input_size = 57
    hidden_size = 128
    output_size = 18
    my_model = NameLSTM(input_size, hidden_size, output_size)
    my_model.load_state_dict(torch.load('C:/Users/Administrator/PycharmProjects/PythonProject/NLP/Day05/name_class_lstm.pth'))
    with torch.no_grad():
        h0,c0 = my_model.init_hidden()
        output,hn,cn = my_model(tensor_x,h0,c0)
        topv,topi = torch.topk(output,k=3,dim=1)
        for i in range(3):
            temv = topv[0][i]
            tempi = topi[0][i]
            name_class = categorys[tempi]
            print(f'需要预测的人名为：{x},预测结果为：{temv:.3f},预测的国家为：{name_class}')

#todo:16.定义gru模型的预测
def predict_gru(x):
    tensor_x = name2tensor(x)
    input_size = 57
    hidden_size = 128
    output_size = 18
    my_model = NameGRU(input_size, hidden_size, output_size)
    my_model.load_state_dict(torch.load('C:/Users/Administrator/PycharmProjects/PythonProject/NLP/Day05/name_class_gru.pth'))
    with torch.no_grad():
        h0 = my_model.init_hidden()
        output,hn = my_model(tensor_x,h0)
        topv,topi = torch.topk(output,k=3,dim=1)
        for i in range(3):
            temv = topv[0][i]
            tempi = topi[0][i]
            name_class = categorys[tempi]
            print(f'需要预测的人名为：{x},预测结果为：{temv:.3f},预测的国家为：{name_class}')


if __name__ == '__main__':
    predict_rnn(x='zhang')
    print('*' * 50)
    predict_lstm(x='zhang')
    print('*' * 50)
    predict_gru(x='zhang')






