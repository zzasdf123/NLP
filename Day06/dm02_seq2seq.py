#-*- coding: utf-8 -*-
import re                                           #用于正则表达
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import time
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

#设备选择，我们可以通过使用cuda或者cpu来运行代码
device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
SOS_token = 0
EOS_token = 1
max_length = 10
data_path = './eng-fra-v2.txt'
#todo：1.定义字符串的清洗函数
def norm_string(s):
    #s代表输入的字符串
    s = s.lower().strip()
    #替换特殊字符
    s = re.sub(r'([.?!])',r' \1',s)
    #将字符串中所有出大小写字母与.?!替换为空
    s = re.sub(r'([^a-zA-Z.?!])+',r' ',s)
    return s

#todo：2.读取文件，获取样本并获取英文词典以及法文词典
def get_data():
    #2.1读取文档数据
    with open('./eng-fra-v2.txt') as fr:
        sequences = fr.read().strip().split('\n')
        #print(type(sequences))
        #print(sequences[:2])
    #2.2获得pair对
    # ['i m .\tj ai ans .']-->[['i m .', 'j ai ans .']]-->[[英文句子，法文句子]，[英文句子，法文句子]，...]
    # temp_pairs = []
    # my_pairs = []
    # for line in sequences:
    #     for s in line.split('\t'):
    #         temp_pairs.append(norm_string(s))
    #     my_pairs.append(temp_pairs)
    #     temp_pairs = []
    my_pairs = [[norm_string(s) for s in line.split('\t')] for line in sequences]
    #print(my_pairs[:2])
    #2.3遍历句子，获得词典
    english_words2index = {'SOS':0,'EOS':1}
    english_words_len = 2
    french_words2index = {'SOS':0,'EOS':1}
    french_words_len = 2
    for pair in my_pairs:
        #构建英文字典
        for word in pair[0].split(' '):
            #print(word)
            if word not in english_words2index:
                english_words2index[word] = english_words_len
                # english_words2index[word] = len(english_words2index)
                english_words_len += 1
        #构建法文字典
        for word in pair[1].split(' '):
            if word not in french_words2index:
                # french_words2index[word] = len(french_words2index)
                french_words2index[word] = french_words_len
                french_words_len += 1
    #获得index2words
    english_index2words = {v:k for k,v in english_words2index.items()}
    french_index2words = {v:k for k,v in french_words2index.items()}
    return english_words2index,english_index2words,english_words_len,french_words2index,french_index2words,french_words_len,my_pairs

result = get_data()
#todo：3.定义数据集
class MyDataset(Dataset):
    def __init__(self,my_pairs):
        super().__init__()
        #获取样本对
        self.my_pairs = my_pairs
        #获取样本总量
        self.sample_len = len(my_pairs)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, item):
        #异常值修正
        item = min(max(item,0),self.sample_len -1)
        #根据索引取样本
        x = self.my_pairs[item][0]
        #print(x)
        y = self.my_pairs[item][1]
        #print(y)
        #样本x数值化
        x2index = [result[0][word] for word in x.split(' ')]
        x2index.append(EOS_token)    #编码器阶段 可加可不加
        #将上述结果转换为张量
        x_tensor = torch.tensor(x2index,dtype=torch.long,device=device)
        #样本y数值化
        y2index = [result[3][word] for word in y.split(' ')]
        y2index.append(EOS_token)    #解码器阶段 必须加
        y_tensor = torch.tensor(y2index,dtype=torch.long,device=device)
        return x_tensor,y_tensor

#todo:4.实例化Dataloader对象
def get_dataloader():
    #1.实例化dataset对象
    my_dataset = MyDataset(result[-1])
    #2.实例化dataloader对象
    my_dataloader = DataLoader(dataset=my_dataset,batch_size=1,shuffle=True)
    return my_dataloader

#todo:5.构建GRU模型
class EncoderGRU(nn.Module):
    def __init__(self,english_words_num,hidden_size):
        super().__init__()
        #english_words_num:英文单词数量,即需要被Embedding的词数量
        self.english_words_num = english_words_num
        #hidden_size:词嵌入维度
        self.hidden_size = hidden_size
        #构建Embedding层
        self.embed = nn.Embedding(english_words_num,hidden_size)
        #构建GRU层，注意：这里的输入和输出维度一致，并设置了batch_first = True 意味着gru的输入是[batch_size,seq_len,embedding_dims]
        self.gru = nn.GRU(hidden_size,hidden_size,batch_first= True)

    def forward(self,x,h0):
        #x是dataloader中的数据，维度是[batch_size,seq_len]，需要先将其送入embedding层将其转换为[batch_size,seq_len,embedding_dims]
        #h0是初始化的隐藏层状态
        tensor_x = self.embed(x)
        output,hn = self.gru(tensor_x,h0)
        return output,hn

    def inithidden(self):
        #注意，自己定义的张量需要单独放到GPU上
        return torch.zeros(1,1,self.hidden_size,device=device)
6
#todo:6.定义不带attention的解码器
class DecoderGRU(nn.Module):
    def __init__(self,french_words_num,hidden_size):
        super().__init__()
        #french_words_num:代表法语单词数量，即需要被Embedding的词数量：4345
        self.french_words_num = french_words_num
        #hidden_size:词嵌入维度：256
        self.hidden_size = hidden_size
        #定义词嵌入层
        self.embed = nn.Embedding(french_words_num,hidden_size)
        #定义GRU层
        self.gru = nn.GRU(hidden_size,hidden_size,batch_first=True)
        #定义输出层
        self.out = nn.Linear(hidden_size,french_words_num)

    def forward(self,y0,h0):
        #y0是dataloader中的数据，维度是[batch_size,seq_len]:[1,1] 一个一个送入解码器
        #h0是初始化的隐藏层状态:[1,1,256]
        #将y0送入embedding层将其转换为[batch_size,seq_len,embedding_dims]：[1,1,256]
        embed_y0 = self.embed(y0)
        #将embed_y0经过relu激活函数，防止过拟合
        relu_y0 = F.relu(embed_y0)
        #将embed_y0送入gru层
        output,hn = self.gru(embed_y0,h0)
        #将gru的输出送入输出层,output：[1.1.256]--》[1,256]-->[1,4345]
        result0 = self.out(output[0])
        result = F.log_softmax(result0)
        return result,hn

    def inithidden(self):
        return torch.zeros(1,1,self.hidden_size).cuda(device)

#todo:7.测试模型
def testmodel():
    #1.获得数据加载器
    my_dataloader = get_dataloader()
    #2.实例化模型
    hidden_size = 256
    encoder = EncoderGRU(len(result[0]),hidden_size)
    encoder = encoder.to(device=device)
    #3.实例化解码器
    decoder = DecoderGRU(len(result[3]),hidden_size)
    decoder = decoder.to(device=device)
    #4.获得数据，送入seq2seq架构得到结果
    for x,y in my_dataloader:
        #将x送入编码器，得到编码器结果
        h0 = encoder.inithidden()
        encoder_output,encoder_hidden = encoder(x,h0)
        #将编码器结果送入解码器，得到解码器结果
        hidden =encoder_hidden
        for idx in range(y.shape[1]):
            temp_vector = y[0][idx].view(1,-1)
            decoder_output,hidden = decoder(temp_vector,hidden)
            print(decoder_output.shape)
            break
        break

#todo:8.定义带attention的解码器
class AttentionDecoder(nn.Module):
    def __init__(self,french_words_num,hidden_size,dropout_p=0.1,max_length=max_length):
        super().__init__()
        #french_words_num:代表法语单词数量，即需要被Embedding的词数量：4345
        self.french_words_num = french_words_num
        #hidden_size:词嵌入维度：256
        self.hidden_size = hidden_size
        #dropout_p:dropout概率
        self.dropout_p = dropout_p
        #max_length:句子最大长度
        self.max_length = max_length
        #定义词嵌入层
        self.embed = nn.Embedding(self.french_words_num,self.hidden_size)
        #定义第一个全连接层：计算注意力权重
        self.atten = nn.Linear(self.hidden_size * 2,self.max_length)
        #定义第二个全连接层：将注意力权重和编码器结果进行加权求和
        self.atten_combin = nn.Linear(self.hidden_size * 2,self.hidden_size)
        #定义GRU层
        self.gru = nn.GRU(self.hidden_size,self.hidden_size,batch_first=True)
        #定义输出层
        self.out = nn.Linear(self.hidden_size,self.french_words_num)
        #定义dropout层
        self.dropout = nn.Dropout(p=self.dropout_p)

    def forward(self, Q, K, V):
        #Q:当前解码时，预测的上一个单词，最开始代表SOS：[1,1]
        #K:解码器上一层隐藏层输出结果，最开始是编码器最后一个单词的隐藏层输出结果：[1,1,hidden_size]——》[1,1,256]
        #V:编码器每一个时间步隐藏层的输出结果，由于我们规定的最大句子长度，所以为：[max_len,hidden_size]——》[10,256]
        #1.需要将Q输出Embedding层
        embed_x = self.embed(Q)
        #2.将Q(embed_x)送入dropout层,防止过拟合
        dropout_x = self.dropout(embed_x)
        #3.选择第一种注意力计算方法计算注意力
        #3.1将Q(dropout_x)与K拼接后送入第一个全连接层，计算注意力权重  [1,1,512]-->经过Linear层[512,10]-->[1,1,10]
        atten_weight = F.softmax(self.atten(torch.cat((dropout_x, K),dim=-1)), dim=-1)
        #3.2将atten_weight[1,1,10]和V[10,256]进行相乘  -->[1,1,256]
        temp_vc = torch.bmm(atten_weight,V.unsqueeze(dim=0))
        #3.3将temp_vc[1,1,256]和Q[1,1,256]拼接后送入第二个全连接层
        attention_output = F.relu(self.atten_combin(torch.cat((temp_vc,dropout_x),dim=-1)))
        #4将attention_output[1,1,256]与K[1,1,256]送入GRU，得到最终的输出结果 [1,1,256]
        output,hidden = self.gru(attention_output, K)
        #5将output降维送入输出层 [1,256]-->[1,4345]
        result = self.out(output[0])
        return F.log_softmax(result,dim=-1),hidden,atten_weight

#todo:9.定义带attention的测试模型
def Attention_modeltest():
    # 1.获得数据加载器
    my_dataloader = get_dataloader()
    # 2.实例化模型
    hidden_size = 256
    encoder = EncoderGRU(len(result[1]), hidden_size)
    encoder = encoder.to(device=device)
    # 3.实例化解码器
    decoder = AttentionDecoder(len(result[3]),hidden_size)
    decoder = decoder.to(device=device)
    # 4.获得数据，送入seq2seq架构得到结果
    for x, y in my_dataloader:
        # 将x送入编码器，得到编码器结果
        h0 = encoder.inithidden()
        encoder_output, encoder_hidden = encoder(x, h0)
        # print(f'encoder_output:{encoder_output}')
        # print(f'encoder_output.shape:{encoder_output.shape}')
        # print(f'encoder_hidden.shape:{encoder_hidden.shape}')
        #定义中间语义张量C
        encoder_output_C = torch.zeros(max_length, encoder.hidden_size,device= device)
        #将真实的x编码后的值赋给C，其余为0
        #encoder_output:[1,5,256]
        for idx in range(encoder_output.shape[1]):
            encoder_output_C[idx] = encoder_output[0,idx]
        # print(f'encoder_output_C:{encoder_output_C}')
        #解码，一个token一个token的去解码
        for j in range(y.shape[1]):
            temp_vec = y[0][j].view(1,-1)
            hidden = encoder_hidden
            output,hidden,atten_weight = decoder(Q=temp_vec,K=hidden,V=encoder_output_C)
            print(f'output:{output.shape}')
            print(f'hidden:{hidden.shape}')
            print(f'atten_weight:{atten_weight.shape}')
        break

#todo:10.定义模型的训练函数
#模型的超参数
my_lr = 1e-4
epochs = 1
teacher_forcing_ratio = 0.5
def train_attention_seq2seq():
    #1.获得数据加载器
    my_dataloader = get_dataloader()
    #2.实例化编码器与解码器
    english_words_num = len(result[0])
    hidden_size = 256
    encoder = EncoderGRU(english_words_num,hidden_size).to(device = device)
    french_words_num = len(result[3])
    decoder = AttentionDecoder(french_words_num,hidden_size).to(device = device)
    #3.实例化优化器
    encoder_optimizer = optim.Adam(encoder.parameters(),lr=my_lr)
    decoder_optimizer = optim.Adam(decoder.parameters(),lr=my_lr)
    #4.实例化损失函数对象
    criterion = nn.NLLLoss()
    #5.定义存储损失的列表
    plot_loss_list = []
    #6.开始外部循环
    for epoch in range(epochs):
        #定义训练日志参数
        print_loss_total,plot_loss_total = 0.0,0.0
        start_time = time.time()
        #7.1开始内部循环
        for item,(x,y) in enumerate(tqdm(my_dataloader),start=1):
            #7.2开始调用内容训练函数
            #print(f'x:{x}')
            #print(f'y:{y}')
            my_loss = train_iter(x,y,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion)
            #print(f'my_loss:{my_loss}')
            print_loss_total += my_loss
            plot_loss_total += my_loss
            #7.3每隔1000步打印一次日志
            if item % 1000 == 0:
                ave_loss = print_loss_total / 1000
                print(f'训练轮数：{epoch},训练次数：{item},当前损失：{ave_loss:.6f},时间：{time.time()-start_time:.2f}')
                print_loss_total = 0.0
            #每隔100步保存平均损失，画图
            if item % 100 == 0:
                plot_loss_list.append(plot_loss_total/100)
                plot_loss_total = 0.0
        torch.save(encoder.state_dict(),f'./save_model/seq2seq_encoder_{epoch}.pth')
        torch.save(decoder.state_dict(),f'./save_model/seq2seq_decoder_{epoch}.pth')
    #8.绘图
    plt.figure(0)
    plt.plot(plot_loss_list)
    plt.savefig('seq2seq_loss_fig')
    plt.show()

#todo:11.定义内容训练函数
def train_iter(x,y,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion):
    # x:代表英文原始数据输入，[batch_size,seq_len] -->[1,6]
    # y:代表法文原始数据输入.[batch_size,seq_len] -->[1,8]
    # encoder:编码器模型
    # decoder:解码器模型
    # encoder_optimizer:编码器优化器
    # decoder_optimizer:解码器优化器
    # criterion:损失函数对象
    #1.将x与初始化h0送入编码器，得到编码器结果
    h0 = encoder.inithidden()
    encoder_output,encoder_hidden = encoder(x,h0)
    #2.中间语义张量C，代表V
    encoder_output_c = torch.zeros(max_length,encoder.hidden_size).to(device=device)
    for idx in range(x.shape[1]):
        encoder_output_c[idx] = encoder_output[0,idx]
    #print(f'c:{encoder_output_c.shape}')
    #3.将encoder_hidden代表解码器上一时间步的隐藏层结果，当作Key
    #这里解码器的第一个时间步的隐藏层输入用编码器的最后一层隐藏层输出进行初始化
    decoder_hidden = encoder_hidden
    #print(f'decoder_hidden:{decoder_hidden.shape}')
    #定义解码器开始解码的第一个字符为SOS
    input_y = torch.tensor([[SOS_token]]).to(device=device)
    #print(f'input_y:{input_y}')
    #4.定义变量
    my_loss = 0.0
    y_len = y.shape[1]
    use_teacher_forcing = True if random.random() <teacher_forcing_ratio else False
    #5.将数据送入解码器，进行解码
    if use_teacher_forcing:
        for idx in range(y_len):
            #获得预测结果
            decoder_output,decoder_hidden,attention_weight = decoder(Q=input_y,K=decoder_hidden,V=encoder_output_c)
            #print(f'decoder_output:{decoder_output.shape}')
            #获得真实结果
            tatget_y = y[0][idx].view(1)
            my_loss += criterion(decoder_output,tatget_y)
            #print(f'my__loss:{my_loss}')
            #用真实的label当作输入
            input_y = y[0][idx].view(1,-1)
    else:
        for idx in range(y_len):
            decoder_output,decoder_hidden,attention_weight = decoder(Q=input_y,K=decoder_hidden,V=encoder_output_c)
            target_y = y[0][idx].view(1)
            my_loss += criterion(decoder_output,target_y)
            topv,topi = torch.topk(decoder_output,1)
            if topi.item() == EOS_token:
                break                            #可要可不要，训练阶段训练数据一般都是有标签的，所以这里可以不加
            input_y = topi.detach()
    #6.梯度清零+反向传播+优化器更新
    encoder.zero_grad()
    decoder.zero_grad()
    my_loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return my_loss.item()/y_len

#todo:12.模型预测
def seq2seq_evaluate(tensor_x,encoder,decoder):
    with torch.no_grad():
        #1.将x送入编码器，得到编码器结果
        encoder_hidden = encoder.inithidden()
        encoder_output,encoder_hidden =encoder(tensor_x,encoder_hidden)
        #print(f'encoder_output:{encoder_output.shape}')
        #2.中间语义张量C，
        encoder_output_c = torch.zeros(max_length,encoder.hidden_size,device=device)
        for idx in range(encoder_output.shape[1]):
            encoder_output_c[idx] = encoder_output[0,idx]
        #3.将encoder_hidden代表解码器上一时间步的隐藏层结果，当作Key
        decoder_hidden = encoder_hidden
        #4.定义解码器开始解码的输入字符SOS
        input_y = torch.tensor([SOS_token]).view(1,-1).to(device= device)
        #定义存储解码出法文的结果列表与解码出每个token的注意力权重张量
        decoder_list = []
        attention_weight_list = torch.zeros(max_length,max_length)
        for idx in range(max_length):
            decoder_output,decoder_hidden,attention_weight = decoder(input_y,decoder_hidden,encoder_output_c)
            #print(decoder_output.shape)
            #print(attention_weight.shape)
            topv,topi = torch.topk(decoder_output,k=1)
            #print(f'topi = {topi}')
            #将每一步的预测权重进行赋值
            attention_weight_list[idx] = attention_weight[0,0]
            #attention_weight_list[idx] = attention_weight   效果同上
            if topi.item() == EOS_token:
                decoder_list.append('EOS')
                break
            else:
                decoder_list.append(result[4][topi.item()])
                #print(decoder_list)
            input_y = topi
        return decoder_list,attention_weight_list[:idx+1]

def use_evaluate():
    #1.定义模型路径
    encoder_path = './save_model/seq2seq_encoder_0.pth'
    decoder_path = './save_model/seq2seq_decoder_{epoch}.pth'
    #2.实例化编码器与解码器
    english_words_num = len(result[0])
    fresh_words_num = len(result[3])
    hidden_size = 256
    encoder = EncoderGRU(english_words_num,hidden_size).to(device = device)
    encoder.load_state_dict(torch.load(encoder_path))
    decoder = AttentionDecoder(fresh_words_num,hidden_size).to(device= device)
    decoder.load_state_dict(torch.load(decoder_path))
    #用于加载其他人训练的模型参数，map_location是调整模型参数的存储位置，strict=False表示忽略模型参数中不存在的参数
    #decoder.load_state_dict(torch.load(decoder_path,map_location='cpu',strict=False))
    #print(f'encoder:{encoder}')
    #print(f'decoder:{decoder}')
    #3.输入数据
    my_sample_pairs = [['i m impressed with your french .', 'je suis impressionne par votre francais .'],
                      ['i m more than a friend .', 'je suis plus qu une amie .'],
                      ['she is beautiful like her mother .', 'elle est belle comme sa mere .']]
    print('my_sample_pairs--->', len(my_sample_pairs))
    #4.遍历每一个输入样本，将英文句子转换为张量送入模型，并将输出结果与法文进行对比
    for item,pairs in enumerate(my_sample_pairs):
        x = pairs[0]
        y = pairs[1]
        temp_x = [result[0][word] for word in x.split(' ')]
        tensor_x = torch.tensor(temp_x,dtype=torch.long,device=device).view(1,-1)
        decoder_list,attention_weight_list =seq2seq_evaluate(tensor_x,encoder,decoder)
        #print(f'decoder_list:{decoder_list}')
        #print(f'attention_weight_list:{attention_weight_list.shape}')
        predit_y = ' '.join(decoder_list)
        print(f'x:{x}')
        print(f'y:{y}')
        print(f'predit_y:{predit_y}')

def show_attention():
    encoder_path = './save_model/seq2seq_encoder_0.pth'
    decoder_path = './save_model/seq2seq_decoder_{epoch}.pth'
    hidden_size = 256
    encoder = EncoderGRU(len(result[0]),hidden_size).to(device= device)
    encoder.load_state_dict(torch.load(encoder_path))
    decoder = AttentionDecoder(len(result[3]),hidden_size).to(device= device)
    decoder.load_state_dict(torch.load(decoder_path))
    sentence = 'we are both teachers .'
    temp_x = [result[0][word] for word in sentence.split(' ')]
    tensor_x = torch.tensor(temp_x,dtype=torch.long,device=device).view(1,-1)
    decoder_list,attention_weight_list=seq2seq_evaluate(tensor_x,encoder,decoder)
    pred_y = ' '.join(decoder_list)
    print(f'sentence:{sentence}')
    print(f'pred_y:{pred_y}')
    plt.matshow(attention_weight_list.detach().numpy())
    plt.savefig('attention_show')
    plt.show()

if __name__ == '__main__':
   use_evaluate()


