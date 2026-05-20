# -*- coding:utf-8 -*-
import torch
import random
import torch.nn as nn
from dask.array import argmax
from datasets import load_dataset
from transformers import BertTokenizer,BertModel
from torch.optim import AdamW
from torch.utils.data import DataLoader,Dataset
import time
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载分词器
bert_tokenizer = BertTokenizer.from_pretrained('C:/Users/Administrator/PycharmProjects/PythonProject/NLP/Day09/03-预训练模型/bert-base-chinese')
# 加载模型
bert_model = BertModel.from_pretrained('C:/Users/Administrator/PycharmProjects/PythonProject/NLP/Day09/03-预训练模型/bert-base-chinese')
# 如果使用gpu 需要将预训练模型也放到gpu上
bert_model = bert_model.to(device)

# 自定义dataset对象
class MyDataset(Dataset):
    def __init__(self,data_path):
        super().__init__()
        # 加载数据集
        dataset = load_dataset('csv',data_files=data_path,split='train')
        # 筛选数据集
        self.dataset = dataset.filter(lambda x : len(x['text']) > 44)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, item):
        # 因为要做NSP任务：即给你两句话，判断第二个句子是不是第一个句子的下一句，因此我们要构建句子对
        #(Seq1,Seq2)-->标签可以为0或者1，0表示句2不是句1的下一句，1表示句2是句1的下一句
        sequence = self.dataset[item]['text']
        label = 1
        seq1 = sequence[:22]
        seq2 = sequence[22:44]
        # 构建负样本
        if random.randint(0,1) == 0:
            # 随机选择一个索引
            j = random.randint(0,len(self.dataset)-1)
            seq2 = self.dataset[j]['text'][22:44]
            label = 0
        return seq1,seq2,label

def collate_fn(data):
    # data:[(seq1,seq2,label),...]
    # print(f'data:{data}')
    # 取出每个句子对
    sequences = [value[:2]for value in data]
    # 取出所有标签
    labels = [value[-1] for value in data]
    # print(f'sequences:{sequences}')
    # print(f'labels:{labels}')
    # 对数据进行编码
    inputs = bert_tokenizer(sequences,padding='max_length',truncation=True,max_length=50,return_tensors='pt')
    # print(f'inputs:{inputs}')
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    attention_mask = inputs['attention_mask']
    labels = torch.tensor(labels,dtype=torch.long)
    return input_ids,token_type_ids,attention_mask,labels

def  dm_test_dataloader():
    # 实例化数据集
    train_dataset = MyDataset('./03-data/train.csv')
    # 实例化数据加载器
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=8,shuffle=True,drop_last= True,collate_fn=collate_fn)
    return train_dataloader

# todo:3.定义模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 因为是微调，所以承接的是bert预训练模型的输出结果，是768维
        self.out = nn.Linear(768,2)
    def forward(self,input_ids, attention_mask, token_type_ids):
        # 将input_ids, attention_mask, token_type_ids三个参数传入bert模型中，需要注意的是bert模型参数不参与更新，需要with torch.no_grad()
        with torch.no_grad():
            bert_outputs = bert_model(input_ids = input_ids,
                                 attention_mask = attention_mask,
                                 token_type_ids = token_type_ids)
        result = self.out(bert_outputs.pooler_output)
        return result

def model2train():
    # 1.实例化dataset
    train_dataset = MyDataset(data_path = './03-data/train.csv')
    # 2.实例化dataloader
    my_dataloader= DataLoader(dataset=train_dataset, shuffle = True, batch_size = 8, collate_fn = collate_fn,drop_last= True)
    # 3.实例化模型
    my_model = MyModel().to(device)
    # 4.实例化优化器
    my_optimizer = AdamW(params = my_model.parameters(),lr = 3e-5)
    # 5.实例化损失函数
    my_loss = nn.CrossEntropyLoss()
    # 6.强调预训练模型的参数不参与更新
    for param in bert_model.parameters():
        param.requires_grad_(False)
    # 7.开始训练
    my_model.train()
    epochs = 3
    for epoch in range(epochs):
        # 开始时间
        start_time = time.time()
        for idx,(input_ids, attention_mask, token_type_ids, label_y) in enumerate(tqdm(my_dataloader)):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            label_y = label_y.to(device)
            output = my_model(input_ids, attention_mask, token_type_ids)
            # 计算损失
            loss = my_loss(output,label_y)
            # print(loss)
            # 梯度清零
            my_optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            my_optimizer.step()
            # 每5步打印训练日志
            if idx % 20 ==0:
                # 取出一个批次样本的预测结果
                predicts = torch.argmax(output,dim = -1)
                ave_predict = (predicts == label_y).sum().item() / len(label_y)
                print(f'第{epoch}轮第{idx}步训练结果：损失为{loss.item():.5f}，准确率为{ave_predict},时间：{time.time()-start_time}')
    # 保存模型
    torch.save(my_model.state_dict(),'./save_model/NSP.bin')

def model2eval():
    # 1.实例化dataset
    test_dataset = MyDataset(data_path = './03-data/test.csv')
    # 2.实例化dataloader
    test_dataloader = DataLoader(dataset = test_dataset,shuffle = False,batch_size = 8,collate_fn = collate_fn,drop_last = True)
    # 3.实例化模型
    my_model = MyModel().to(device)
    my_model.load_state_dict(torch.load('save_model/NSP.bin'))
    my_model.eval()
    total = 0 # 总样本数
    acc_num = 0 # 预测正确的样本数
    # 4.开始预测
    for idx,(input_ids, attention_mask, token_type_ids, labels) in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            output = my_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            if idx%5 == 0:
                predicts = torch.argmax(output,dim = -1)
                # 预测正确的样本数
                acc_num += (predicts == labels).sum().item()
                total += len(labels)
                # 每5步打印训练日志
                print(f'第{idx}步训练结果：准确率为{acc_num/total:.5f}',end= ' ')
                print(f'取出样本:{bert_tokenizer.decode(input_ids[0],skip_special_tokens=True)}',end= ' ')
                print(f'预测标签为{predicts[0]},实际标签为{labels[0]}')
                print('*'*80)

if __name__ == '__main__':
    model2eval()
