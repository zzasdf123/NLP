# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from dask.array import argmax
from datasets import load_dataset
from transformers import BertTokenizer,BertModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载分词器
bert_tokenizer = BertTokenizer.from_pretrained('C:/Users/Administrator/PycharmProjects/PythonProject/NLP/Day09/03-预训练模型/bert-base-chinese')
# 加载模型
bert_model = BertModel.from_pretrained('C:/Users/Administrator/PycharmProjects/PythonProject/NLP/Day09/03-预训练模型/bert-base-chinese')
# 如果使用gpu 需要将预训练模型也放到gpu上
bert_model = bert_model.to(device)

def collate_fn(data):
    # data为一个批次样本的列表，列表中的每个元素都是一个字典，字典中包含text和label两个键值对
    # print(f'data:{data}')
    senquences = [value['text'] for value in data]
    # 进行tokenizer编码
    inputs = bert_tokenizer(senquences,return_tensors='pt',padding='max_length',max_length=32,truncation=True)
    # print(f'inputs:{inputs}')
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    attention_mask = inputs['attention_mask']
    # 取出第十六个位置的token作为标签 ,必须进行clone(),相当于深copy
    labels = input_ids[:,16].clone()
    # 将16位置的token替换为mask
    # input_ids[:,16]= bert_tokenizer.get_vocab()[bert_tokenizer.mask_token]
    input_ids[:,16] = bert_tokenizer.mask_token_id
    # print(f'input_ids:{input_ids}')
    # print(f'labels:{labels}')
    labels = torch.tensor(labels,dtype=torch.long)
    return input_ids,token_type_ids,attention_mask,labels

def dm_test_dataloader():
    # 获取训练集数据
    train_dataset = load_dataset('csv',data_files='./03-data/train.csv',split='train')
    # 对训练数据进行筛选
    new_train_dataset = train_dataset.filter(lambda x:len(x['text']) > 35)
    # print(f'筛选后的数据集大小为：{len(new_train_dataset)}')
    # 将数据进行dataloader的封装
    train_dataloader = DataLoader(dataset=new_train_dataset,shuffle=True,batch_size=8,collate_fn=collate_fn,drop_last=True)
    # 遍历dataloader验证自定义函数
    # for input_ids,token_type_ids,attention_mask,labels in train_dataloader:
    #     print(f'input_ids:{input_ids.shape}')
    #     print(f'token_type_ids:{token_type_ids.shape}')
    #     print(f'attention_mask:{attention_mask.shape}')
    #     print(f'labels:{labels.shape}')
    #     print('这是测试')
    #     break
    return train_dataloader

# 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 获取bert的输出结果维度 bert_tokenizer.vocab_size:21128
        self.out = nn.Linear(768,bert_tokenizer.vocab_size)

    def forward(self,input_ids,token_type_ids,attention_mask):
        with torch.no_grad():
            bert_outputs = bert_model(input_ids=input_ids,
                                      token_type_ids=token_type_ids,
                                      attention_mask=attention_mask)
            # print(f'bert_outputs:{bert_outputs}')
            # bert_outputs['last_hidden_state'].shape————》[8,32,768]
            # 只取出第十六的位置的张量送入输出层 ————》[8,21128]
        outputs = self.out(bert_outputs['last_hidden_state'][:,16])
        return  outputs

# 模型训练
def model2train():
    # 1.实例化数据集
    train_dataset = load_dataset('csv',data_files='./03-data/train.csv',split='train')
    # 2.截取文本长度大于32的数据
    new_train_dataset = train_dataset.filter(lambda x :len(x['text']) > 32)
    # 3.实例化数据加载器
    train_dataloader = DataLoader(dataset=new_train_dataset,shuffle=True,batch_size=8,collate_fn=collate_fn,drop_last=True)
    # 4.实例化模型
    my_model = MyModel().to(device)
    # 5.实例化优化器
    optimizer = AdamW(my_model.parameters())
    # 6.实例化损失函数
    criterion = nn.CrossEntropyLoss()
    # 7.模型训练
    my_model.train()
    for param in bert_model.parameters():
        param.requires_grad_(False)
    epochs = 5
    correct_num = 0
    total_num = 0
    for epoch in range(epochs):
        start_time = time.time()
        for idx,(input_ids,token_type_ids,attention_mask,labels) in enumerate(tqdm(train_dataloader)):
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = my_model(input_ids,token_type_ids,attention_mask)
            # 计算损失函数
            my_loss = criterion(outputs,labels)
            # 梯度清零 + 反向传播 + 梯度更新
            optimizer.zero_grad()
            my_loss.backward()
            optimizer.step()
            # 打印训练日志
            if idx % 20 ==0:
                # 取出一个批次样本的预测结果
                predicts = torch.argmax(outputs,dim=-1)
                # 计算正确的样本的数量
                correct_num += (predicts == labels).sum().item()
                total_num += len(labels)
                print(f'第{epoch}轮第{idx}步的训练损失为：{my_loss.item()}:.5f,训练准确率为：{correct_num/total_num:.5f},训练时间：{time.time()-start_time:.5f}s')
    torch.save(my_model.state_dict(),'./save_model/fill_mask.bin')

# 模型测试
def model2test():
    # 1.实例化数据集
    test_dataset = load_dataset('csv',data_files='./03-data/test.csv',split='train')
    new_test_dataset = test_dataset.filter(lambda x:len(x['text']) > 32)
    # 2.实例化数据加载器
    test_dataloader = DataLoader(dataset=new_test_dataset,shuffle=True,batch_size=8,collate_fn=collate_fn,drop_last=True)
    # 3.实例化模型
    my_model = MyModel().to(device)
    # 4.加载模型参数
    my_model.load_state_dict(torch.load('./save_model/fill_mask.bin'))
    my_model.eval()
    correct_num = 0
    total_num = 0
    for idx,(input_ids,token_type_ids,attention_mask,labels) in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            output = my_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            if idx % 20 == 0:
                predicts = torch.argmax(output, dim=-1)
                # 预测正确的样本数
                correct_num  += (predicts == labels).sum().item()
                total_num += len(labels)
                # 每5步打印训练日志
                print(f'第{idx}步训练结果：准确率为{correct_num / total_num:.5f}', end=' ')
                print(f'取出样本:{bert_tokenizer.decode(input_ids[0])}', end=' ')
                print(f'预测标签为{bert_tokenizer.decode(predicts[0])},实际标签为{bert_tokenizer.decode(labels[0])}')
                print('*' * 80)

if __name__ == '__main__':
    model2test()