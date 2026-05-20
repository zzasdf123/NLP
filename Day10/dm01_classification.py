# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
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
# todo:1.读取数据
def read_data():
    # 1.读取训练集
    train_dataset = load_dataset('csv',data_files='./03-data/train.csv',split='train')
    # 三种使用方法
    # print(f'样本总个数：{len(train_dataset)}')
    # print(f'第一个样本为：{train_dataset[0]}')
    # print(f'前三个样本为{train_dataset[:3]}"')
    # 2.读取测试集
    test_dataset = load_dataset('csv',data_files='./03-data/test.csv',split='train')
    # print(f'样本总个数：{len(test_dataset)}')
    # print(f'第一个样本为：{test_dataset[0]}')
    # print(f'前三个样本为{test_dataset[:3]}"')
    # # 3.读取验证集
    valid_dataset = load_dataset('csv',data_files='./03-data/validation.csv',split='train')
    # print(f'样本总个数：{len(valid_dataset)}')
    # print(f'第一个样本为：{valid_dataset[0]}')
    # print(f'前三个样本为{valid_dataset[:3]}"')
    return train_dataset,test_dataset,valid_dataset

def collate_fn(data):
    # 自定义函数，目的是对dataset中的数据进行处理
    # print(f'自定义函数参数data展示：{data}')
    # print(f'自定义函数参数data展示：{len(data)}')
    # print(f'自定义函数参数data展示：{data[0]}')
    # 获得一个批次样本的所有句子
    sentences = [value['text'] for value in data]
    # print(f'sentences展示：{sentences}')
    # 获取一个批次样本的所有标签
    label = [value['label'] for value in data]
    # print(f' label展示：{ label}')
    # 对一个批次的句子进行张量的转换，一定要对齐长度
    inputs = bert_tokenizer(text = sentences,
                                       padding = 'max_length',
                                       truncation = True,
                                       max_length = 200,
                                       return_tensors = 'pt')
    # print(f'inputs:{inputs}')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']
    label_y = torch.tensor(label,dtype=torch.long)
    return input_ids, attention_mask, token_type_ids, label_y

# todo:2.将dataset封装为dataloader
def get_dataloader():
    train_dataset, _, _ = read_data()
    train_dataloader = DataLoader(dataset = train_dataset,
                                  batch_size = 8,
                                  collate_fn = collate_fn,
                                  shuffle =  True,
                                  drop_last =  True)
    # 一定要迭代 train_dataloader ，才能查验collate_fn
    # for input_ids, attention_mask, token_type_ids, label_y in train_dataloader:
    #     print(f'input_ids:{input_ids.shape}')
    #     print(f'attention_mask:{attention_mask.shape}')
    #     print(f'token_type_ids:{token_type_ids.shape}')
    #     print(f'label_y:{label_y.shape}')
    #     break
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
        # print(f'bert_outputs:{bert_outputs}')
        # last_hidden_state————》[8,200,768]
        # print(f'last_hidden_state:{bert_outputs.last_hidden_state.shape}')
        # pooler_output————》[8,768]  代表每个样本的CLS对应的隐藏层输出结果，代表整个句子的语义
        # print(f'pooler_output:{bert_outputs.pooler_output.shape}')
        # 将bert编码之后的结果pooler_output送入输出层
        result = self.out(bert_outputs.pooler_output)
        return result

# todo:4.模型训练
def model2train():
    # 1.实例化dataset
    train_dataset = load_dataset('csv',data_files='./03-data/train.csv',split='train')
    # 2.实例化dataloader
    my_dataloader= DataLoader(dataset=train_dataset, shuffle = True, batch_size = 8, collate_fn = collate_fn,drop_last= True)
    # 3.实例化模型
    my_model = MyModel().to(device)
    # 4.实例化优化器
    my_optimizer = AdamW(params = my_model.parameters(),lr = 1e-5)
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
            my_model.train()
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
            if idx % 5 ==0:
                # 取出一个批次样本的预测结果
                predicts = torch.argmax(output,dim = -1)
                ave_predict = (predicts == label_y).sum().item() / len(label_y)
                print(f'第{epoch}轮第{idx}步训练结果：损失为{loss.item():.5f}，准确率为{ave_predict},时间：{time.time()-start_time}')
    # 保存模型
    torch.save(my_model.state_dict(),'./save_model/classification.bin')

# todo:5.模型预测
def model2eval():
    # 1.实例化dataset
    test_dataset = load_dataset('csv',data_files='./03-data/test.csv',split='train')
    # 2.实例化dataloader
    test_dataloader = DataLoader(dataset = test_dataset,shuffle = False,batch_size = 8,collate_fn = collate_fn,drop_last = True)
    # 3.实例化模型
    my_model = MyModel().to(device)
    my_model.load_state_dict(torch.load('save_model/classification.bin'))
    my_model.eval()
    total = 0 # 总样本数
    acc_num = 0 # 预测正确的样本数
    # 4.开始预测
    for idx,(input_ids, attention_mask, token_type_ids, label_y) in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            label_y = label_y.to(device)
            output = my_model(input_ids, attention_mask, token_type_ids)
            if idx%5 == 0:
                predicts = torch.argmax(output,dim = -1)
                # 预测正确的样本数
                acc_num += (predicts == label_y).sum().item()
                total += len(label_y)
                # 每5步打印训练日志
                print(f'第{idx}步训练结果：准确率为{acc_num/total:.5f}',end= ' ')
                print(f'取出样本:{bert_tokenizer.decode(input_ids[0],skip_special_tokens=True)}',end= ' ')
                print(f'预测标签为{predicts[0]},实际标签为{label_y[0]}')
                print('*'*80)

if __name__ == '__main__':
    model2eval()