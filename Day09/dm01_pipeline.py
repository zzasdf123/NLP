# -*- coding:utf-8 -*-
import torch
from transformers import pipeline
# todo:1.完成情感分类任务
def dm_test_classification():
    # 调用pipeline方法，直接返回模型
    # my_model = pipeline(task = 'sentiment-analysis',model = './03-预训练模型/chinese_sentiment')
    my_model = pipeline(task='text-classification', model='./03-预训练模型/chinese_sentiment')
    # 调用模型
    # result = my_model('我爱北京天安门,天安门上太阳升')
    result = my_model('我讨厌北京天安门')
    print(f'情感分类结果:{result}')

# todo:2.完成特征提取任务
def dm_test_feature_extraction():
    my_model = pipeline(task='feature-extraction', model='./03-预训练模型/bert-base-chinese')
    # 调用模型
    result = my_model('我爱北京天安门,天安门上太阳升')
    # 预训练模型输出结果会默认在句子的前后加上特殊字符,一个CLS,一个SEP
    print(f'特征提取结果:{torch.tensor(result).shape}')

# todo:3.完成完形填空任务
def dm_test_fill_mask():
    my_model = pipeline(task='fill-mask', model='./03-预训练模型/chinese-bert-wwm')
    # 调用模型
    # result = my_model('我明天去你家吃饭[MASK]')
    # print(f'完形填空结果:{result[0]}')
    input = '我明天想去[MASK]家吃[MASK]。'
    for i in range(2):
        result = my_model(input)
        if type(result[0]) == list:
            input = ''.join(result[0][0]['sequence'].split(' ')[1:-1])
        else:
            break
    print(f'完型填空结果:{result}')

# todo:4.完成阅读理解任务
def dm_test_qa():
    my_model = pipeline(task='question-answering', model='./03-预训练模型/chinese_pretrain_mrc_roberta_wwm_ext_large')
    # 准备语料
    context = '我叫张三，我是一个程序员，我喜欢打篮球。'
    questions = ['我是谁','我的爱好是什么','我是做什么的']
    result = my_model(context = context,question = questions)
    print(f'阅读理解结果:{result}')

# todo:5.完成文本摘要任务
def dm_test_summarization():
    my_model = pipeline(task='text-generation', model='./03-预训练模型/distilbart-cnn-12-6')
    # 准备语料
    text = "BERT is a transformers model pretrained on a large corpus of English data " \
           "in a self-supervised fashion. This means it was pretrained on the raw texts " \
           "only, with no humans labelling them in any way (which is why it can use lots " \
           "of publicly available data) with an automatic process to generate inputs and " \
           "labels from those texts. More precisely, it was pretrained with two objectives: Masked " \
           "language modeling (MLM): taking a sentence, the model randomly masks 15% of the " \
           "words in the input then run the entire masked sentence through the model and has " \
           "to predict the masked words. This is different from traditional recurrent neural "
    result = my_model(text)
    print(f'文本摘要结果:{result}')

# todo:6.完成ner任务
def dm_test_ner():
    my_model = pipeline(task='token-classification', model='./03-预训练模型/roberta-base-finetuned-cluener2020-chinese')
    # 调用模型
    result = my_model('我爱北京天安门,天安门上太阳升')
    print(f'ner结果:{result}')

if __name__ == '__main__':
    # dm_test_classification()
    # dm_test_feature_extraction()\
    # dm_test_fill_mask()
    # dm_test_qa()
    dm_test_summarization()
    # dm_test_ner()