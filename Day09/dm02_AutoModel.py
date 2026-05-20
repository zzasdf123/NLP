# -*- coding:utf-8 -*-
import torch
from transformers import AutoModel,AutoConfig,AutoTokenizer
from transformers import AutoModelForSequenceClassification,AutoModelForMaskedLM,AutoModelForQuestionAnswering
from transformers import AutoModelForSeq2SeqLM,AutoModelForTokenClassification

#todo:1.文本分类任务
def dm_test_classification():
    # 1.加载分词器
    my_tokenizer = AutoTokenizer.from_pretrained('./03-预训练模型/chinese_sentiment')
    # 2.加载模型
    my_model = AutoModelForSequenceClassification.from_pretrained('./03-预训练模型/chinese_sentiment')
    # 3.准备语料
    sentence = '人生该如何起头'
    # 4.需要将原始的语句进行tokenizer
    # cls-->101,sep-->102
    # padding='max_length':按照最大句子长度补齐
    # truncation=True:超出最大长度的截断
    # return_tensors='pt':将结果转换成pytorch张量，使用返回二维张量，不使用返回以为列表
    # max_length:句子最大长度
    tensor_x = my_tokenizer.encode(sentence,return_tensors='pt',padding='max_length',max_length=20,truncation=True)
    # 预测试，如果用到预训练的模型，需要加上eval() 如果预训练模型中含有norm层与dropout层需加上
    my_model.eval()       #计算梯度，但不更新
    with torch.no_grad(): #不计算梯度
        # 5将上述tensor_x[1,20],送入模型
        output = my_model(tensor_x)
        # output = my_model(**tensor_x,return_dict=False)
        print(f'文本分类的结果为{output['logits']}')
        # print(f'文本分类的结果为{output.logits}')
        # 6.预测最终类别对应的索引
        idex = torch.argmax(output.logits,dim=-1)
        print(f'文本分类结果为：{idex}')

#todo:2.特征提取任务
def dm_test_feature_extraction():
    # 1.加载分词器
    my_tokenizer = AutoTokenizer.from_pretrained('./03-预训练模型/bert-base-chinese')
    # 2.加载模型
    my_model = AutoModel.from_pretrained('./03-预训练模型/bert-base-chinese')
    # 3.准备语料
    sentence = ['我是谁','人生该如何起头']
    # 4.需要将原始的语句进行tokenizer
    # cls-->101,sep-->102
    # padding='max_length':按照最大句子长度补齐
    # truncation=True:超出最大长度的截断
    # return_tensors='pt':将结果转换成pytorch张量，使用返回二维张量，不使用返回以为列表
    # max_length:句子最大长度
    tensor_x = my_tokenizer(sentence,return_tensors='pt',padding='max_length',max_length=20,truncation=True)
    print(f'tensor_x:{tensor_x}')
    # 预测试，如果用到预训练的模型，需要加上eval() 如果预训练模型中含有norm层与dropout层需加上
    my_model.eval()
    # 5将上述tensor_x[1,20],送入模型
    # output = my_model(input_ids = tensor_x['input_ids'],token_type_ids = tensor_x['token_type_ids'],attention_mask = tensor_x['attention_mask'])
    output = my_model(**tensor_x)
    print(f'特征提取结果为：{output['last_hidden_state'].shape}')
    print(f'特征提取结果为：{output.pooler_output.shape}')

#todo:3.完型填空任务
def dm_test_fill_mask():
    # 1.加载分词器
    my_tokenizer = AutoTokenizer.from_pretrained('./03-预训练模型/chinese-bert-wwm')
    # 2.加载模型
    my_model = AutoModelForMaskedLM.from_pretrained('./03-预训练模型/chinese-bert-wwm')
    # 3.准备语料
    sentence = '我明天想去[MASK]家吃饭。'
    # 4.需要将原始的语句进行tokenizer
    # cls-->101,sep-->102
    # padding='max_length':按照最大句子长度补齐
    # truncation=True:超出最大长度的截断
    # return_tensors='pt':将结果转换成pytorch张量，使用返回二维张量，不使用返回以为列表
    # max_length:句子最大长度
    tensor_x = my_tokenizer(sentence,return_tensors='pt')
    print(f'tensor_x:{tensor_x}')
    # 预测试，如果用到预训练的模型，需要加上eval() 如果预训练模型中含有norm层与dropout层需加上
    my_model.eval()
    # 5将上述tensor_x[1,12],送入模型
    # output = my_model(input_ids = tensor_x['input_ids'],token_type_ids = tensor_x['token_type_ids'],attention_mask = tensor_x['attention_mask'])
    output = my_model(**tensor_x)
    print(f'完型填空结果为：{output}')
    print(f'特征提取结果为：{output['logits'].shape}')  #[1,12,21128]需要从中找到[MASK]对应位置输出的21128个概率
    # 取出对应位置的预测概率值
    tem_vector = output['logits'][:,6]
    print(tem_vector.shape)
    idx = torch.argmax(tem_vector,dim=-1)
    print(f'完型填空结果为：{idx}')
    print(my_tokenizer.convert_ids_to_tokens(idx))

#todo:4.阅读理解任务
def dm_test_qa():
    # 1.加载分词器
    my_tokenizer = AutoTokenizer.from_pretrained('./03-预训练模型/chinese_pretrain_mrc_roberta_wwm_ext_large')
    # 2.加载模型
    my_model = AutoModelForQuestionAnswering.from_pretrained('./03-预训练模型/chinese_pretrain_mrc_roberta_wwm_ext_large')
    # 3.准备语料
    context = '我叫张三 我是一个程序员 我喜欢打篮球。'
    questions = ['我是谁?', '我的爱好是什么?', '我是做什么的?']
    my_model.eval()
    # 4.需要一个问题一个问题进行解答
    for question in questions:
        tensor_x = my_tokenizer._encode_plus(question,context,return_tensors='pt')
        # print(f'tensor_x:{tensor_x}')
        # print(f'input_ids:{tensor_x['input_ids'].shape}')
        # 5将上述tensor_x送入模型
        output = my_model(**tensor_x)
        # print(f'阅读理解结果为：{output}')
        # print(f'start_logits:{output['start_logits'].shape}')
        # 找到问题对应答案开始与结束的索引
        start_idx = torch.argmax(output['start_logits'],dim=-1)
        end_idx = torch.argmax(output['end_logits'],dim=-1)
        answer_idx = tensor_x['input_ids'][0][start_idx:end_idx+1]
        answer =''.join(my_tokenizer.convert_ids_to_tokens(answer_idx))
        print(f'阅读理解结果为：{answer}')

#todo:5.文本摘要任务
def dm_test_summarization():
    # 1.加载分词器
    my_tokenizer = AutoTokenizer.from_pretrained('./03-预训练模型/distilbart-cnn-12-6')
    # 2.加载模型
    my_model = AutoModelForSeq2SeqLM.from_pretrained('./03-预训练模型/distilbart-cnn-12-6')
    # 3.准备语料
    text = "BERT is a transformers model pretrained on a large corpus of English data " \
           "in a self-supervised fashion. This means it was pretrained on the raw texts " \
           "only, with no humans labelling them in any way (which is why it can use lots " \
           "of publicly available data) with an automatic process to generate inputs and " \
           "labels from those texts. More precisely, it was pretrained with two objectives: Masked " \
           "language modeling (MLM): taking a sentence, the model randomly masks 15% of the " \
           "words in the input then run the entire masked sentence through the model and has " \
           "to predict the masked words. This is different from traditional recurrent neural " \
           "networks (RNNs) that usually see the words one after the other, or from autoregressive " \
           "models like GPT which internally mask the future tokens. It allows the model to learn "
    # 4.需要将原始的语句进行tokenizer处理
    my_model.eval()
    input = my_tokenizer._encode_plus(text,return_tensors='pt')
    # 5将上述input送入模型
    output = my_model.generate(**input)
    # output = my_model.generate(input['input_ids'])   # 由于没有进行截断或者补齐操作，故不进行掩码操作
    # skip_special_tokens=True 去除特殊的字符
    # clean_up_tokenization_spaces=False 标点符号与单词分隔开
    outputs =my_tokenizer.decode(output[0],skip_special_tokens=True,clean_up_tokenization_spaces=False)
    print(f'文本摘要结果为：{outputs}')

# todo:6.NER任务
def dm_test_ner():
    # 1.加载分词器
    my_tokenizer = AutoTokenizer.from_pretrained("./03-预训练模型/roberta-base-finetuned-cluener2020-chinese")
    # 2.加载模型
    my_model = AutoModelForTokenClassification.from_pretrained("./03-预训练模型/roberta-base-finetuned-cluener2020-chinese")
    # 3.加载该模型的配置文件
    my_config = AutoConfig.from_pretrained("./03-预训练模型/roberta-base-finetuned-cluener2020-chinese")
    print(f"my_confing:{my_config}")
    # 4.基于my_tokenizer进行分词处理文本
    inputs = my_tokenizer._encode_plus("鲁迅先生的代表作是《朝花夕拾》",return_tensors='pt')
    # print(f'inputs:{inputs}')
    # 5将上述inputs送入模型
    my_model.eval()
    outputs = my_model(**inputs)
    # print(f' ner结果为：{outputs.logits.shape}')
    # 6.将inputs_ids对齐原来的token
    original_tokens = my_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    # print(f"original_tokens是{original_tokens}")
    # print(f"all_special_tokens:{my_tokenizer.all_special_tokens}")
    # 7.创建一个outputs_list来存储结果
    outputs_list = []
    # 8.遍历outputs
    for token,value in zip(original_tokens,outputs.logits[0]):
        print(f"token:{token}")
        print(f"value:{value.shape}")
        if token in my_tokenizer.all_special_tokens:
            continue
        idx = torch.argmax(value,dim=-1).item()
        outputs_list.append((token,my_config.id2label[idx]))
    print(f"outputs_list:{outputs_list}")

if __name__ == '__main__':
    # dm_test_classification()
    # dm_test_feature_extraction()
    # dm_test_fill_mask()
    # dm_test_qa()
    # dm_test_summarization()
    dm_test_ner()