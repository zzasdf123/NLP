# -*- coding:utf-8 -*-
import torch
from transformers import BertTokenizer,BertForMaskedLM
def dm_test_fill_mask():
    # 1.加载分词器
    my_tokenizer = BertTokenizer.from_pretrained('./03-预训练模型/chinese-bert-wwm')
    # 2.加载模型
    my_model = BertForMaskedLM.from_pretrained('./03-预训练模型/chinese-bert-wwm')
    # 3.准备语料
    sentence = "我喜欢吃蓝莓，我也给你[MASK]一包蓝莓"
    # 4.将原始进行处理
    input = my_tokenizer._encode_plus(sentence,return_tensors="pt")
    # 5.输出模型
    my_model.eval()
    output = my_model(**input)
    print(output.logits.shape)
    idx = torch.argmax(output.logits[:, 12],dim=-1)
    print(f"完型填空结果为{my_tokenizer.convert_ids_to_tokens(idx)}")

if __name__ == '__main__':
    dm_test_fill_mask()