import jieba
from tensorboard import summary
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch

def dm01_embedding_show():
    #1.定义语料
    sentence1 = '传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能'
    sentence2 = '我爱自然语言处理'
    sentences = [sentence1,sentence2]
    #2.对所有的句子进行分此次
    word_list = []
    for sentence in sentences:
        word_list.append(jieba.lcut(sentence))
    #print(f'分词结果：{word_list}')
    #3.获得word_index,index_word
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(word_list)
    #print(f'tokenizer.word_index:{tokenizer.word_index}')
    #print(f'tokenizer.index_word:{tokenizer.index_word}')
    #4.将文本序列转换为数字序列
    seq_idx = tokenizer.texts_to_sequences(word_list)
    #print(f'分词索引：{seq_idx}')
    #5.获取样本中所有单词
    words1 = tokenizer.word_index.keys()
    words2 = tokenizer.index_word.values()
    #print(f'word:{words1}')
    #print(f'word:{words2}')
    #6.实例化Embedding层
    #参1：需要进行词向量表示的单词数量 （去重后）参2：每个单词嵌入维度
    embed = nn.Embedding(num_embeddings=len(words1),embedding_dim=8)
    print(f'embed.weight:{embed.weight}')
    print(f'embed.weight.shape:{embed.weight.shape}')
    #7.可视化Embedding层
    #summary.add_embedding(embed.weight.data,words1)
    #summary.close()
    #cd程序的当前目录下执行下面的命令
    #启动tensorboard服务 tensorboard --logdir=runs --host 0.0.0.0#通过浏览器，查看词向量可视化效果http://127.0.0.1:6006
    #8.获取每个丹迪对应的词向量
    for idx in range(len(tokenizer.word_index)):
        output = embed(torch.tensor([idx]))
        #print(f'output:{output}')
        #output = embed.forward(torch.tensor[idx])  # 结果相同
        print(f'{tokenizer.index_word[idx+1]}:{output}')
        print('*' * 50)
if __name__ == '__main__':
    dm01_embedding_show()

