import jieba.posseg as pseg
import pandas as pd
from wordcloud import WordCloud
from itertools import chain
import matplotlib.pyplot as plt
#获取句子text中的所有形容词
def get_a_list(text):
    r = []
    for value in pseg.cut(text):
       if value.flag == 'a':
           r.append(value.word)
    return r
#画图展示词云，输入参数为：text列表形式，储存的是高频词
def get_word_cloud(keywords):
    #1.实例化词云对象
    wordcloud = WordCloud(font_path='./cn_data/SimHei.ttf',max_words=100,background_color='white')
    #2.获取展示词云的数据，是字符串形式 空格隔开
    keywords_str = ' '.join(keywords)
    #3.生成词云
    wordcloud.generate(keywords_str)
    #4.画图，控制台展示
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

def main():
    #1.读取数据
    train_data = pd.read_csv('./cn_data/train.tsv',sep='\t')
    dev_data = pd.read_csv('./cn_data/dev.tsv',sep='\t')
    #2.获取训练集中的正样本
    train_p_sentence = train_data[train_data['label'] == 1]['sentence']
    train_n_sentence = train_data[train_data['label'] == 0]['sentence']
    dev_p_sentence = dev_data[dev_data['label'] == 1]['sentence']
    dev_n_sentence = dev_data[dev_data['label'] == 0]['sentence']
    #3.获得正样本中的所有形容词
    p_train_vocabs = list(chain(*map(lambda x: get_a_list(x),train_p_sentence)))
    n_train_vocabs = list(chain(*map(lambda x: get_a_list(x),train_n_sentence)))
    p_dev_vocabs = list(chain(*map(lambda x :get_a_list(x),dev_p_sentence)))
    n_dev_vocabs = list(chain(*map(lambda x :get_a_list(x),dev_n_sentence)))
    #print(f"p_vocabs:{p_train_vocabs}")
    #print(f'p_vocabs_len:{len(p_train_vocabs)}')
    #4.调用get_word_cloud方法
    get_word_cloud(p_train_vocabs)
    get_word_cloud(n_train_vocabs)
    get_word_cloud(p_dev_vocabs)
    get_word_cloud(n_dev_vocabs)

if __name__ == '__main__':
    main()
