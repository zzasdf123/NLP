import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def dm_label_sns_countplot():
#1.设置显示风格
    plt.style.use('fivethirtyeight')
    #2.读训练集 验证集数据
    train_data = pd.read_csv(filepath_or_buffer ='./cn_data/train.tsv',sep = '\t')
    dev_data = pd.read_csv(filepath_or_buffer ='./cn_data/dev.tsv',sep = '\t')
    #3.求数据长度列 然后求数据长度的分布
    train_data['sentence_length'] = list(map(lambda x: len(x),train_data['sentence']))
    #print(f'train_data:{train_data}')
    #4.绘制数据长度分布-柱状图
    sns.countplot(x='sentence_length',data=train_data,hue='label')
    plt.xticks([])
    plt.show()
    #5.绘制数据长度分布-曲线图
    sns.displot(x='sentence_length',data=train_data,kind='kde')
    plt.yticks()
    plt.show()
    #6.验证集数据可视化
    dev_data['sentence_length'] = list(map(lambda x:len(x),dev_data['sentence']))
    sns.countplot(x='sentence_length',data=dev_data,hue='label')
    plt.xticks([])
    plt.show()
    sns.displot(x='sentence_length',data=dev_data,kind='kde')
    plt.yticks()
    plt.show()
    #7.绘制散点图
    sns.stripplot(y='sentence_length',x='label',data=train_data,hue='label')
    plt.show()
    sns.stripplot(y='sentence_length',x='label',data=dev_data,hue='label')
    plt.show()

if __name__ == '__main__':
    dm_label_sns_countplot()
