import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def dm_label_sns_countplot():
#1.设置显示风格
    plt.style.use('fivethirtyeight')
    #2.读训练集 验证集数据
    train_data = pd.read_csv(filepath_or_buffer ='./cn_data/train.tsv',sep = '\t')
    dev_data = pd.read_csv(filepath_or_buffer ='./cn_data/dev.tsv',sep = '\t')
    # 3.统计label标签的0,1分组数量
    sns.countplot(x='label',data=train_data,hue='label')
    plt.title('train_label')
    plt.show()
    # 验证集上标签的数量分布
    #sns.countplot(x='label',data=dev_data,hue='label')
    sns.countplot(x=dev_data['label'],hue=dev_data['label'])
    plt.title('dev_label')
    plt.show()

if __name__ == '__main__':
    dm_label_sns_countplot()
