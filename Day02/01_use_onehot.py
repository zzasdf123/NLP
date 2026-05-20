from tensorflow.keras.preprocessing.text import Tokenizer
import joblib
def get_onehot():
    #1.准备语料
    vocabs = {'周杰伦','陈奕迅','张学友','王力宏','林志玲','鹿晗'}
    #2.实例化分词器
    my_tokenizer = Tokenizer()
    #3.训练分词器
    my_tokenizer.fit_on_texts(vocabs)
    #4.检验训练效果
    print(my_tokenizer.word_index)
    print(my_tokenizer.index_word)
    #5.查找每个单词的onehot编码
    for vocab in vocabs:
        #初始化一个全零的列表，长度为len（vocabs）
        zero_list = [0] * len(vocabs)
        #找到当前单词对应的索引
        idx = my_tokenizer.word_index[vocab] - 1
        zero_list[idx] = 1
        print(f'{vocab}的onehot编码为：{zero_list}')
    #6.保存训练好的分词器
    my_path = './my_tokenizer'
    joblib.dump(my_tokenizer,my_path)
    print(f'保存分词器成功，保存路径为：{my_path}')

def ues_onehot():
    vocabs = {'周杰伦', '陈奕迅', '张学友', '王力宏', '林志玲', '鹿晗'}
    #1.加载训练好的分词器
    my_path = './my_tokenizer'
    my_tokenizer = joblib.load(my_path)
    #2.需要进行onehot编码的token
    token = '周杰伦'
    zero_list = [0] * len(vocabs)
    idx = my_tokenizer.word_index[token] - 1
    zero_list[idx] = 1
    print(f'{token}的onehot编码为：{zero_list}')



if __name__ == '__main__':
    get_onehot()
    ues_onehot()
