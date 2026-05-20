import fasttext
def dm01_use_fasttext():
    #1.直接调用无监督训练方法训练词向量模型
    #model = fasttext.train_unsupervised('./fil9',epoch=1)
    #2.保存模型
    #model_path = 'ai23_fil9.bin'
    #model.save_model(model_path)
    #3.加载模型，查看某个单词的向量
     model = fasttext.load_model('./ai23_fil9.bin')
     result = model.get_word_vector('the')
     print(f'result.type:{type(result)}')
     print(f'result.shape:{result.shape}')
    # #4.查看模型效果
    #print(model.get_nearest_neighbors('sports'))
    #5.修改训练模型的参数
    #my_model = fasttext.train_unsupervised('./fil9', model='cbow',epoch=1,lr=0.1,dim=100)
if __name__ == '__main__':
    dm01_use_fasttext()

