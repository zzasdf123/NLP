n_gram = 2
def dm01_Ngram(input_list):
    alist = [input_list[i:] for i in range(n_gram)]
    print(alist)
    return set(zip(*alist))
#句子长度规范
from keras.preprocessing import sequence
sequence_len  = 10
def padding(inputs):
    #padding = “post”在后面补齐 padding = “pre” 在前面补齐
    #truncating = “pre” 截取前面 truncating = “post” 截取后面
    return sequence.pad_sequences(inputs,maxlen=sequence_len,padding='post',truncating='post')
def my_padding(inputs):
    alist = []
    for value in inputs:
        if len(value) >= sequence_len:
            alist.append(value[:sequence_len])
        else:
            value1 = value + [0] * (sequence_len - len(value))
            alist.append(value1)
    return alist


if __name__ == '__main__':
    # input_list = ['我','是','一个','程序员']
    # result = dm01_Ngram(input_list)
    # print(result)
    x_train = [[1,23,5,32,55,63,2,21,78,32,23,1],[2,32,1,23,1]]
    result1 = padding(x_train)
    result2 = my_padding(x_train)
    print(result1,result2)