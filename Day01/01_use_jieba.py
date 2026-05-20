#coding:utf-8
import jieba
import jieba.posseg as pseg
#todo：1.jieba实现精确模式的分词
def dm1_jieba():
    #1.给出需要被切分的文本
    content = "小明硕士毕业于中国科学院计算所，后在日本京都大学深造"
    #2.基于jieba.cut()方法实现文本的切分  #返回是生成器，取出元素的方法：for循环，next（），强转list（）等
    #result1 = jieba.cut(content,cut_all=False)
    #print(f'result1:{result1}')
    #3.基于jieba.lcut()实现文本切分
    result2 = jieba.lcut(content,cut_all=False)
    print(f'result2:{result2}')
#todo：2.jieba实现全模式模式的分词
def dm2_jieba():
    #1.给出需要被切分的文本
    content = "小明硕士毕业于中国科学院计算所，后在日本京都大学深造"
    #2.基于jieba.cut()方法实现文本的切分  #返回是生成器，取出元素的方法：for循环，next（），强转list（）等
    # result1 = jieba.cut(content,cut_all=True)
    # print(f'result1:{result1}')
    #3.基于jieba.lcut()实现文本切分
    result2 = jieba.lcut(content,cut_all=True)
    print(f'result2:{result2}')
#todo：3.jieba实现搜索引擎模式的分词
def dm3_jieba():
    #1.给出需要被切分的文本
    content = "小明硕士毕业于中国科学院计算所，后在日本京都大学深造"
    #2.基于jieba.cut_for_search()方法实现文本的切分  #返回是生成器，取出元素的方法：for循环，next（），强转list（）等
    #result1 = jieba.cut_for_search(content)
    #print(f'result1:{list(result1)}')
    #3.基于jieba.lcut_for_search()实现文本切分
    result2 = jieba.lcut_for_search(content)
    print(f'result2:{result2}')
#todo：4.jieba支持自定义词典：如果加载了自定义词典，那么分词会优先按照词典里面的词进行分词，但是要考虑词频
#词典的格式：每行样本：word fred cixing
def dm4_jieba():
    #未加入词典分词
    content = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"
    result1 = jieba.lcut(content)
    print(f'result1:{result1}')
    #加入词典分词
    jieba.load_userdict('./userdict.txt')
    result2 = jieba.lcut(content)
    print(f'result2:{result2}')
#todo：5.jieba支持词性标注
def dm5_jieba():
    content = "小明硕士毕业于中国科学院计算所，后在日本京都大学深造"
    result = pseg.lcut(content)
    print(f'result:{result}')



if __name__ == '__main__':
    #dm1_jieba()
    #dm2_jieba()
    #dm3_jieba()
    #dm4_jieba()
    dm5_jieba()
