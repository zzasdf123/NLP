#map:第一个参数：fun--》代表函数对象，第二个参数：iterable代表可迭代对象
#map的作用：将可迭代对象中的每一个元素都经过func的处理，得到新的map对象（迭代器）
#可迭代对象：list,tuple,str,dict,set,file 凡是能用for循环取出数据的对象，就是可迭代对象
# def fun(a):
#     return a+1
# lambda a: a+1
# fun = lambda a: a+1
# list1 = [1,2,3,4,5]
# b = map(fun,list1)
# print(list(b))

# from itertools import chain
# list1 = [1,2,3,4,5]
# list2 = [5,7,8,9,10]
# a = chain(list1,list2)
# print(list(a))
# a = list1 + list2
# print(a)
# print(set(a))
# list1.extend(list2)
# print(list1)
# b = map(lambda x:[len(x)],['nihao','haha','tes'])
# c = chain(*b)
# print(list(c))

# def fun(a):
#     return a+6

# def fun1(c):
#     return fun(c) + 2
#
# print(fun1(4))
# a = lambda c: fun(c) + 2
# map(a,[1,2,3,4,5])
# b = map(a,[1,2,3,4,5])
# print(list(b))

# list1 = [1,2,3,5,0,0]
# list2 = [2,3,4,0,0,0]
# list3 = [1,2,3,4,5,6]
# a = zip(list1,list2,list3)
# b,c,d = zip(*a)
# print(b,c,d)

# list1 = [1,2,3,5]
# list2 = [2,3,4]
# list3 = [1,2,3,4,5,6]
# a = zip(list1,list2,list3)
# b,c,d = zip(*a)
# print(b,c,d)
# alist = [[1,2,3,5],[2,3,4],[1,2,3,4,5,6]]
# b = zip(*alist)
# print(list(b))
input_list = [1,2,3,1,5,2]
print([input_list[i:] for i in range(2)])
c = zip(*[input_list[i:] for i in range(2)])
print(list(c))