# import string
# all_letters = string.ascii_letters + " ,;.'"
# print(f'all_letters:{all_letters}')
# print(all_letters.find('A'))
#
# list = ['zz','bb','cc']
# print(list.index('cc'))
# import torch
# import torch.nn as nn
# a = torch.tensor([[1.0,1.0,1.0],[2.0,2.0,2.0]])
# c = nn.Softmax(dim=-1)
# print(c(a))
# print('%04d你好' % (1))
# print('%.3f你好' % (1))

import json as json
# dict1 = {'name':'luck',
#     'age':18,
#     'sex':'nan'}
# dict1_str = json.dumps(dict1)
# # print(f'dict1_str:{dict1_str}')
# # print(f'type:{type(dict1_str)}')
# # print(type(dict1))
# with open('a.json','a') as fr:
#     fr.write(dict1_str)

# with open('a.json','r') as fr:
#     b =fr.readlines()
#     print(b)
#     print(type(b))
#     print(type(b[0]))
#     c = json.loads(b[0])
#     print(c)
#     print(type(c))
# dict1 = {'loss':[1.0,2.0,3.0],
#          'acc':[3.6,4.8,6.7]}
# dict1_str = json.dumps(dict1)
# print(dict1_str)
# print(type(dict1_str))
# with open('a.json','a') as fw:
#     fw.write(dict1_str)
# with open('a.json','r') as fr:
#     b =fr.readlines()
#     print(b[0])
#     print(type(b[0]))
#     c = json.loads(b[0])
#     print(c['loss'])
import torch
a = torch.tensor([[5.1,2.1,3.1],
                [2.1,3.1,4.1]])
topk,topi = torch.topk(a,k=1)
print(topk)
print(topi)
topk,topi = torch.topk(a,k=2)
print(topk)
print(topi)

