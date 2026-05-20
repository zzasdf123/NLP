# -*- coding: utf-8 -*-
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# input = torch.randn(1,2,3)
# a = nn.ReLU()
# print(a(input))
# print(F.relu(input))
import re
# a ='what?time is it?'
# print(a.lower())
# print(a.lower().strip())
# print(re.sub(r'([.?!])',r' \1 ',a))

# b = 'abdcWDAD我爱你DFADADA'
# print(re.sub(r'[^a-zA-Z]',' ',b))
# print(re.sub(r'[^a-zA-Z]+',' ',b))
# print(re.sub(r'[^a-zA-Z]*',' ',b))

# a = {'姓名':5, '年龄':6}
# b = ['i am .', 'j ai ans .']
# for i in b[0].split(' '):
#     a[i] = len(a)
# print(a)
# print(a.items())
# print(a[0])

import torch
a = torch.randn(2,3,4)
print(a)
# print(a[:].shape)  #[2,3,4]
# print(a[1:].shape)  #[1,3,4]
# print(a[1:,:,].shape) #[1,3,4]
# print(a[1:,:2,:].shape)#[1,2,4]
# print(a[1:,:2,:])
# print(a[1:,2,:])  #[1,4]
# print(a[0,:1,3])  #[1]
# print(a[0,:,:300].shape)  #[3,4]
# print(a[:,2,:2].shape) #[2,2]
# print(a[:1,2,].shape) #[1,4]
# print(a[0,2:].shape) #[1,4]
# print(a[0][2:].shape) #[1,4]






