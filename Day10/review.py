import torch
a = torch.tensor([[2.1,5.1],
                  [3.1,4.2],])
b = torch.argmax(a,dim=-1)
print(b)
c = torch.tensor([1,0])
print(len(c))
print((b == c).sum())