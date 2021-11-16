import torch

x = torch.tensor([3.0])
y = torch.tensor([2.0])

print(x + y, x * y, x / y, x**y)
'''向量'''
x = torch.arange(4,dtype = torch.float32)
print(x)
'''向量的点积（dot，product）'''
y = torch.ones(4, dtype = torch.float32)
print(x)
print(y)
k = torch.dot(x,y)
print(k)
'''
关于按照某一轴进行求和时
a的shape为[5,4]
按照axis=0进行求和之后，维度为4
按照axis=1进行求和之后，维度为5
'''
import torch
a = torch.ones((2,5,4))
print(a.shape) #torch.Size([2, 5, 4])
print(a.sum(axis=1).shape) #torch.Size([2, 4])
print(a.sum(axis=1,keepdim=True).shape) #torch.Size([2, 1, 4])
