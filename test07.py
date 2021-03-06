'''自动求导'''
import torch

x = torch.arange(4.0)
print(x)
x.requires_grad_(True)  # 等价于 `x = torch.arange(4.0, requires_grad=True)`
print(x.grad)  # 默认值是None
y = 2 * torch.dot(x, x)
print(y)
y.backward()
print(x.grad)

x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

x.grad.zero_()
y = x*x
y.sum().backward()
print(x.grad)

#分离计算
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
print(x.grad)
#python控制流的梯度计算
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
a.grad == d / a
print(a)