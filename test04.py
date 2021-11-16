import torch
x = torch.tensor([1.0,2,4,8])
y = torch.tensor([2,2,2,2])
print(x+y)
print(x-y)
print(x*y)
print(x/y)
print(x**y)
print(x==y)
'''保存数据集的方式'''
import os

os.makedirs(os.path.join('.', 'data'), exist_ok=True)
data_file = os.path.join('.', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')


'''从创建的csv文件中加载原始数据集'''
import pandas as pd

data = pd.read_csv(data_file)
print(data)
'''处理缺失值NaN代表缺失值'''
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]#这句话代表前两列是属于输入inputs，最后一列是输出outputs其中iloc是pandas用来切片的函数
inputs = inputs.fillna(inputs.mean())
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)#get_dummies是为了转为one-hot编码，将字符转为不同的类
print(inputs)
import torch
'''转换为张量格式'''
X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(X)
print(y)
'''reshape和view的区别
reshape的话，当b=a.reshape之后，修改b会直接改变a
'''