# -*- encoding: utf-8 -*-
# @auther  : d2l, wangzs
# @time    : 2022-12-14
# @file    : linear_regression_advanced.py
# @function: 线性回归的简洁实现（PyTorch）


import torch
from torch import nn
from torch.utils import data


def create_samples(w, b, num_examples):
    """
    人工生成测试数据集(带标签): y=Xw+b+噪声
    :param w: 线性模型参数
    :param b: 偏置项
    :param num_examples: 数据集中的总样本数
    :return:
    """
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b  # 加上偏置项
    y += torch.normal(0, 0.01, y.shape)  # 加上噪声项
    return X, y.reshape((-1, 1))


def load_array(data_arrays, batch_size, is_train=True):
    """
    构造一个PyTorch数据迭代器
    :param data_arrays: 数据集
    :param batch_size: 批量大小
    :param is_train: 训练数据要随机打乱
    :return:
    """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# 生成测试数据. features中的每一行都包含一个二维数据样本, labels中的每一行都包含一维标签值(标量)
true_w = torch.tensor([2, -3.4])
true_b = 4.2
num = 1000
print(f'正在生成数据集. 含{num}个样本.')
features, labels = create_samples(true_w, true_b, num)
print(f'数据集已生成. 第一个样本为:\nfeatures: {features[0]}\nlabel: {labels[0]}')
# d2l.set_figsize()
# d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
# d2l.plt.show()

# 读取数据集，构造批量迭代器
batch_size = 10
data_iter = load_array((features, labels), batch_size)
# print(next(iter(data_iter)))  # 使用iter构造Python迭代器, 并使用next从迭代器中获取第一项

# 定义模型
net = nn.Sequential(nn.Linear(2, 1))

# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 定义损失函数
loss = nn.MSELoss()  # 计算均方误差使用的是MESLoss类,也称为平方L2范数. 默认情况下,返回所有样本损失的平均值

# 定义优化算法
# 通过net.parameters从模型中获得要优化的参数以及优化算法所需的超参数字典. 小批量随机梯度下降SGD只需要设置学习率lr
trainer = torch.optim.SGD(net.parameters(), lr=0.05)

# 模型训练
print('\n开始训练.')
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()  # pytorch不会自动将梯度归零，
        l.backward()  # 反向传播，得到每个参数的梯度
        trainer.step()  # 执行一次优化步骤.
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

print('\n训练完成.')
w = net[0].weight.data
print('w的估计误差: ', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差: ', true_b - b)
