# -*- encoding: utf-8 -*-
# @auther  : d2l, wangzs
# @time    : 2022-12-20
# @file    : softmax_regression_advanced.py
# @function: Softmax回归的简洁实现(Torch)


import torch
from torch import nn
from d2l import torch as d2l


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
# 初始化权重
net.apply(init_weights)
# 损失函数. reduction=none表示直接返回n个样本的loss
loss = nn.CrossEntropyLoss(reduction='none')
# 优化器
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
# 训练
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()
