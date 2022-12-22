# -*- encoding: utf-8 -*-
# @auther  : d2l, wangzs
# @time    : 2022-12-13
# @file    : linear_regression_legacy.py
# @function: 线性回归的从零开始实现

import random
import torch


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


def data_iter(batch_size, features, labels):
    """
    读取数据集. 生成大小为batch_size的小批量. 每个小批量包含一组特征和标签.
    :param batch_size: 批量样本的大小
    :param features: 特征矩阵
    :param labels: 标签向量
    :return:
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本应随机读取, 无特定顺序
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def linreg(X, w, b):
    """
    定义模型
    :param X: 输入特征
    :param w: 模型权重
    :param b: 偏置
    :return: 线性模型的输出
    """
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """
    定义损失函数: 平方损失
    :param y_hat: 预测值
    :param y: 真实值
    :return: 平方损失
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """
    小批量随机梯度下降
    :param params: 要学习的参数
    :param lr: 学习率
    :param batch_size: 批量大小
    :return:
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


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

# 设置超参数
lr = 0.03  # 学习率
num_epochs = 5  # 迭代周期个数
net = linreg  # 学习算法
loss = squared_loss  # 损失函数
batch_size = 10  # 批量大小

# 初始化参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 迭代训练
print('\n开始训练...')
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        l.sum().backward()  # l的形状是(batch_size,1), 而不是一个标量. l中所有元素被加到一起, 并以此计算关于[w,b]的梯度.
        sgd([w, b], lr, batch_size)  # 使用梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print('\n训练结束.')
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
