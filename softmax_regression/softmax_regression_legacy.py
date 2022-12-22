# -*- encoding: utf-8 -*-
# @auther  : d2l, wangzs
# @time    : 2022-12-19
# @file    : softmax_regression_legacy.py
# @function: Softmax回归的从零开始实现


import torch
import torchvision
from IPython import display
from d2l import torch as d2l
from torch.utils import data
from torchvision import transforms

batch_size = 256
num_inputs = 784  # 28*28*1
num_outputs = 10


class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4


def load_data_fashion_mnist(batch_size, resize=None):
    """
    获取和读取Fashion-MNIST数据集
    :param batch_size: 批量大小
    :param resize: 用来将图像大小调整为另一种形状
    :return: 训练集和验证集的数据迭代器
    """
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


def softmax(X):
    """
    定义softmax函数操作
    对于任何随机输入，我们将每个元素变成一个非负数。 此外，依据概率原理，每行总和为1。
    :param X: 矩阵
    :return:
    """
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)  # 按行求和
    return X_exp / partition


def net(X):
    """
    实现softmax回归模型. 定义了输入如何通过网络映射到输出
    :param X: 批量输入
    :return:
    """
    # 将数据传递到模型之前，我们使用reshape函数将每张原始图像展平为向量。
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


def cross_entropy(y_hat, y):
    """
    定义损失函数: 交叉熵
    使用特殊的运算, 利用真实标签作为索引, 从输出结果中选择对应分类的概率
    :param y_hat: 输出预测矩阵, Tensor大小: batch_size * num_outputs
    :param y: 真实标签, Tensor大小: batch_size
    :return:
    """
    return - torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y):
    """
    计算正确预测的数量
    :param y_hat: 输出预测矩阵
    :param y: 真实标签
    :return:
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y  # 一个包含0（错）和1（对）的张量
    return float(cmp.type(y.dtype).sum())  # 求和会得到正确预测的数量


def evaluate_accuracy(net, data_iter):
    """
    计算在指定数据集上模型的精度
    :param net: 网络
    :param data_iter: 数据集
    :return:
    """
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设为评估模式
    metric = Accumulator(2)  # 两个计数: 正确预测数 & 预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


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


def updater(batch_size):
    """
    指定优化器
    :param batch_size:  批量大小
    :return:
    """
    return sgd([W, b], lr, batch_size)


def train_epoch(net, train_iter, loss, updater):
    """
    训练模型一个迭代周期
    :param net: 网络
    :param train_iter: 训练集
    :param loss: 损失函数
    :param updater: 优化器
    :return:
    """
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 累加器: 训练损失总和 训练准确度总和 样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train(net, train_iter, test_iter, loss, num_epochs, updater):
    """
    训练
    :param net: 网络
    :param train_iter: 训练集
    :param test_iter: 测试集
    :param loss: 损失函数
    :param num_epochs: 迭代次数
    :param updater: 优化器
    :return:
    """
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert 1 >= train_acc > 0.7, train_acc
    assert 1 >= test_acc > 0.7, test_acc


def predict(net, test_iter, n=6):
    """
    预测
    :param net: 网络
    :param test_iter: 测试集
    :param n:
    :return:
    """
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


# 读取数据集
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# 初始化模型参数. 使用正态分布初始化我们的权重W，偏置初始化为0。
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# 测试softmax功能
# X = torch.normal(0, 1, (2, 5))
# X_prob = softmax(X)
# print(X_prob)
# print(X_prob.sum(1))

# 测试交叉熵损失函数
# y = torch.tensor([0, 2])
# y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# print(cross_entropy(y_hat, y))

lr = 0.1
num_epochs = 10
train(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

predict(net, test_iter)
d2l.plt.show()
